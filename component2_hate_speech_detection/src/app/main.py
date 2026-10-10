"""C2 Analyst desktop app: a window and a tray icon around the agent.

    python -m component2_hate_speech_detection.src.app.main [--age 10] [--no-audio] [--paused]

The window is a native Windows window (pywebview on Edge WebView2). The page is
passed as an HTML string rather than a file path, so pywebview starts no local
web server: the app opens no network port at all, which the window itself shows
live (proposal NFR4). Closing the window keeps protection running in the tray;
the tray icon stays visible, because a child-safety tool must never run hidden
(Guardian §10.4).
"""
from __future__ import annotations

import os

# Zero network (proposal NFR4), set before any library that could reach out loads.
# Measured on 10 Oct 2026 before these were set: the Whisper loader contacted the
# Hugging Face CDN, and the WebView2 engine contacted Microsoft services and opened
# a UDP socket. The page is a local HTML string, so the engine needs no network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", " ".join([
    "--disable-background-networking", "--disable-component-update", "--disable-sync", "--no-pings",
    "--disable-domain-reliability", "--disable-client-side-phishing-detection", "--disable-breakpad",
    "--dns-prefetch-disable", "--disable-default-apps", "--no-default-browser-check",
    "--disable-features=ElasticOverscroll,msSmartScreenProtection,OptimizationHints,MediaRouter,"
    "DialMediaRouteProvider,NetworkTimeServiceQuerying,AutofillServerCommunication,"
    "WebRtcHideLocalIpsWithMdns,msEdgeSidebarV2",
]))

import argparse
import base64
import ctypes
import io
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
import psutil
from PIL import Image, ImageDraw

from ..agent.runtime import Agent, AgentConfig
from ..capture.sources import LoopbackAudioSource, default_screen_source
from ..engine.decide import persona_for_age
from ..engine.verdict import Verdict

UI_FILE = Path(__file__).resolve().parent / "ui" / "index.html"
TITLE = "C2 Analyst - on-device harmful content detection"


def verdict_view(v: Verdict) -> dict:
    """What the panel shows for one decision: the record, its action and the C3 payload."""
    view = v.to_record()
    view["recommended_action"] = v.recommended_action
    view["trigger_payload"] = v.to_trigger_payload()
    view["clock"] = time.strftime("%H:%M:%S")
    return view


def _connections(processes) -> int:
    total = 0
    for proc in processes:
        try:
            total += len(proc.net_connections(kind="inet"))
        except Exception:
            pass
    return total


def wav_bytes_to_audio(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    audio = (audio - 128.0) / 128.0 if width == 1 else audio / float(np.iinfo(dtype).max)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        n = int(len(audio) * 16000 / rate)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


class Api:
    """Methods the page calls through window.pywebview.api.*; everything returned is JSON."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self._feed: Deque[dict] = deque(maxlen=100)
        self._seq = 0
        self._lock = threading.Lock()
        self._proc = psutil.Process()
        agent.on_alert = self._on_alert

    def _on_alert(self, v: Verdict) -> None:
        with self._lock:
            self._seq += 1
            view = verdict_view(v)
            view["seq"] = self._seq
            self._feed.appendleft(view)

    def status(self) -> dict:
        """Snapshot plus live network counts, reported separately and honestly.

        `engine` is this process: capture, OCR, speech, models and decisions.
        `window` is the WebView2 browser engine drawing this page. Measured on
        10 Oct 2026: the engine opens no connections; WebView2 keeps one to a
        Microsoft address for its own runtime services even with SmartScreen,
        field trials and background networking switched off. The page is a
        local string, so no content reaches it.
        """
        snap = self._agent.snapshot()
        snap["network"] = {"engine": _connections([self._proc]),
                           "window": _connections(self._proc.children(recursive=True))}
        snap["persona"] = persona_for_age(self._agent.config.age)
        return snap

    def feed(self, after: int = 0) -> List[dict]:
        with self._lock:
            return [v for v in self._feed if v["seq"] > after]

    def start(self) -> str:
        self._agent.start()
        return self._agent.state

    def stop(self) -> str:
        self._agent.stop()
        return self._agent.state

    def set_age(self, age: int) -> str:
        self._agent.set_age(int(age))
        return persona_for_age(int(age))

    def check_text(self, text: str) -> dict:
        return verdict_view(self._agent.check_text(text or ""))

    def check_file(self, name: str, data_b64: str) -> List[dict]:
        raw = base64.b64decode(data_b64.split(",", 1)[-1])
        suffix = Path(name).suffix.lower()
        if suffix == ".wav":
            return [verdict_view(v) for v in self._agent.check_audio(wav_bytes_to_audio(raw))]
        if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            with Image.open(io.BytesIO(raw)) as img:
                return [verdict_view(self._agent.check_image(img.convert("RGB")))]
        raise ValueError("Drop a .png/.jpg screenshot or a .wav clip")


WDA_EXCLUDEFROMCAPTURE = 0x11


def exclude_from_capture(window) -> bool:
    """Hide the app's own window from screen capture (Guardian FR-C1-02, threat T8).

    Without this the agent read its own window: the live counters change every
    second, so every screen check became a re-read (measured: 0% of checks
    skipped). Windows 10 2004+ removes the window from every capture path.
    """
    try:
        hwnd = int(window.native.Handle.ToInt64())
        return bool(ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))
    except Exception:
        return False


def tray_icon_image(alert: bool = False) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    colour = (220, 60, 60, 255) if alert else (38, 132, 255, 255)
    d.polygon([(32, 4), (58, 14), (54, 40), (32, 60), (10, 40), (6, 14)], fill=colour)
    d.line([(20, 33), (29, 42), (45, 22)], fill=(255, 255, 255, 255), width=6)
    return img


def run(age: int = 10, audio: bool = True, start_paused: bool = False, agent: Optional[Agent] = None) -> None:
    import pystray
    import webview

    agent = agent or Agent(
        config=AgentConfig(age=age, audio=audio),
        screen_source=default_screen_source(),
        audio_source=LoopbackAudioSource() if audio else None,
    )
    api = Api(agent)
    window = webview.create_window(TITLE, html=UI_FILE.read_text(encoding="utf-8"), js_api=api,
                                   width=1320, height=860, min_size=(1000, 660))
    quitting = threading.Event()

    def on_closing():
        if quitting.is_set():
            return True
        window.hide()  # keep protecting from the tray
        return False

    window.events.closing += on_closing
    window.events.shown += lambda: exclude_from_capture(window)

    def toggle(icon, _item):
        agent.stop() if agent.state != "stopped" else agent.start()

    def show(icon, _item):
        window.show()
        window.restore()

    def quit_app(icon, _item):
        quitting.set()
        icon.stop()
        agent.stop()
        window.destroy()

    tray = pystray.Icon(
        "c2-analyst", tray_icon_image(), "C2 Analyst: protecting",
        menu=pystray.Menu(
            pystray.MenuItem("Open C2 Analyst", show, default=True),
            pystray.MenuItem(lambda _i: "Pause protection" if agent.state != "stopped" else "Resume protection",
                             toggle),
            pystray.MenuItem("Quit", quit_app),
        ),
    )
    tray.run_detached()
    if not start_paused:
        agent.start()
    try:
        webview.start(gui="edgechromium", private_mode=True)
    finally:
        quitting.set()
        agent.stop()
        tray.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--age", type=int, default=10, help="child's age, 8-15 (sets the persona)")
    parser.add_argument("--no-audio", action="store_true", help="monitor the screen only")
    parser.add_argument("--paused", action="store_true", help="open without starting protection")
    args = parser.parse_args()
    run(age=args.age, audio=not args.no_audio, start_paused=args.paused)


if __name__ == "__main__":
    main()

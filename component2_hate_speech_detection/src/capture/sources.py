"""Input adapters: where C2's screen frames and audio come from (proposal stage 1, Table 4.2).

Capture is Component 1's job. C2 depends only on two small interfaces, so the
source can change without touching the analysis:

    ScreenSource.grab()  -> PIL image of the primary screen, or None
    AudioSource          -> pushes float32 16 kHz mono chunks to a callback

Screen: C2 uses C1's own `capture_frame()` when the C1 package is importable,
and falls back to an equivalent mss grab otherwise, so the component also runs
on its own. Audio: C1 does not capture audio yet, so C2 ships a temporary
loopback source (what the speakers play: game voice chat, videos), clearly
named as a development stand-in until C1 provides one.

Frames and audio are only ever held in memory (proposal NFR5).
"""
from __future__ import annotations

import threading
from typing import Callable, Optional, Protocol

import numpy as np
from PIL import Image

SAMPLE_RATE = 16000


class ScreenSource(Protocol):
    name: str

    def grab(self) -> Optional[Image.Image]: ...


class C1ScreenSource:
    """Component 1's capture_frame(), primary monitor."""

    def __init__(self) -> None:
        from component1_screen_monitoring.src.capture import capture_frame

        self._capture = capture_frame
        self.name = "component1.capture_frame"
        self.last_error = ""

    def grab(self) -> Optional[Image.Image]:
        try:
            return self._capture(monitor_index=1, source="c2").image
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            return None


class MssScreenSource:
    """Stand-in used only when the C1 package is not importable."""

    def __init__(self) -> None:
        import mss

        self._mss = mss.MSS()
        self.name = "mss (C1 stand-in)"
        self.last_error = ""

    def grab(self) -> Optional[Image.Image]:
        try:
            shot = self._mss.grab(self._mss.monitors[1])
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            return None


def default_screen_source() -> ScreenSource:
    try:
        return C1ScreenSource()
    except Exception:
        return MssScreenSource()


class AudioSource(Protocol):
    name: str

    def start(self, on_chunk: Callable[[np.ndarray], None]) -> bool: ...

    def stop(self) -> None: ...


class LoopbackAudioSource:
    """System audio via WASAPI loopback (soundcard). Development stand-in until C1 captures audio."""

    BLOCK_S = 0.25

    def __init__(self) -> None:
        self.name = "wasapi-loopback (C1 stand-in)"
        self.last_error = ""
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.running = False

    def start(self, on_chunk: Callable[[np.ndarray], None]) -> bool:
        if self.running:
            return True
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._loop, args=(on_chunk,), name="c2-loopback", daemon=True)
        self._thread.start()
        # Wait for the device to open, so a dead device is reported, not hidden.
        self._ready.wait(timeout=8.0)
        return self.running

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.running = False

    def _loop(self, on_chunk: Callable[[np.ndarray], None]) -> None:
        try:
            import soundcard as sc

            speaker = sc.default_speaker()
            mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            block = int(SAMPLE_RATE * self.BLOCK_S)
            with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
                self.running = True
                self._ready.set()
                while not self._stop.is_set():
                    data = np.asarray(rec.record(numframes=block), dtype=np.float32)
                    on_chunk(data.mean(axis=1) if data.ndim > 1 else data.reshape(-1))
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
        finally:
            self.running = False
            self._ready.set()

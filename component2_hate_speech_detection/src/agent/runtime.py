"""The C2 agent: runs the readers and the engine on a child's laptop (proposal NFR1-NFR8).

Two loops share one engine:

    screen   every 2 s: grab a frame -> change-gated OCR -> analyze -> alert
    audio    continuous: loopback chunks -> utterances -> Whisper -> analyze -> alert

Behaving like a good guest on the child's device:
- Below-normal process priority and 2 threads per model, so the child's game
  or homework always wins the CPU.
- Nothing is captured while the session is locked.
- Checks slow to every 6 s while Windows battery saver is on.
- Models load in the background at start, not on the first harmful message.
- A harmful message that stays on screen alerts once (verdict fingerprint +
  cooldown), not every two seconds.
- Frames and audio stay in memory and are released after each check (NFR5).

Every count and timing a panel would ask about is kept in `snapshot()`.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

import numpy as np
import psutil
from PIL import Image

from ..capture.sources import AudioSource, ScreenSource
from ..capture.system import battery_saver_on, foreground_app, session_locked
from ..engine.analyzer import Analyzer, Observation, TextSpan
from ..engine.decide import persona_for_age
from ..engine.verdict import Verdict
from ..readers.screen import ScreenReader
from ..readers.speech import SpeechReader


@dataclass
class AgentConfig:
    age: int = 10
    screen: bool = True
    audio: bool = True
    screen_interval_s: float = 2.0
    battery_interval_s: float = 6.0
    alert_cooldown_s: float = 300.0
    low_priority: bool = True


class Rolling:
    """Last N timings, with percentiles for the panel."""

    def __init__(self, n: int = 200) -> None:
        self._v: Deque[float] = deque(maxlen=n)

    def add(self, value: float) -> None:
        self._v.append(float(value))

    def summary(self) -> Dict[str, float]:
        if not self._v:
            return {"n": 0}
        arr = np.asarray(self._v)
        return {"n": len(arr), "p50": round(float(np.percentile(arr, 50)), 1),
                "p95": round(float(np.percentile(arr, 95)), 1), "max": round(float(arr.max()), 1)}


@dataclass
class AgentStats:
    counts: Dict[str, int] = field(default_factory=lambda: {
        "screen_checks": 0, "unchanged": 0, "partial_reads": 0, "full_reads": 0, "ocr_crops": 0,
        "capture_failures": 0, "paused_locked": 0, "utterances": 0, "transcribed": 0, "silent_after_vad": 0,
        "audio_dropped": 0, "verdicts": 0, "alerts": 0, "repeats_suppressed": 0,
    })
    timings: Dict[str, Rolling] = field(default_factory=lambda: {
        k: Rolling() for k in ("screen_check_ms", "diff_ms", "ocr_ms", "score_ms", "asr_ms", "audio_verdict_ms")
    })

    def inc(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n


class Agent:
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        analyzer: Optional[Analyzer] = None,
        screen_source: Optional[ScreenSource] = None,
        audio_source: Optional[AudioSource] = None,
        screen_reader: Optional[ScreenReader] = None,
        speech_reader: Optional[SpeechReader] = None,
        context: Callable[[], Dict[str, str]] = foreground_app,
        locked: Callable[[], bool] = session_locked,
        battery_saver: Callable[[], bool] = battery_saver_on,
        on_alert: Optional[Callable[[Verdict], None]] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.analyzer = analyzer or Analyzer()
        self.screen_source = screen_source
        self.audio_source = audio_source
        self.screen_reader = screen_reader or ScreenReader()
        self.speech_reader = speech_reader or SpeechReader()
        self.context, self.locked, self.battery_saver = context, locked, battery_saver
        self.on_alert = on_alert
        self.stats = AgentStats()
        self.alerts: Deque[dict] = deque(maxlen=200)
        self.warm = False
        self.state = "stopped"
        self._recent: Dict[str, float] = {}
        self._ocr_lock = threading.Lock()
        self._utterances: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._started_at = 0.0
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)

    # -- lifecycle --------------------------------------------------------------

    def start(self) -> None:
        if self.state == "protecting":
            return
        if self.config.low_priority:
            try:
                self._proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            except Exception:
                pass
        self._stop.clear()
        self._started_at = time.time()
        self.state = "protecting"
        self._spawn(self._warm_up, "c2-warmup")
        if self.config.screen and self.screen_source is not None:
            self._spawn(self._screen_loop, "c2-screen")
        if self.config.audio and self.audio_source is not None:
            self._spawn(self._audio_worker, "c2-asr")
            if not self.audio_source.start(self.feed_audio):
                self.state = "degraded"

    def stop(self) -> None:
        self._stop.set()
        if self.audio_source is not None:
            self.audio_source.stop()
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads = []
        self.screen_reader.reset()
        self.state = "stopped"

    def set_age(self, age: int) -> None:
        self.config.age = int(age)

    def _spawn(self, target, name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _warm_up(self) -> None:
        self.analyzer.scorer.warm_up()
        if self.config.audio:
            self.speech_reader.asr.transcribe(np.zeros(16000, dtype=np.float32))
        self.warm = True

    # -- screen -------------------------------------------------------------------

    def _screen_loop(self) -> None:
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                self.screen_tick()
            except Exception:
                self.stats.inc("capture_failures")
            interval = self.config.battery_interval_s if self.battery_saver() else self.config.screen_interval_s
            self._stop.wait(max(0.1, interval - (time.perf_counter() - started)))

    def screen_tick(self) -> Optional[Verdict]:
        """One screen check. Public so tests and the panel can drive it directly."""
        if self.locked():
            self.stats.inc("paused_locked")
            self.screen_reader.reset()
            return None
        t = time.perf_counter()
        frame = self.screen_source.grab() if self.screen_source else None
        if frame is None:
            self.stats.inc("capture_failures")
            return None
        with self._ocr_lock:
            read = self.screen_reader.read(frame)
        frame.close()
        del frame
        self.stats.inc("screen_checks")
        self.stats.timings["diff_ms"].add(read.diff_ms)
        if not read.changed:
            self.stats.inc("unchanged")
            self.stats.timings["screen_check_ms"].add((time.perf_counter() - t) * 1000)
            return None
        self.stats.inc("partial_reads" if read.mode == "partial" else "full_reads")
        self.stats.inc("ocr_crops", read.crops)
        self.stats.timings["ocr_ms"].add(read.ocr_ms)
        obs = Observation(read.spans, source="screen", app=self.context(),
                          read_ms={"diff_ms": read.diff_ms, "ocr_ms": read.ocr_ms})
        verdict = self.analyzer.analyze(obs, self.config.age)
        self.stats.timings["screen_check_ms"].add((time.perf_counter() - t) * 1000)
        self._handle(verdict)
        return verdict

    # -- audio ----------------------------------------------------------------------

    def feed_audio(self, chunk: np.ndarray) -> None:
        """Called from the audio source thread: segmentation only, which is cheap."""
        if self.locked():
            return
        for utterance in self.speech_reader.feed(chunk):
            self.stats.inc("utterances")
            try:
                self._utterances.put_nowait(utterance)
            except queue.Full:
                self.stats.inc("audio_dropped")
                utterance.fill(0.0)

    def _audio_worker(self) -> None:
        while not self._stop.is_set():
            try:
                utterance = self._utterances.get(timeout=0.5)
            except queue.Empty:
                continue
            self.process_utterance(utterance)

    def process_utterance(self, utterance: np.ndarray) -> Optional[Verdict]:
        t = time.perf_counter()
        read = self.speech_reader.transcribe(utterance)
        self.stats.inc("transcribed")
        self.stats.timings["asr_ms"].add(read.transcript.asr_ms)
        if not read.transcript.text:
            self.stats.inc("silent_after_vad")
            return None
        obs = Observation([TextSpan(read.transcript.text, "asr")], source="audio", app=self.context(),
                          read_ms={"asr_ms": read.transcript.asr_ms})
        verdict = self.analyzer.analyze(obs, self.config.age)
        self.stats.timings["audio_verdict_ms"].add((time.perf_counter() - t) * 1000)
        self._handle(verdict)
        return verdict

    # -- one-off checks from the panel ("try it", dropped files) ------------------------

    def check_text(self, text: str) -> Verdict:
        return self._handle(self.analyzer.analyze_text(text, self.config.age, source="typed"), force=True)

    def check_image(self, image: Image.Image) -> Verdict:
        with self._ocr_lock:
            read = ScreenReader(self.screen_reader.ocr).read(image)
        obs = Observation(read.spans, source="screen", read_ms={"ocr_ms": read.ocr_ms})
        return self._handle(self.analyzer.analyze(obs, self.config.age), force=True)

    def check_audio(self, audio: np.ndarray) -> List[Verdict]:
        reader = SpeechReader(self.speech_reader.asr)
        step = 4000
        utterances = []
        padded = np.concatenate([audio.astype(np.float32), np.zeros(16000, dtype=np.float32)])
        for i in range(0, len(padded), step):
            utterances += reader.feed(padded[i:i + step])
        verdicts = []
        for utterance in utterances:
            text = reader.transcribe(utterance).transcript.text
            if text:
                verdicts.append(self._handle(self.analyzer.analyze_text(text, self.config.age, source="audio"),
                                             force=True))
        return verdicts

    # -- alerts ---------------------------------------------------------------------------

    def _handle(self, verdict: Verdict, force: bool = False) -> Verdict:
        self.stats.inc("verdicts")
        self.stats.timings["score_ms"].add(verdict.latency_ms.get("score_ms", 0.0))
        if not verdict.alerts:
            return verdict
        now = time.time()
        last = self._recent.get(verdict.fingerprint)
        if not force and last is not None and now - last < self.config.alert_cooldown_s:
            self.stats.inc("repeats_suppressed")
            return verdict
        self._recent[verdict.fingerprint] = now
        self.stats.inc("alerts")
        self.alerts.appendleft(verdict.to_record())
        if self.on_alert is not None:
            try:
                self.on_alert(verdict)
            except Exception:
                pass
        return verdict

    # -- panel ----------------------------------------------------------------------------

    def snapshot(self) -> dict:
        mem = self._proc.memory_info()
        cache = self.analyzer.cache
        lookups = cache.hits + cache.misses
        return {
            "state": self.state,
            "warm": self.warm,
            "age": self.config.age,
            "persona": persona_for_age(self.config.age),
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0,
            "cpu_pct": round(self._proc.cpu_percent(None) / psutil.cpu_count(), 1),
            "rss_mb": round(mem.rss / 2**20, 1),
            "sources": {
                "screen": getattr(self.screen_source, "name", "off"),
                "audio": getattr(self.audio_source, "name", "off"),
                "ocr": self.screen_reader.ocr.name,
                "asr": self.speech_reader.asr.name,
                "text": self.analyzer.scorer.name,
                "policy": self.analyzer.policy.version,
            },
            "counts": dict(self.stats.counts),
            "timings": {k: v.summary() for k, v in self.stats.timings.items()},
            "cache": {"hits": cache.hits, "misses": cache.misses,
                      "hit_rate": round(cache.hits / lookups, 3) if lookups else 0.0},
            "model_calls": self.analyzer.model_calls,
            "speech": {"in_speech": self.speech_reader.segmenter.in_speech,
                       "gate": round(self.speech_reader.segmenter.gate, 4)},
        }

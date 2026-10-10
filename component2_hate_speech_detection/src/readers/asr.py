"""Speech to text with faster-whisper on CPU (proposal stage 2, FR1, SO3; Guardian M-ASR).

Ported from the J26-DS-316 MVP (`analyst/extract/asr.py`). Audio arrives as a
float32 array held in RAM and never touches disk (proposal NFR5).

Whisper invents fluent text out of silence and room tone (the MVP saw it emit
Japanese on a quiet desktop, which then scored as real speech). Guards:
    language="en"                  English classifiers only, so never drift
    condition_on_previous_text     one hallucination cannot seed the next
    vad_filter                     Silero VAD drops non-speech before decoding
    per-segment no_speech_prob and avg_logprob floors

The model loads on first use, so a silent machine never pays for it.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

NO_SPEECH_MAX = 0.6
AVG_LOGPROB_MIN = -1.0
SAMPLE_RATE = 16000
MODEL_SIZE = os.environ.get("C2_ASR_MODEL", "tiny")
ASR_THREADS = int(os.environ.get("C2_ASR_THREADS", "2"))


@dataclass
class Transcript:
    text: str
    audio_s: float
    asr_ms: float


class AsrEngine:
    def __init__(self, model_size: str = MODEL_SIZE,
                 transcribe_fn: Optional[Callable[[np.ndarray], str]] = None) -> None:
        """`transcribe_fn` injects a fake recogniser for tests."""
        self.model_size = model_size
        self._override = transcribe_fn
        self._model = None
        self._lock = threading.Lock()
        self.last_error = ""
        self.name = "injected" if transcribe_fn else f"faster-whisper-{model_size}"

    @property
    def loaded(self) -> bool:
        return self._model is not None or self._override is not None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8",
                                       cpu_threads=ASR_THREADS, num_workers=1)
        return self._model

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """float32 mono 16 kHz in RAM -> text. Returns empty text rather than raising."""
        t = time.perf_counter()
        audio_s = round(len(audio) / SAMPLE_RATE, 2)
        try:
            if self._override is not None:
                text = (self._override(audio) or "").strip()
            else:
                with self._lock:
                    segments, _ = self._ensure_model().transcribe(
                        audio.astype(np.float32, copy=False), beam_size=1, vad_filter=True,
                        language="en", condition_on_previous_text=False,
                    )
                    kept = [s.text.strip() for s in segments
                            if getattr(s, "no_speech_prob", 0.0) <= NO_SPEECH_MAX
                            and getattr(s, "avg_logprob", 0.0) >= AVG_LOGPROB_MIN and s.text.strip()]
                text = " ".join(kept)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            text = ""
        return Transcript(text, audio_s, round((time.perf_counter() - t) * 1000, 1))

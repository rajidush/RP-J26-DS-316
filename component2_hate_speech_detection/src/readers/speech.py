"""Speech reader: cut system audio into utterances, transcribe, hand to the scorer (SO3).

The MVP pulled a 3-second window every check (once every 30 s), so most of
what was said was never heard. Here audio streams continuously into a
segmenter, and Whisper runs once per *utterance*:

    quiet          only an RMS level is computed (~0 CPU)
    speech starts  the level rises above an adaptive noise floor
    speech ends    600 ms below the floor closes the utterance -> transcribe
    long speech    cut at 8 s so a monologue still reaches a verdict quickly

The noise floor is the 20th percentile of the last 10 s of levels heard
outside utterances. An utterance that runs to the 8 s limit is counted as
background too, so steady game music opens the gate once and is then absorbed,
while real speech (which pauses between phrases) keeps the floor low.
Whisper's own Silero VAD then drops any non-speech that got through, so music
costs one short VAD pass and no decoding. Audio stays in RAM and each
utterance buffer is zeroed after transcription (proposal NFR5).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np

from .asr import SAMPLE_RATE, AsrEngine, Transcript

FRAME_S = 0.1                 # analysis frame
MIN_GATE = 0.004              # absolute RMS floor (loopback is digital; silence is ~0)
FLOOR_FACTOR = 2.5            # speech = RMS above this multiple of the noise floor
FLOOR_WINDOW_S = 10.0         # noise floor = 20th percentile of the last 10 s of levels
START_FRAMES = 2              # 200 ms above the gate opens an utterance
END_SILENCE_S = 0.6
MAX_UTTERANCE_S = 8.0
MIN_VOICED_S = 0.4            # clicks and blips shorter than this are dropped
PRE_ROLL_S = 0.3              # keep the onset that preceded the trigger


@dataclass
class SegmenterStats:
    frames: int = 0
    utterances: int = 0
    dropped_short: int = 0
    noise_floor: float = 0.0


class Segmenter:
    """Energy-gated utterance segmentation over a stream of float32 chunks."""

    def __init__(self) -> None:
        self.frame = int(FRAME_S * SAMPLE_RATE)
        self._pending = np.zeros(0, dtype=np.float32)
        self._pre: Deque[np.ndarray] = deque(maxlen=max(1, int(PRE_ROLL_S / FRAME_S)))
        self._utt: List[np.ndarray] = []
        self._above = 0
        self._silent = 0
        self._voiced = 0
        self._utt_levels: List[float] = []
        self._levels: Deque[float] = deque(maxlen=int(FLOOR_WINDOW_S / FRAME_S))
        self.stats = SegmenterStats()

    @property
    def gate(self) -> float:
        floor = float(np.percentile(self._levels, 20)) if len(self._levels) >= 10 else 0.0
        self.stats.noise_floor = round(floor, 5)
        return max(MIN_GATE, FLOOR_FACTOR * floor)

    @property
    def in_speech(self) -> bool:
        return bool(self._utt)

    def feed(self, chunk: np.ndarray) -> List[np.ndarray]:
        """Add audio; return utterances that just finished."""
        data = np.concatenate([self._pending, np.asarray(chunk, dtype=np.float32).reshape(-1)])
        done: List[np.ndarray] = []
        n = len(data) // self.frame
        for i in range(n):
            frame = data[i * self.frame:(i + 1) * self.frame]
            out = self._step(frame)
            if out is not None:
                done.append(out)
        self._pending = data[n * self.frame:].copy()
        return done

    def flush(self) -> Optional[np.ndarray]:
        return self._close() if self._utt else None

    def _step(self, frame: np.ndarray) -> Optional[np.ndarray]:
        self.stats.frames += 1
        rms = float(np.sqrt(np.mean(frame * frame)))
        loud = rms > self.gate
        if not self._utt:
            self._levels.append(rms)
            if loud:
                self._above += 1
                self._pre.append(frame.copy())
                if self._above >= START_FRAMES:
                    self._utt = list(self._pre)
                    self._pre.clear()
                    self._silent, self._voiced, self._utt_levels = 0, self._above, []
            else:
                self._above = 0
                self._pre.append(frame.copy())
            return None

        self._utt.append(frame.copy())
        self._utt_levels.append(rms)
        self._silent = 0 if loud else self._silent + 1
        self._voiced += int(loud)
        if len(self._utt) * FRAME_S >= MAX_UTTERANCE_S:
            # Sound that never pauses for 8 s is more likely background than speech.
            self._levels.extend(self._utt_levels)
            return self._close()
        if self._silent * FRAME_S >= END_SILENCE_S:
            return self._close()
        return None

    def _close(self) -> Optional[np.ndarray]:
        frames, voiced = self._utt, self._voiced
        self._utt, self._above, self._silent, self._voiced, self._utt_levels = [], 0, 0, 0, []
        audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        for f in frames:
            f.fill(0.0)
        if voiced * FRAME_S < MIN_VOICED_S:
            self.stats.dropped_short += 1
            audio.fill(0.0)
            return None
        self.stats.utterances += 1
        return audio


@dataclass
class SpeechRead:
    transcript: Transcript
    utterance_index: int = 0
    notes: List[str] = field(default_factory=list)


class SpeechReader:
    """Segmenter + ASR. Feed raw chunks, get back transcripts of finished utterances."""

    def __init__(self, asr: Optional[AsrEngine] = None) -> None:
        self.asr = asr or AsrEngine()
        self.segmenter = Segmenter()
        self.transcribed = 0

    def feed(self, chunk: np.ndarray) -> List[np.ndarray]:
        return self.segmenter.feed(chunk)

    def transcribe(self, utterance: np.ndarray) -> SpeechRead:
        try:
            transcript = self.asr.transcribe(utterance)
        finally:
            utterance.fill(0.0)  # RAM only, and wiped once read
        self.transcribed += 1
        notes = [] if transcript.text else ["no_speech_after_vad"]
        return SpeechRead(transcript, self.transcribed, notes)

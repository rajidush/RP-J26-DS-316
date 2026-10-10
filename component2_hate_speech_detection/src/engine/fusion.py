"""Late fusion across signals, under two enforced rules (proposal design rule 3, FR5).

Ported from the J26-DS-316 MVP (`analyst/stage2/fusion.py`).

Why: zero-shot CLIP returned 0.324-0.393 for *every* image in the MVP (clean
gaming, hate and an abstract shape alike). Averaged into a confident text
detection it pulled a confirmed "you should kys" from 0.88 to 0.685, clearing
it for ages 14-15. Safety behaviour must never depend on an uninformative input.

    I-1  Only calibrated signals fuse. An uncalibrated branch is evidence, never score.
    I-2  Fusion may raise confidence, never lower it: the fused score is floored
         at the strongest calibrated signal.

Text read *from* an image (OCR) or from audio (ASR) is text, scored by the text
scorer; the vision channel is only for judgements about pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

TEXT_WEIGHT = 0.60
VISION_WEIGHT = 0.40
# Independent mid-band signals corroborate each other: the hateful-meme case,
# where neither picture nor caption is damning alone.
AGREEMENT_BONUS = 0.15
AGREE_LO, AGREE_HI = 0.35, 0.85


@dataclass(frozen=True)
class Signal:
    name: str          # text | vision | audio
    score: float
    calibrated: bool   # may it move the score, or is it evidence only?
    producer: str = ""

    @property
    def contributes(self) -> bool:
        return self.calibrated and self.score > 0.0


@dataclass
class FusionResult:
    fused: float
    mode: str
    floor: float                 # strongest calibrated signal
    weighted: float              # weighted sum before the floor
    agreement: bool = False
    contributing: List[str] = field(default_factory=list)
    ignored: List[str] = field(default_factory=list)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def fuse(signals: List[Signal]) -> FusionResult:
    contributing = [s for s in signals if s.contributes]
    ignored = [s.name for s in signals if not s.calibrated and s.score > 0.0]
    if not contributing:
        return FusionResult(0.0, "idle", 0.0, 0.0, ignored=ignored)

    floor = max(s.score for s in contributing)
    if len(contributing) == 1:
        only = contributing[0]
        return FusionResult(round(_clamp(only.score), 4), f"{only.name}_only", round(floor, 4),
                            round(only.score, 4), contributing=[only.name], ignored=ignored)

    text = max((s.score for s in contributing if s.name in ("text", "audio")), default=0.0)
    vision = max((s.score for s in contributing if s.name == "vision"), default=0.0)
    weighted = TEXT_WEIGHT * text + VISION_WEIGHT * vision
    agreement = all(AGREE_LO <= s.score < AGREE_HI for s in contributing)
    fused = weighted + AGREEMENT_BONUS if agreement else weighted
    fused = max(fused, floor)  # I-2
    return FusionResult(
        round(_clamp(fused), 4),
        "multimodal_agreement" if agreement else "multimodal",
        round(floor, 4),
        round(_clamp(weighted), 4),
        agreement=agreement,
        contributing=[s.name for s in contributing],
        ignored=ignored,
    )


class FullTextReread:
    """Stage-2 confirm: re-read long text in overlapping chunks and keep the worst.

    Stage 1 windows the text too, but Stage 2 runs only after escalation and can
    afford to look at everything. It reuses the Stage-1 scorer rather than
    loading a third model.
    """

    CHUNK_CHARS = 400
    MAX_CHUNKS = 6

    def __init__(self, scorer: Optional[Callable[[str], float]] = None) -> None:
        self._scorer = scorer

    def score(self, text: str, stage1_score: float) -> float:
        blob = (text or "").strip()
        if self._scorer is None or len(blob) <= self.CHUNK_CHARS:
            return stage1_score
        best = stage1_score
        step = self.CHUNK_CHARS // 2  # overlap, so harm spanning a cut survives
        chunks = [blob[i:i + self.CHUNK_CHARS] for i in range(0, len(blob), step)]
        for chunk in [c for c in chunks if c.strip()][: self.MAX_CHUNKS]:
            try:
                best = max(best, float(self._scorer(chunk)))
            except Exception:
                continue
        return round(best, 4)

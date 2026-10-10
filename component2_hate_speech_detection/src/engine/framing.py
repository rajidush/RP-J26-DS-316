"""Framing guard: is this text *being* harm, or *about* harm? (proposal SO2, FR4)

Ported from the J26-DS-316 MVP (`analyst/stage1/framing.py`); the rules now
live in `packs/policy/framing.json`.

Scorers match harmful language wherever it appears, so they flag a child who
is quoting abuse rather than committing it:

    "someone in the group chat told me to kill myself and i'm scared"
    "miss he keeps calling me a retard what do i do"

Interrupting a child who is asking for help teaches them that reporting abuse
triggers the same machinery as committing it.

Design: a **discount, not a veto** (Guardian invariant I-3). With at least one
signal (attribution, condemnation, meta) the score is capped just below the
lowest persona threshold, never zeroed, so the run is still recorded as a
near-miss a parent can review. It must not become an evasion channel.

Categories in `never_discounted` are exempt. Grooming is built to look like
confiding ("don't tell your parents"), and a child expressing self-harm intent
often frames it as fear ("i'm scared, i want to kill myself"). Those share the
surface features of help-seeking, so the safe error is to alert.

Known limit: pattern matching cannot read intent. "i'm not saying kys but you
should quit" has a condemnation shape and is discounted. A learned framing
classifier is the planned replacement (Guardian M-FRM, RQ2b).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .policy import Policy, default_policy


@dataclass
class Framing:
    reporting: bool = False
    signals: List[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if not self.reporting:
            return ""
        return "quoted_or_reported:" + "+".join(sorted(set(self.signals)))


class FramingGuard:
    def __init__(self, policy: Optional[Policy] = None) -> None:
        pack = (policy or default_policy()).framing
        self.version = pack["version"]
        self.cap = float(pack["cap"])
        self.never_discounted = frozenset(pack["never_discounted"])
        self._signals: Tuple[Tuple[str, Tuple[re.Pattern, ...]], ...] = tuple(
            (family, tuple(re.compile(p, re.IGNORECASE) for p in patterns))
            for family, patterns in pack["signals"].items()
        )

    def detect(self, text: str) -> Framing:
        blob = (text or "").strip()
        if not blob:
            return Framing()
        signals = [family for family, patterns in self._signals if any(p.search(blob) for p in patterns)]
        return Framing(reporting=bool(signals), signals=signals)

    def apply(self, score: float, text: str, category: str = "") -> Tuple[float, Framing]:
        """Cap a harmful score when the text is reporting rather than committing.

        Returns the (possibly reduced) score and the framing that explains it,
        so a discount is never invisible.
        """
        framing = self.detect(text)
        if not framing.reporting:
            return score, framing
        if category in self.never_discounted:
            return score, Framing()
        return (min(score, self.cap), framing) if score > self.cap else (score, framing)

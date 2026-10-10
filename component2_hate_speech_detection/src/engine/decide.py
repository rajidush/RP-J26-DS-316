"""Decision function: (scores, persona, policy, temporal) -> rung (proposal SO4, FR6, FR10).

A direct implementation of the Guardian reference pseudo-code (PROJECT_PLAN
§6.5). It is a pure function of its inputs (invariant I-5): the same inputs
always give the same rung, which is what lets the future C# engine be
parity-tested against this one, rung for rung.

Response ladder:
    L0 record  L1 inform  L2 soft blur  L3 hard blur + educate  L4 full block  L5 close app
    urgent     immediate parent alert, whatever the rung

Age matters twice: the persona threshold (0.55 / 0.65 / 0.75, so the same
words can alert for a 9-year-old and not a 15-year-old), and the rung map
(a 15-year-old gets a soft blur where a 9-year-old gets a hard one). Critical
categories (self-harm, sexual content, grooming) use one stricter threshold
for every age.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .policy import Policy, default_policy

RUNGS = ("L0", "L1", "L2", "L3", "L4", "L5")
PERSONAS = ("P1_PROTECT", "P2_GUIDE", "P3_RESPECT")

# Proposal FR10 names three actions; the ladder refines them.
RUNG_TO_ACTION = {"L0": "none", "L1": "notify", "L2": "blur", "L3": "blur", "L4": "block", "L5": "block"}


@dataclass(frozen=True)
class Decision:
    rung: str
    urgent: bool
    persona: str
    theta: float
    category: str
    p: float
    reason: str

    @property
    def recommended_action(self) -> str:
        return RUNG_TO_ACTION[self.rung]

    @property
    def alerts(self) -> bool:
        return self.rung != "L0"


def persona_for_age(age: int) -> str:
    if age <= 10:
        return "P1_PROTECT"
    if age <= 13:
        return "P2_GUIDE"
    return "P3_RESPECT"


def _rung_index(rung: str) -> int:
    return RUNGS.index(rung)


def decide(
    category: str,
    p: float,
    persona: str,
    policy: Optional[Policy] = None,
    *,
    abstained: bool = False,
    repeat_count: int = 0,
    parent_caps: Optional[Mapping[str, str]] = None,
) -> Decision:
    """Map a top category and its probability to a rung for this persona."""
    d = (policy or default_policy()).decision
    if persona not in PERSONAS:
        raise ValueError(f"unknown persona {persona!r}")

    if abstained:
        return Decision(d["abstain_rung"], False, persona, 0.0, category, p, "abstained")
    if category in ("none", "", None):
        return Decision("L0", False, persona, 0.0, "none", p, "no_category")

    critical = category in d["critical_categories"]
    theta = d["critical_theta"] if critical else d["persona_theta"][persona]
    if p < theta:
        return Decision("L0", False, persona, theta, category, p, "below_threshold")

    rung = d["rung_map"].get(category, {}).get(persona, "L1")
    escalate_after = d.get("escalate_after")
    if escalate_after and repeat_count >= escalate_after:
        rung = RUNGS[min(_rung_index(rung) + 1, _rung_index("L5"))]

    urgent = category in d["urgent_categories"] or (
        category == d["urgent_threat_category"] and p >= d["urgent_theta"]
    )
    max_rung = (parent_caps or {}).get(persona) or d["max_rung"][persona]
    if _rung_index(rung) > _rung_index(max_rung):
        rung = max_rung
    return Decision(rung, urgent, persona, theta, category, p, "critical_threshold" if critical else "persona_threshold")

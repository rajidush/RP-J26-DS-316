"""
EVALUATE state: the session assessor (Seam 2).

EVALUATE no longer calls the phrase rules itself: the controller is given an
assessor, which reads the session's turns and returns, for each child reply,
the codebook's three parts (src/codebook.py) with the evidence behind each (for
Component 4's XAI layer), the session's emotional state, and whether the child
constructed a safer plan. Each classifier arm (rules, off-the-shelf, fine-tuned)
is one implementation, so swapping it never touches the controller, and every
arm derives the emotional state with the same rule (derive_emotional_state).

Every arm must pass the contract suite in tests/test_assessor.py.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence

from . import emotion_rules
from .codebook import (  # noqa: F401  (re-exported: the codebook is part of the assessor's contract)
    CHANGE_TALK, CURRENT_DISTRESS, DISTRESS_SEVERITIES, DISTRESS_SEVERITY, EXPRESSED_EMOTION, EXPRESSED_EMOTIONS,
    LABEL_DEFAULTS, STANCE, STANCES,
)
from .schemas import EmotionalState

if TYPE_CHECKING:
    from .fsm_controller import Turn


@dataclass(frozen=True)
class ReplyAssessment:
    """One child reply's three labels, each with its evidence (the phrases that
    fired, or a model's top classes and scores; empty for a part's default value)."""
    reply: str
    labels: dict[str, str]
    evidence: dict[str, list[str]]


@dataclass(frozen=True)
class SessionAssessment:
    replies: list[ReplyAssessment]   # one per child reply, in transcript order
    emotional_state: EmotionalState
    # Becomes the record's self_regulation_shown. In the rules arm this is still the
    # PP1 keyword flag, which also fires on echoed plans, until ticket 04 uses educator turns.
    constructed_plan: bool


class SessionAssessor(Protocol):
    def assess(self, turns: Sequence[Turn]) -> SessionAssessment: ...


# The stance groups that vote; neutral does not. A tie goes in this order (ADR 0006).
_STANCE_VOTES = {"sustain_talk": EmotionalState.DEFENSIVE, "discord": EmotionalState.DEFENSIVE,
                 "disengaged": EmotionalState.UNCLEAR, CHANGE_TALK: EmotionalState.CALM}
_TIE_ORDER = (EmotionalState.DEFENSIVE, EmotionalState.UNCLEAR, EmotionalState.CALM)


def derive_emotional_state(replies: Sequence[ReplyAssessment]) -> EmotionalState:
    """The session label from the reply labels, the same for every arm (ADR 0006):
    any current distress gives distressed; otherwise the dominant stance (agreed
    29 Oct) gives defensive (sustain talk or discord), unclear (disengaged) or calm
    (change talk), and a session of neutral replies is calm. No replies is unclear."""
    if not replies:
        return EmotionalState.UNCLEAR
    if any(r.labels[DISTRESS_SEVERITY] in CURRENT_DISTRESS for r in replies):
        return EmotionalState.DISTRESSED
    votes = Counter(_STANCE_VOTES[r.labels[STANCE]] for r in replies if r.labels[STANCE] in _STANCE_VOTES)
    if not votes:
        return EmotionalState.CALM
    top = max(votes.values())
    return next(state for state in _TIE_ORDER if votes[state] == top)


class RulesAssessor:
    """The codebook phrase rules (src/emotion_rules.py, ticket 02)."""

    def assess(self, turns: Sequence[Turn]) -> SessionAssessment:
        replies = [t.child_response for t in turns if t.child_response is not None]
        per_reply = [ReplyAssessment(reply, *emotion_rules.label_reply(reply)) for reply in replies]
        return SessionAssessment(
            replies=per_reply,
            emotional_state=derive_emotional_state(per_reply),
            constructed_plan=any(emotion_rules.SELF_REGULATION.search(r.lower()) for r in replies),
        )

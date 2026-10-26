"""
EVALUATE state: the session assessor (Seam 2).

EVALUATE no longer calls the phrase rules itself: the controller is given an
assessor, which reads the session's turns and returns a label set for each
child reply (with the evidence behind each label, for Component 4's XAI
layer), the session's emotional state, and whether the child constructed a
safer plan. Each classifier arm (rules, off-the-shelf, fine-tuned) is one
implementation, so swapping it never touches the controller.

Every arm must pass the contract suite in tests/test_assessor.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence

from . import emotion_rules
from .schemas import EmotionalState

if TYPE_CHECKING:
    from .fsm_controller import Turn


@dataclass(frozen=True)
class ReplyAssessment:
    """One child reply's labels, each with its evidence (the phrases that fired,
    or a model's top classes and scores; empty when nothing supports it)."""
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


class RulesAssessor:
    """The PP1 phrase rules (src/emotion_rules.py), unchanged. Each reply is
    labelled by which signals fired in it ("yes"/"no"); the emotional state is
    still classify() over all replies together, so no session label changes."""

    def assess(self, turns: Sequence[Turn]) -> SessionAssessment:
        replies = [t.child_response for t in turns if t.child_response is not None]
        per_reply = []
        for reply in replies:
            hits = emotion_rules.signal_hits(reply)
            per_reply.append(ReplyAssessment(reply, {k: "yes" if v else "no" for k, v in hits.items()}, hits))
        return SessionAssessment(
            replies=per_reply,
            emotional_state=emotion_rules.classify(replies).state,
            constructed_plan=any(emotion_rules.SELF_REGULATION.search(r.lower()) for r in replies),
        )

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


# Every arm labels each reply's stance (the codebook's five values, ADR 0006):
# INQUIRE ends once the child states a safer plan, which is change talk.
STANCE = "stance"
CHANGE_TALK = "change_talk"
STANCES = (CHANGE_TALK, "neutral", "sustain_talk", "discord", "disengaged")
# Provisional rules-arm stance until ticket 02's codebook rules: the first PP1
# signal that fired, in this order. The PP1 rules have nothing for discord.
_STANCE_FROM_SIGNAL = (("reflective", CHANGE_TALK), ("defensive", "sustain_talk"), ("withdrawn", "disengaged"))


def _rules_stance(hits: dict[str, list[str]]) -> tuple[str, list[str]]:
    return next(((stance, hits[signal]) for signal, stance in _STANCE_FROM_SIGNAL if hits[signal]), ("neutral", []))


class RulesAssessor:
    """The PP1 phrase rules (src/emotion_rules.py), unchanged. Each reply is
    labelled by which signals fired in it ("yes"/"no"), plus a stance mapped from
    them; the emotional state is still classify() over all replies together, so
    no session label changes."""

    def assess(self, turns: Sequence[Turn]) -> SessionAssessment:
        replies = [t.child_response for t in turns if t.child_response is not None]
        per_reply = []
        for reply in replies:
            hits = emotion_rules.signal_hits(reply)
            labels = {k: "yes" if v else "no" for k, v in hits.items()}
            labels[STANCE], stance_evidence = _rules_stance(hits)
            per_reply.append(ReplyAssessment(reply, labels, {**hits, STANCE: stance_evidence}))
        return SessionAssessment(
            replies=per_reply,
            emotional_state=emotion_rules.classify(replies).state,
            constructed_plan=any(emotion_rules.SELF_REGULATION.search(r.lower()) for r in replies),
        )

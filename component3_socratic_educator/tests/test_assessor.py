"""
Seam 2 tests: the session assessor EVALUATE calls (src/assessor.py).

The contract suite runs against every arm in ARMS; a new classifier arm joins
by being added there. The rules-arm tests below it pin the PP1 rules wrapper
to src/emotion_rules.py, so the prefactor changes no label.
"""
import pytest

from component3_socratic_educator.src import emotion_rules
from component3_socratic_educator.src.assessor import (
    STANCE, STANCES, ReplyAssessment, RulesAssessor, SessionAssessment,
)
from component3_socratic_educator.src.fsm_controller import DialogueState, Turn
from component3_socratic_educator.src.schemas import EmotionalState as E

ARMS = [pytest.param(RulesAssessor(), id="rules")]


def session(*replies):
    """Turns as the controller records them: an INTERCEPT line, INQUIRE lines, then EVALUATE."""
    turns = [Turn(DialogueState.INTERCEPT if i == 0 else DialogueState.INQUIRE, f"question {i}?", r)
             for i, r in enumerate(replies)]
    return turns + [Turn(DialogueState.EVALUATE, "[internal evaluation, no child-facing text]")]


SESSIONS = [
    session("I clicked a link by accident", "I closed it after that"),
    session("It's not my fault.", "I feel so scared."),
    session("idk", "leave me alone"),
    session("It was really scary.", "I think I should talk to my mom about it."),
    session("", "ok"),
    session(),
]


# -- contract: every arm --------------------------------------------------

@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("turns", SESSIONS)
def test_one_result_per_child_reply_in_order(arm, turns):
    result = arm.assess(turns)
    assert isinstance(result, SessionAssessment)
    assert [r.reply for r in result.replies] == [t.child_response for t in turns if t.child_response is not None]
    assert all(isinstance(r, ReplyAssessment) for r in result.replies)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("turns", SESSIONS)
def test_every_label_comes_with_evidence(arm, turns):
    for r in arm.assess(turns).replies:
        assert r.labels and all(isinstance(k, str) and isinstance(v, str) for k, v in r.labels.items())
        assert set(r.evidence) == set(r.labels)
        assert all(isinstance(e, list) and all(isinstance(x, str) for x in e) for e in r.evidence.values())


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("turns", SESSIONS)
def test_every_reply_has_a_stance_from_the_codebook(arm, turns):
    # INQUIRE ends on change talk (ticket 03), so every arm must give a stance.
    assert all(r.labels[STANCE] in STANCES for r in arm.assess(turns).replies)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("turns", SESSIONS)
def test_session_state_and_plan_are_typed(arm, turns):
    result = arm.assess(turns)
    assert isinstance(result.emotional_state, E)
    assert isinstance(result.constructed_plan, bool)


# -- rules arm: identical to the PP1 rules ----------------------------------

@pytest.mark.parametrize("replies", [
    ["I clicked a link by accident", "I closed it after that"],
    ["It's not my fault.", "I didn't do anything."],
    ["I feel so alone.", "My heart feels heavy."],
    ["I don't know.", "Just... quiet."],
    ["I'm so scared and alone.", "Maybe I should tell someone."],
    ["I'm so scared and alone and it hurts.", "Maybe I should tell someone."],
    ["It's not my fault.", "It's scary."],
    ["no", "ok"],
    [],
])
def test_rules_state_is_the_pp1_classifier(replies):
    assert RulesAssessor().assess(session(*replies)).emotional_state == emotion_rules.classify(replies).state


def test_rules_evidence_is_the_phrases_that_fired_in_each_reply():
    first, second = RulesAssessor().assess(session("It's not my fault.", "I feel so scared.")).replies
    assert first.labels["defensive"] == "yes" and first.evidence["defensive"] == ["not my fault"]
    assert first.labels["distress"] == "no" and first.evidence["distress"] == []
    assert second.labels["distress"] == "yes" and second.evidence["distress"] == ["scared"]


@pytest.mark.parametrize("turns", SESSIONS)
def test_rules_never_give_a_yes_without_evidence(turns):
    for r in RulesAssessor().assess(turns).replies:
        assert all(r.evidence[k] for k, v in r.labels.items() if v == "yes")


@pytest.mark.parametrize("replies,expected", [
    (["hmm", "I should have closed it instead"], True),
    (["I couldn't stop watching"], False),   # whole words: "i could" not inside "i couldn't"
    (["I clicked a link by accident"], False),
])
def test_rules_plan_is_the_pp1_self_regulation_keywords(replies, expected):
    assert RulesAssessor().assess(session(*replies)).constructed_plan is expected


@pytest.mark.parametrize("reply,stance,evidence", [
    ("Maybe I should tell someone.", "change_talk", ["maybe i", "tell someone"]),
    ("It's not my fault.", "sustain_talk", ["not my fault"]),
    ("leave me alone", "disengaged", ["leave me alone"]),
    ("I clicked a link by accident", "neutral", []),
    ("", "neutral", []),
])
def test_rules_stance_is_a_provisional_mapping_of_the_pp1_signals(reply, stance, evidence):
    # Until ticket 02 rewrites the rules to the codebook: a planning phrase is change
    # talk, then a denial is sustain talk, then a non-answer is disengaged.
    (result,) = RulesAssessor().assess(session(reply)).replies
    assert (result.labels[STANCE], result.evidence[STANCE]) == (stance, evidence)

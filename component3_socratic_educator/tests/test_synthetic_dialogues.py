"""
2 Oct task: automated tests over ~15-20 synthetic dialogues, covering the
normal flow and edge cases of the FSM Controller (Function 1).

Each dialogue is a scripted child (replies are consumed in order; the last
reply repeats if the FSM asks more questions than the script has) plus the
outcome the FSM must produce. Every dialogue is also checked against the
invariants that hold for ALL sessions (state order, schema validity, no
flagged-content leak, no repeated INQUIRE question).

Replies are consumed as: reply[0] -> INTERCEPT, then up to MAX_INQUIRE_MOVES
replies -> INQUIRE, one per Socratic move (since 26 Oct, ticket 03: it stops
early on a refusal or a safer plan, and "I don't know" repeats a move in its
easier form). Before 26 Oct a "complete reply" also ended INQUIRE, so many
dialogues below had 2 child turns; with no plan or refusal they now get all
three moves, and no session label changed.

Runs offline: tests/conftest.py blocks the live model, so INTERCEPT uses
the template pool.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jsonschema
import pytest

from component3_socratic_educator.src import config
from component3_socratic_educator.src.fsm_controller import DialogueState, FSMController
from component3_socratic_educator.src.schemas import (
    EmotionalState, RiskCategory, RiskLevel, TriggerPayload, load_mock_trigger,
)

MOCK_INPUTS = Path(__file__).resolve().parents[1] / "mock_inputs"

_COMP1 = {RiskCategory.EXPLICIT_VISUAL, RiskCategory.VIOLENCE,
          RiskCategory.SELF_HARM_IMAGERY, RiskCategory.UNKNOWN_FLAGGED}

CLOSE_NORMAL = "Thanks for talking this through with me."
CLOSE_ESCALATE = "I'm going to let a parent know we talked about this, so they can support you."
CLOSE_REFUSED = "That's okay, you don't have to talk right now. I'm here if you want to talk later."


def make_trigger(rc: RiskCategory, confidence: float) -> TriggerPayload:
    if rc in _COMP1:
        return TriggerPayload.new("component1_screen_monitoring", "image", rc, confidence,
                                  app_or_window="browser")
    return TriggerPayload.new("component2_hate_speech_detection", "text_message", rc, confidence)


def scripted(replies):
    state = {"i": 0}

    def respond(_prompt):
        reply = replies[min(state["i"], len(replies) - 1)]
        state["i"] += 1
        return reply
    return respond


@dataclass
class Dialogue:
    name: str
    rc: RiskCategory
    confidence: float
    replies: list
    child_turns: int                 # INTERCEPT reply + INQUIRE replies
    emotional_state: EmotionalState
    self_regulation: bool
    risk_level: RiskLevel
    note: Optional[str] = None
    refuses: bool = False            # the child's last reply is a refusal

    @property
    def escalates(self) -> bool:
        return (self.risk_level.value in config.ESCALATION_RISK_LEVELS
                or self.emotional_state.value in config.ESCALATION_EMOTIONAL_STATES)


DIALOGUES = [
    # --- normal flow ------------------------------------------------------
    Dialogue("calm_explains_first_try", RiskCategory.EXPLICIT_VISUAL, 0.62,
             ["hmm", "I clicked a link by accident"],
             4, EmotionalState.CALM, False, RiskLevel.MODERATE,
             note="was \"I'm not sure\" until 14 Oct; uncertainty now scores unclear (test_emotion_rules)"),
    Dialogue("self_regulation_instead", RiskCategory.VIOLENCE, 0.90,
             ["I saw it", "I should have closed it instead"],
             2, EmotionalState.CALM, True, RiskLevel.HIGH),
    Dialogue("self_regulation_next_time", RiskCategory.CYBERBULLYING, 0.55,
             ["they were being mean", "next time I will tell a teacher"],
             2, EmotionalState.CALM, True, RiskLevel.MODERATE),
    Dialogue("self_regulation_uppercase", RiskCategory.HATE_SPEECH, 0.70,
             ["hmm", "INSTEAD I just closed the chat"],
             2, EmotionalState.CALM, True, RiskLevel.MODERATE),
    Dialogue("short_replies_get_every_move", RiskCategory.UNKNOWN_FLAGGED, 0.40,
             ["idk", "no", "I did not mean it"],
             4, EmotionalState.UNCLEAR, False, RiskLevel.LOW,
             note="was complete_on_second_inquire (3 turns) until the complete-reply rule was retired, 26 Oct"),
    Dialogue("dont_know_gets_the_easier_form", RiskCategory.CYBERBULLYING, 0.60,
             ["hmm", "idk", "they kept posting about me", "I felt left out"],
             4, EmotionalState.DISTRESSED, False, RiskLevel.MODERATE,
             note="clarify, clarify's easier form, consequences: the repeat counts toward the cap; "
                  "'left out' is current mild distress under the codebook (ticket 02; was unclear)"),
    Dialogue("low_risk_closes_normally", RiskCategory.GROOMING_LANGUAGE, 0.30,
             ["someone messaged me", "they asked where I live"],
             4, EmotionalState.CALM, False, RiskLevel.LOW),

    # --- minimal / defensive / withdrawn responses ------------------------
    Dialogue("one_word_every_turn", RiskCategory.HATE_SPEECH, 0.70,
             ["whatever"],
             4, EmotionalState.DEFENSIVE, False, RiskLevel.MODERATE,
             note="a denial no longer ends INQUIRE (26 Oct): it is sustain talk, not a plan"),
    Dialogue("two_words_is_still_defensive", RiskCategory.CYBERBULLYING, 0.60,
             ["no", "so what", "not really", "my fault"],
             4, EmotionalState.DEFENSIVE, False, RiskLevel.MODERATE,
             note="'so what' ended INQUIRE as a complete reply from 14 to 26 Oct"),
    Dialogue("three_word_deflection_is_defensive", RiskCategory.VIOLENCE, 0.60,
             ["hmm", "it was nothing", "it was nothing", "just a video"],
             4, EmotionalState.DEFENSIVE, False, RiskLevel.MODERATE,
             note="a 3-word denial ended INQUIRE as a complete reply from 14 to 26 Oct"),
    Dialogue("four_word_reply_gets_every_move", RiskCategory.EXPLICIT_VISUAL, 0.20,
             ["idk", "it was an ad"],
             4, EmotionalState.UNCLEAR, False, RiskLevel.LOW,
             note="was exactly_four_words_completes (2 turns) until 26 Oct"),
    Dialogue("silent_child", RiskCategory.SELF_HARM_LANGUAGE, 0.88,
             [""],
             4, EmotionalState.UNCLEAR, False, RiskLevel.HIGH),
    Dialogue("whitespace_only_child", RiskCategory.SELF_HARM_IMAGERY, 0.50,
             ["   "],
             4, EmotionalState.UNCLEAR, False, RiskLevel.MODERATE,
             note="whitespace is no answer, same as an empty reply"),

    # --- heuristic edge cases ---------------------------------------------
    Dialogue("couldnt_is_not_i_could", RiskCategory.VIOLENCE, 0.60,
             ["I couldn't stop watching it"],
             4, EmotionalState.CALM, False, RiskLevel.MODERATE,
             note="'i could' keyword must not match inside \"couldn't\""),
    Dialogue("off_topic_reply", RiskCategory.UNKNOWN_FLAGGED, 0.45,
             ["can I play minecraft now", "can I play minecraft now"],
             4, EmotionalState.CALM, False, RiskLevel.LOW,
             note="off-topic is not detected yet: every move is asked and it scores calm"),
    Dialogue("distress_is_detected", RiskCategory.SELF_HARM_LANGUAGE, 0.70,
             ["I feel really scared and I do not want to talk about it"],
             1, EmotionalState.DISTRESSED, False, RiskLevel.MODERATE,
             note="was a known limitation (always calm) until emotion_rules.py, 6 Oct; "
                  "a refusal to the opening line skips INQUIRE since 14 Oct",
             refuses=True),

    # --- risk-level / escalation boundaries -------------------------------
    Dialogue("boundary_085_escalates", RiskCategory.SELF_HARM_IMAGERY, 0.85,
             ["I was scrolling", "it just popped up on my feed"],
             4, EmotionalState.CALM, False, RiskLevel.HIGH),
    Dialogue("boundary_084_does_not_escalate", RiskCategory.SELF_HARM_IMAGERY, 0.84,
             ["I was scrolling", "it just popped up on my feed"],
             4, EmotionalState.CALM, False, RiskLevel.MODERATE),
    Dialogue("boundary_050_is_moderate", RiskCategory.GROOMING_LANGUAGE, 0.50,
             ["ok", "a stranger added me today"],
             4, EmotionalState.CALM, False, RiskLevel.MODERATE),
]


def run_dialogue(d: Dialogue):
    return FSMController().run(make_trigger(d.rc, d.confidence), scripted(d.replies))


@pytest.mark.parametrize("d", DIALOGUES, ids=lambda d: d.name)
def test_dialogue_outcome(d):
    result, transcript = run_dialogue(d)
    assert result.dialogue_turns == d.child_turns
    assert result.emotional_state == d.emotional_state
    assert result.self_regulation_shown is d.self_regulation
    assert result.risk_level == d.risk_level
    assert result.escalation_flag is d.escalates
    closing = transcript.turns[-1]
    assert closing.state == DialogueState.CONTRACT
    assert closing.system_text == (CLOSE_ESCALATE if d.escalates else CLOSE_REFUSED if d.refuses else CLOSE_NORMAL)


@pytest.mark.parametrize("d", DIALOGUES, ids=lambda d: d.name)
def test_dialogue_invariants(d):
    result, transcript = run_dialogue(d)
    states = [t.state for t in transcript.turns]

    # State order: exactly one INTERCEPT, 1..MAX INQUIRE (0 only after a refusal to the
    # opening line), one EVALUATE, one CONTRACT.
    n_inquire = states.count(DialogueState.INQUIRE)
    assert states == ([DialogueState.INTERCEPT] + [DialogueState.INQUIRE] * n_inquire
                      + [DialogueState.EVALUATE, DialogueState.CONTRACT])
    assert (0 if d.refuses and result.dialogue_turns == 1 else 1) <= n_inquire <= config.MAX_INQUIRE_MOVES

    # The child never sees the flagged category named.
    for turn in transcript.turns:
        for word in d.rc.value.split("_"):
            assert word not in turn.system_text.lower()

    # INQUIRE never asks the same question twice in one session.
    questions = [t.system_text for t in transcript.turns if t.state == DialogueState.INQUIRE]
    assert len(questions) == len(set(questions))

    # Output always honours the Component 4 contract.
    result.validate()
    assert result.session_id == transcript.trigger.session_id


def test_suite_size_matches_plan():
    """Plan target for 2 Oct: ~15-20 synthetic dialogues."""
    assert 15 <= len(DIALOGUES) <= 20


@pytest.mark.parametrize("mock_file", ["trigger_from_component1.json", "trigger_from_component2.json"])
def test_mock_upstream_triggers_run_end_to_end(mock_file):
    trigger = load_mock_trigger(MOCK_INPUTS / mock_file)
    result, _ = FSMController().run(trigger, scripted(["not sure", "I should have told my dad instead"]))
    result.validate()
    assert result.self_regulation_shown is True


def test_invalid_trigger_is_rejected_before_the_child_is_asked_anything():
    trigger = make_trigger(RiskCategory.VIOLENCE, 0.7)
    trigger.confidence_score = 1.7  # outside the contract's [0, 1]
    asked = []
    with pytest.raises(jsonschema.ValidationError):
        FSMController().run(trigger, lambda prompt: asked.append(prompt) or "ok")
    assert asked == []


def test_category_from_wrong_component_is_rejected():
    # hate_speech is a Component 2 category; Component 1's contract must refuse it.
    trigger = TriggerPayload.new("component1_screen_monitoring", "image",
                                 RiskCategory.HATE_SPEECH, 0.7)
    with pytest.raises(jsonschema.ValidationError):
        FSMController().run(trigger, lambda _p: "ok")

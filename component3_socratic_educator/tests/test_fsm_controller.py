"""
Function 1 tests. Run with: pytest component3_socratic_educator/tests -v
"""
import sys
from pathlib import Path
import pytest
from component3_socratic_educator.src.fsm_controller import FSMController, SessionTranscript
from component3_socratic_educator.src import config
from component3_socratic_educator.src.schemas import RiskCategory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component3_socratic_educator.src.fsm_controller import DialogueState, FSMController
from component3_socratic_educator.src.schemas import RiskCategory, TriggerPayload


def make_trigger(confidence=0.9, risk_category=RiskCategory.EXPLICIT_VISUAL):
    return TriggerPayload.new(
        source_component="component1_screen_monitoring",
        content_type="image",
        risk_category=risk_category,
        confidence_score=confidence,
        app_or_window="browser",
    )


def test_full_session_completes_and_validates():
    controller = FSMController()
    trigger = make_trigger()
    responses = iter(["I'm not sure", "I guess I was just curious", "I should have closed it instead"])
    result, transcript = controller.run(trigger, respond=lambda _: next(responses))

    assert result.session_id == trigger.session_id
    assert result.dialogue_turns >= 1
    # will raise if it doesn't match docs/interface-contracts/comp3_to_comp4.schema.json
    result.validate()


def test_inquire_stops_at_max_attempts_even_with_short_answers():
    controller = FSMController()
    trigger = make_trigger()
    # every child response is a single word -> never "complete" ->
    # must still terminate via MAX_INQUIRE_ATTEMPTS, not loop forever
    result, transcript = controller.run(trigger, respond=lambda _: "no")
    inquire_turns = [t for t in transcript.turns if t.state == DialogueState.INQUIRE]
    assert len(inquire_turns) <= 3


def test_high_confidence_trigger_produces_escalation():
    controller = FSMController()
    trigger = make_trigger(confidence=0.95)
    result, _ = controller.run(trigger, respond=lambda _: "whatever")
    assert result.risk_level.value == "high"
    assert result.escalation_flag is True


def test_low_confidence_trigger_does_not_escalate():
    controller = FSMController()
    trigger = make_trigger(confidence=0.2)
    result, _ = controller.run(trigger, respond=lambda _: "I was just looking around, nothing serious")
    assert result.escalation_flag is False

@pytest.mark.parametrize("rc", list(RiskCategory))
def test_intercept_never_names_flagged_content(rc):
    controller = FSMController()
    trigger = make_trigger(risk_category=rc)
    text = controller.opening_question(trigger).lower()
    # neither the category label nor its words may appear
    for word in rc.value.split("_"):
        assert word not in text

def test_intercept_is_reproducible():
    controller = FSMController()
    trigger = make_trigger()
    assert controller.opening_question(trigger) == controller.opening_question(trigger)
    
def test_inquire_stops_at_max_attempts_with_short_answers():
    controller = FSMController()
    trigger = make_trigger()
    transcript = SessionTranscript(trigger=trigger)
    responses = controller.inquire_loop(trigger, respond=lambda _: "no", transcript=transcript)
    assert len(responses) == config.MAX_INQUIRE_ATTEMPTS

def test_inquire_stops_early_on_complete_answer():
    controller = FSMController()
    trigger = make_trigger()
    transcript = SessionTranscript(trigger=trigger)
    answers = iter(["no", "I was just curious about it"])
    responses = controller.inquire_loop(trigger, respond=lambda _: next(answers), transcript=transcript)
    assert len(responses) == 2
    
def test_inquire_logs_turns_to_transcript():
    controller = FSMController()
    trigger = make_trigger()
    transcript = SessionTranscript(trigger=trigger)
    controller.inquire_loop(trigger, respond=lambda _: "no", transcript=transcript)
    assert len(transcript.turns) == config.MAX_INQUIRE_ATTEMPTS
    
def test_run_actually_uses_inquire_loop():
    controller = FSMController()
    trigger = make_trigger()
    _, transcript = controller.run(trigger, respond=lambda _: "no")
    inquire_turns = [t for t in transcript.turns if t.state.name == "INQUIRE"]
    assert len(inquire_turns) == config.MAX_INQUIRE_ATTEMPTS


# --- 14 Oct fixes -------------------------------------------------------------

def _inquire_replies(replies):
    """Run a session; return the replies the child was asked for during INQUIRE."""
    it = iter(replies)
    _, transcript = FSMController().run(make_trigger(confidence=0.6), respond=lambda _: next(it, "ok"))
    return [t.child_response for t in transcript.turns if t.state == DialogueState.INQUIRE]


@pytest.mark.parametrize("reply", ["I feel sick", "so scared", "not my fault", "I'll tell mum"])
def test_a_short_reply_that_states_a_feeling_denial_or_plan_is_complete(reply):
    # 12 Oct run: "I feel sick" (3 words) was asked again; it had already said something.
    assert _inquire_replies(["hmm", reply, "never asked"]) == [reply]


def test_a_short_reply_with_nothing_in_it_is_still_followed_up():
    assert _inquire_replies(["hmm", "ok", "no", "fine"]) == ["ok", "no", "fine"]


REFUSAL_CLOSE = "That's okay, you don't have to talk right now. I'm here if you want to talk later."


@pytest.mark.parametrize("refusal", ["leave me alone", "I don't want to talk", "I do not want to talk about it",
                                     "stop asking", "go away"])
def test_a_refusal_ends_inquire_at_once(refusal):
    # 12 Oct run: the withdrawn child was asked again after "leave me alone".
    assert _inquire_replies(["hmm", refusal, "never asked"]) == [refusal]


def test_a_refusal_to_the_opening_line_skips_inquire():
    assert _inquire_replies(["leave me alone", "never asked"]) == []


def test_a_refusing_child_who_is_not_escalated_hears_that_it_is_okay_not_to_talk():
    it = iter(["hmm", "leave me alone"])
    result, transcript = FSMController().run(make_trigger(confidence=0.6), respond=lambda _: next(it))
    assert result.escalation_flag is False
    assert transcript.turns[-1].system_text == REFUSAL_CLOSE


def test_a_refusing_child_who_is_escalated_still_hears_the_parent_line():
    it = iter(["hmm", "leave me alone"])
    result, transcript = FSMController().run(make_trigger(confidence=0.9), respond=lambda _: next(it))
    assert result.escalation_flag is True
    assert "parent" in transcript.turns[-1].system_text

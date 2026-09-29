"""
Function 1 tests. Run with: pytest component3_socratic_educator/tests -v
"""
import sys
from pathlib import Path
import pytest
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
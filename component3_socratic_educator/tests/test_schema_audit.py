"""
src/schema_audit: one check per contract boundary, used by
evidence/validate_schemas.py to re-validate every recorded output (plan, Thu 15 Oct).
"""
import pytest

from component3_socratic_educator.src import schema_audit

TRIGGER = {"session_id": "28342c30-858f-46ee-b29d-21af97d5ee82", "source_component": "component1_screen_monitoring",
           "content_type": "image", "risk_category": "explicit_visual", "confidence_score": 0.88,
           "timestamp": "2026-10-14T15:05:27.355503+00:00", "context_metadata": {}}
RECORD = {"session_id": "28342c30-858f-46ee-b29d-21af97d5ee82", "risk_level": "high", "emotional_state": "calm",
          "self_regulation_shown": True, "escalation_flag": True, "dialogue_turns": 2,
          "dialogue_summary": "2 turns; risk_category=explicit_visual; escalated: risk_level=high",
          "timestamp": "2026-10-14T15:05:29.729407+00:00"}
MODEL_LINE = "Something you just saw might need a closer look. What were you focusing on?"


def test_a_valid_trigger_and_record_pass():
    assert schema_audit.trigger_error(TRIGGER) is None
    assert schema_audit.record_error(RECORD) is None


def test_a_trigger_is_checked_against_its_own_source_contract():
    # a text category is not allowed in Component 1's contract
    assert "risk_category" in schema_audit.trigger_error({**TRIGGER, "risk_category": "hate_speech"})
    text_trigger = {**TRIGGER, "source_component": "component2_hate_speech_detection",
                    "content_type": "text_message", "risk_category": "hate_speech"}
    assert schema_audit.trigger_error(text_trigger) is None


def test_errors_name_the_failing_field():
    assert "risk_level" in schema_audit.record_error({**RECORD, "risk_level": "severe"})
    assert "dialogue_turns" in schema_audit.record_error({k: v for k, v in RECORD.items() if k != "dialogue_turns"})


@pytest.mark.parametrize("source,reason,raw", [
    ("model", None, "valid"),                         # parsed, passed every check
    ("fallback", "invites_description", "valid"),     # valid JSON, blocked for its words
    ("fallback", "invalid_json", "parse_failure"),    # the reply was not JSON
    ("fallback", "grammar", "parse_failure"),         # JSON that breaks the schema (shape, opener enum or question pattern)
    ("fallback", "model_unreachable", "no_reply"),    # nothing to validate
])
def test_an_intercept_call_is_classified_by_its_recorded_outcome(source, reason, raw):
    assert schema_audit.intercept_outcome(source, reason) == raw


def test_a_delivered_intercept_line_is_checked_against_the_controller_grammar():
    assert schema_audit.delivered_intercept_error(MODEL_LINE) is None
    assert schema_audit.delivered_intercept_error("Hello there. What did you see?") == "grammar"
    assert schema_audit.delivered_intercept_error(
        "I noticed something on your screen just now. What did you see?") == "invites_description"


def test_a_delivered_inquire_line_must_be_one_of_the_current_template_lines():
    from component3_socratic_educator.src.grammar_decoder import INQUIRE_TEMPLATES
    assert schema_audit.delivered_inquire_error(INQUIRE_TEMPLATES["clarify"]["11-13"][1]) is None
    assert schema_audit.delivered_inquire_error("I understand. Is there another way you could've reacted?") == "grammar"
    # a 14 Oct reply-type line, retired on 26 Oct (ticket 03)
    assert schema_audit.delivered_inquire_error("That makes sense. Can you say a bit more about that?") == "grammar"


def test_only_a_failure_by_the_rule_that_did_not_exist_yet_counts_as_resolved():
    # 15 Oct: a passing line (None) in a run with no such rule (None) was counted as resolved.
    assert schema_audit.failed_only_by(None, None) is False
    assert schema_audit.failed_only_by(None, "invites_description") is False
    assert schema_audit.failed_only_by("invites_description", "invites_description") is True
    assert schema_audit.failed_only_by("grammar", "invites_description") is False


def test_an_unknown_fallback_reason_is_an_error_not_silently_valid():
    with pytest.raises(ValueError, match="no_such_reason"):
        schema_audit.intercept_outcome("fallback", "no_such_reason")

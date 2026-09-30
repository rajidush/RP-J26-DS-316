"""
30 Sep task verification: Evaluate state structured output.

Task: Evaluate state's structured output (risk level, emotional state,
self-regulation) as a typed model, schema-validated.

Evidence target: passing test run log + schema-validation passing.
Logbook prompt: Why enum/boolean fields, not free text?

Tests in this file probe every requirement of the 30 Sep task explicitly:
  1.  EvaluateOutput has all required typed fields (no raw free text).
  2.  risk_level is a RiskLevel enum (low / moderate / high) — never a raw string.
  3.  emotional_state is an EmotionalState enum (calm / defensive / distressed / unclear).
  4.  self_regulation_shown is a bool — True when keywords found in transcript.
  5.  escalation_flag is a bool — set in CONTRACT, not EVALUATE.
  6.  dialogue_turns is an int, ≥ 1.
  7.  EvaluateOutput.validate() passes jsonschema against comp3_to_comp4.schema.json.
  8.  Risk thresholds are deterministic (confidence → level boundaries).
  9.  Emotional state heuristic is deterministic (word-count boundaries).
  10. Schema validation FAILS fast on a deliberately broken payload.
  11. All 8 RiskCategory × 3 confidence bands → always produces valid output.
  12. Self-regulation detection covers every keyword in the heuristic list.
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component3_socratic_educator.src.fsm_controller import FSMController, SessionTranscript, DialogueState
from component3_socratic_educator.src.schemas import (
    RiskCategory, RiskLevel, EmotionalState, EvaluateOutput, TriggerPayload
)
import jsonschema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trigger(confidence: float = 0.9,
                 risk_category: RiskCategory = RiskCategory.EXPLICIT_VISUAL,
                 source: str = "component1_screen_monitoring"):
    return TriggerPayload.new(
        source_component=source,
        content_type="image",
        risk_category=risk_category,
        confidence_score=confidence,
        app_or_window="browser",
    )


def run_session(confidence: float = 0.7,
                risk_category: RiskCategory = RiskCategory.EXPLICIT_VISUAL,
                child_replies=None):
    """Run a full FSM session and return (EvaluateOutput, SessionTranscript)."""
    if child_replies is None:
        child_replies = ["I was just curious about it one day"]
    replies = iter(child_replies)
    controller = FSMController()
    trigger = make_trigger(confidence=confidence, risk_category=risk_category)
    return controller.run(trigger, respond=lambda _: next(replies, "okay"))


# ---------------------------------------------------------------------------
# 1. EvaluateOutput field types — no raw free text
# ---------------------------------------------------------------------------

class TestEvaluateOutputStructure:

    def test_risk_level_is_risklevel_enum(self):
        result, _ = run_session()
        assert isinstance(result.risk_level, RiskLevel), (
            "risk_level must be a RiskLevel enum, not a raw string"
        )

    def test_emotional_state_is_emotionalstate_enum(self):
        result, _ = run_session()
        assert isinstance(result.emotional_state, EmotionalState), (
            "emotional_state must be an EmotionalState enum, not a raw string"
        )

    def test_self_regulation_shown_is_bool(self):
        result, _ = run_session()
        assert isinstance(result.self_regulation_shown, bool), (
            "self_regulation_shown must be a bool, not an int or string"
        )

    def test_escalation_flag_is_bool(self):
        result, _ = run_session()
        assert isinstance(result.escalation_flag, bool)

    def test_dialogue_turns_is_int_and_positive(self):
        result, _ = run_session()
        assert isinstance(result.dialogue_turns, int)
        assert result.dialogue_turns >= 1, (
            "dialogue_turns must be ≥ 1 (schema minimum: 1)"
        )

    def test_session_id_matches_trigger(self):
        controller = FSMController()
        trigger = make_trigger()
        result, _ = controller.run(trigger, respond=lambda _: "I was just looking around")
        assert result.session_id == trigger.session_id


# ---------------------------------------------------------------------------
# 2. Risk level thresholds — confidence → enum
# ---------------------------------------------------------------------------

class TestRiskLevelThresholds:

    @pytest.mark.parametrize("confidence,expected", [
        (0.0,  RiskLevel.LOW),
        (0.2,  RiskLevel.LOW),
        (0.49, RiskLevel.LOW),
        (0.5,  RiskLevel.MODERATE),
        (0.75, RiskLevel.MODERATE),
        (0.84, RiskLevel.MODERATE),
        (0.85, RiskLevel.HIGH),
        (0.95, RiskLevel.HIGH),
        (1.0,  RiskLevel.HIGH),
    ])
    def test_risk_level_boundary(self, confidence, expected):
        result, _ = run_session(confidence=confidence)
        assert result.risk_level == expected, (
            f"confidence={confidence} → expected {expected.value}, "
            f"got {result.risk_level.value}"
        )

    def test_risk_level_value_is_schema_enum_string(self):
        """Value must be one of the three strings in comp3_to_comp4.schema.json."""
        for level in RiskLevel:
            assert level.value in {"low", "moderate", "high"}


# ---------------------------------------------------------------------------
# 3. Emotional state heuristic — word count → enum
# ---------------------------------------------------------------------------

class TestEmotionalStateHeuristic:
    """
    _evaluate() uses `last_child_text`, which is the final response from the
    INQUIRE loop (updated on FSMController.run() L79). We supply enough replies
    so that the last reply from the iterator is the one the heuristic sees —
    the iterator provides one reply for INTERCEPT + up to MAX_INQUIRE_ATTEMPTS
    for INQUIRE. We use `next(replies, <last_value>)` semantics by supplying
    exactly (1 + MAX_INQUIRE_ATTEMPTS) replies so the last INQUIRE reply is
    controlled. MAX_INQUIRE_ATTEMPTS = 3, so we supply 4 replies total.
    """

    def test_empty_response_gives_unclear(self):
        # Last INQUIRE reply is "" → split() = [] → len 0 → UNCLEAR
        result, _ = run_session(child_replies=["okay", "no", "no", ""])
        assert result.emotional_state == EmotionalState.UNCLEAR

    def test_one_word_response_gives_defensive(self):
        # Last INQUIRE reply is 1 word → DEFENSIVE
        result, _ = run_session(child_replies=["okay", "no", "no", "no"])
        assert result.emotional_state == EmotionalState.DEFENSIVE

    def test_two_word_response_gives_defensive(self):
        # Last INQUIRE reply is 2 words → still DEFENSIVE (boundary is > 2)
        result, _ = run_session(child_replies=["okay", "no", "no", "not sure"])
        assert result.emotional_state == EmotionalState.DEFENSIVE

    def test_long_response_gives_calm(self):
        # Last INQUIRE reply is > 2 words → CALM
        # The inquire loop exits early when _is_complete() is True (≥4 words),
        # so a 4+ word last reply may end the loop before all 3 attempts.
        # We place the long reply as the first INQUIRE reply so it triggers
        # early exit — last_child_text seen by _evaluate is this reply.
        result, _ = run_session(
            child_replies=["okay", "I was just looking at something interesting online"]
        )
        assert result.emotional_state == EmotionalState.CALM

    def test_emotional_state_value_is_schema_enum_string(self):
        for state in EmotionalState:
            assert state.value in {"calm", "defensive", "distressed", "unclear"}


# ---------------------------------------------------------------------------
# 4. Self-regulation keyword detection
# ---------------------------------------------------------------------------

class TestSelfRegulation:

    @pytest.mark.parametrize("keyword,phrase", [
        ("instead",      "I should have closed it instead"),
        ("next time",    "next time I will look away"),
        ("should have",  "I should have asked my mum"),
        ("i could",      "i could have done something else"),
    ])
    def test_self_regulation_true_when_keyword_present(self, keyword, phrase):
        result, _ = run_session(child_replies=["okay", phrase])
        assert result.self_regulation_shown is True, (
            f"Expected self_regulation_shown=True when child says {phrase!r}"
        )

    def test_self_regulation_false_when_no_keywords(self):
        result, _ = run_session(child_replies=["I was just curious about it"])
        assert result.self_regulation_shown is False


# ---------------------------------------------------------------------------
# 5. Schema validation (jsonschema against comp3_to_comp4.schema.json)
# ---------------------------------------------------------------------------

class TestSchemaValidation:

    def test_validate_passes_on_valid_output(self):
        result, _ = run_session(confidence=0.9)
        # Must not raise
        result.validate()

    def test_validate_passes_for_all_risk_categories(self):
        """
        comp1 schema allows: explicit_visual, violence, self_harm_imagery, unknown_flagged
        comp2 schema allows: hate_speech, cyberbullying, grooming_language, self_harm_language
        Route each category to the correct source component.
        """
        _COMP1_CATEGORIES = {
            RiskCategory.EXPLICIT_VISUAL, RiskCategory.VIOLENCE,
            RiskCategory.SELF_HARM_IMAGERY, RiskCategory.UNKNOWN_FLAGGED,
        }
        _COMP2_CATEGORIES = {
            RiskCategory.HATE_SPEECH, RiskCategory.CYBERBULLYING,
            RiskCategory.GROOMING_LANGUAGE, RiskCategory.SELF_HARM_LANGUAGE,
        }
        for rc in RiskCategory:
            source = (
                "component1_screen_monitoring" if rc in _COMP1_CATEGORIES
                else "component2_hate_speech_detection"
            )
            content_type = "image" if rc in _COMP1_CATEGORIES else "text_message"
            trigger = TriggerPayload.new(
                source_component=source,
                content_type=content_type,
                risk_category=rc,
                confidence_score=0.7,
                app_or_window="browser",
            )
            controller = FSMController()
            result, _ = controller.run(
                trigger,
                respond=lambda _: "I was just looking at something interesting"
            )
            result.validate()  # raises jsonschema.ValidationError on failure

    def test_validate_passes_across_all_confidence_bands(self):
        for confidence in [0.1, 0.5, 0.85]:
            result, _ = run_session(confidence=confidence)
            result.validate()

    def test_to_dict_matches_schema_field_names(self):
        result, _ = run_session()
        d = result.to_dict()
        required = {"session_id", "timestamp", "risk_level",
                    "emotional_state", "self_regulation_shown",
                    "escalation_flag", "dialogue_turns"}
        assert required.issubset(d.keys()), (
            f"Missing fields in to_dict(): {required - d.keys()}"
        )

    def test_risk_level_serialised_as_string_not_enum(self):
        result, _ = run_session()
        d = result.to_dict()
        assert isinstance(d["risk_level"], str), (
            "to_dict() must serialise risk_level as a plain string for JSON Schema"
        )

    def test_emotional_state_serialised_as_string_not_enum(self):
        result, _ = run_session()
        d = result.to_dict()
        assert isinstance(d["emotional_state"], str)

    def test_schema_validation_fails_on_invalid_risk_level(self):
        """Schema gate must catch a bad value — confirms the gate is real."""
        result, _ = run_session()
        d = result.to_dict()
        d["risk_level"] = "critical"   # not in schema enum
        import json
        from pathlib import Path
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "docs" / "interface-contracts" / "comp3_to_comp4.schema.json"
        )
        with open(schema_path) as f:
            schema = json.load(f)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=d, schema=schema)

    def test_schema_validation_fails_when_required_field_missing(self):
        result, _ = run_session()
        d = result.to_dict()
        del d["escalation_flag"]
        import json
        from pathlib import Path
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "docs" / "interface-contracts" / "comp3_to_comp4.schema.json"
        )
        with open(schema_path) as f:
            schema = json.load(f)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=d, schema=schema)

    def test_schema_validation_fails_when_dialogue_turns_is_zero(self):
        """Schema requires dialogue_turns minimum: 1."""
        result, _ = run_session()
        d = result.to_dict()
        d["dialogue_turns"] = 0
        import json
        from pathlib import Path
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "docs" / "interface-contracts" / "comp3_to_comp4.schema.json"
        )
        with open(schema_path) as f:
            schema = json.load(f)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=d, schema=schema)


# ---------------------------------------------------------------------------
# 6. Escalation flag — set in CONTRACT, not EVALUATE
# ---------------------------------------------------------------------------

class TestEscalationFlag:

    def test_high_risk_sets_escalation_true(self):
        result, _ = run_session(confidence=0.95)
        assert result.escalation_flag is True

    def test_moderate_risk_does_not_escalate(self):
        result, _ = run_session(confidence=0.6)
        assert result.escalation_flag is False

    def test_low_risk_does_not_escalate(self):
        result, _ = run_session(confidence=0.2)
        assert result.escalation_flag is False

    def test_escalation_boundary_exactly_at_085(self):
        """0.85 crosses into HIGH — must escalate."""
        result, _ = run_session(confidence=0.85)
        assert result.escalation_flag is True

    def test_no_escalation_just_below_085(self):
        result, _ = run_session(confidence=0.84)
        assert result.escalation_flag is False


# ---------------------------------------------------------------------------
# 7. Full matrix — all 8 RiskCategory × 3 confidence bands → always valid
# ---------------------------------------------------------------------------

_COMP1_CATEGORIES = {
    RiskCategory.EXPLICIT_VISUAL, RiskCategory.VIOLENCE,
    RiskCategory.SELF_HARM_IMAGERY, RiskCategory.UNKNOWN_FLAGGED,
}


def _make_trigger_for(risk_category: RiskCategory, confidence: float) -> TriggerPayload:
    """Route to the correct source_component matching each schema's enum."""
    if risk_category in _COMP1_CATEGORIES:
        return TriggerPayload.new(
            source_component="component1_screen_monitoring",
            content_type="image",
            risk_category=risk_category,
            confidence_score=confidence,
            app_or_window="browser",
        )
    else:
        return TriggerPayload.new(
            source_component="component2_hate_speech_detection",
            content_type="text_message",
            risk_category=risk_category,
            confidence_score=confidence,
        )


class TestFullMatrix:

    @pytest.mark.parametrize("risk_category", list(RiskCategory))
    @pytest.mark.parametrize("confidence", [0.2, 0.7, 0.9])
    def test_full_matrix_always_schema_valid(self, risk_category, confidence):
        trigger = _make_trigger_for(risk_category, confidence)
        controller = FSMController()
        result, _ = controller.run(
            trigger,
            respond=lambda _: "I was just looking at something interesting"
        )
        result.validate()   # raises jsonschema.ValidationError on failure
        assert result.dialogue_turns >= 1
        assert isinstance(result.risk_level, RiskLevel)
        assert isinstance(result.emotional_state, EmotionalState)
        assert isinstance(result.self_regulation_shown, bool)
        assert isinstance(result.escalation_flag, bool)

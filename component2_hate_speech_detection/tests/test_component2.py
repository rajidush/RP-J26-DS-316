import json
import sys
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component2_hate_speech_detection.src.component2 import analyze_text
from component2_hate_speech_detection.src.keywords import (
    list_family_ids,
    screen_keywords,
)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "interface-contracts"
    / "comp2_to_comp3.schema.json"
)


def _schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def test_stub_emits_required_fields():
    payload = analyze_text()
    assert payload is not None
    for field in (
        "session_id",
        "timestamp",
        "source_component",
        "content_type",
        "risk_category",
        "confidence_score",
    ):
        assert field in payload
    assert payload["source_component"] == "component2_hate_speech_detection"


def test_stub_validates_against_shared_schema():
    payload = analyze_text()
    jsonschema.validate(instance=payload, schema=_schema())


def test_invalid_risk_category_would_fail_schema():
    bad = analyze_text()
    assert bad is not None
    bad["risk_category"] = "not_a_real_category"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=_schema())


def test_keyword_screen_flags_cyberbullying():
    result = screen_keywords("You are so stupid and nobody likes you")
    assert result.matched is True
    assert result.risk_category == "cyberbullying"
    assert "cyberbullying_insults" in result.matched_families
    assert 0.0 < result.confidence_score <= 1.0


def test_keyword_screen_flags_grooming_without_leaking_raw_text():
    result = screen_keywords("this is our little secret, don't tell your parents")
    assert result.matched is True
    assert result.risk_category == "grooming_language"
    # Evidence is family IDs only — raw text must not appear in the result.
    assert "secret" not in str(result.matched_families)


def test_keyword_screen_clean_chat_returns_no_match():
    result = screen_keywords("gg well played, see you in the next match")
    assert result.matched is False
    assert result.risk_category is None
    assert result.confidence_score == 0.0


def test_analyze_text_live_path_emits_schema_valid_trigger():
    payload = analyze_text(
        "go back to your country",
        platform="group_chat",
    )
    assert payload is not None
    jsonschema.validate(instance=payload, schema=_schema())
    assert payload["risk_category"] == "hate_speech"
    assert payload["context_metadata"]["screen_stage"] == "keyword_layer"
    assert "hate_speech_identity_attack" in payload["context_metadata"]["matched_families"]
    # Outbound payload must never carry the verbatim input.
    dumped = json.dumps(payload)
    assert "go back to your country" not in dumped


def test_analyze_text_live_path_returns_none_when_clean():
    assert analyze_text("want to play minecraft later?") is None


def test_family_ids_are_stable_for_audit():
    ids = list_family_ids()
    assert "cyberbullying_insults" in ids
    assert "grooming_secrecy" in ids

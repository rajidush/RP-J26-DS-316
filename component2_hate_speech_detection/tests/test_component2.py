import json
import sys
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component2_hate_speech_detection.src import component2
from component2_hate_speech_detection.src.component2 import analyze, analyze_text
from component2_hate_speech_detection.src.engine.analyzer import Analyzer
from component2_hate_speech_detection.src.engine.text_scorer import TextScorer

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "interface-contracts"
    / "comp2_to_comp3.schema.json"
)


@pytest.fixture(autouse=True)
def lexicon_only_engine():
    """Public-API tests run on the lexicon so they stay fast and model-free."""
    component2.set_analyzer(Analyzer(scorer=TextScorer(use_heads=False)))
    yield
    component2.set_analyzer(None)


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
    jsonschema.validate(instance=analyze_text(), schema=_schema())


def test_invalid_risk_category_would_fail_schema():
    bad = analyze_text()
    bad["risk_category"] = "not_a_real_category"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=_schema())


def test_live_path_emits_schema_valid_trigger_without_raw_text():
    text = "nobody likes you go back to your country"
    payload = analyze_text(text, platform="group_chat", age=9)
    assert payload is not None
    jsonschema.validate(instance=payload, schema=_schema())
    assert payload["risk_category"] == "cyberbullying"
    assert payload["context_metadata"]["rung"] == "L3"
    assert "bullying:exclusion" in payload["context_metadata"]["matched_families"]
    assert text not in json.dumps(payload)


def test_live_path_returns_none_when_clean():
    assert analyze_text("want to play minecraft later?") is None


def test_live_path_respects_age_thresholds():
    # Mid-band bullying (0.58) alerts for a 9-year-old and stays below a 15-year-old's threshold.
    assert analyze_text("nobody likes you", age=9) is not None
    assert analyze_text("nobody likes you", age=15) is None


def test_voice_transcripts_are_labelled_as_such():
    payload = analyze_text("you should kys", content_type="voice_transcript")
    assert payload["content_type"] == "voice_transcript"


def test_child_reporting_abuse_does_not_wake_c3():
    assert analyze_text("someone in the group chat told me to kys and i'm scared") is None


def test_analyze_returns_full_verdict():
    v = analyze("you should kys", age=10)
    assert v.rung == "L3" and v.top_category == "threat" and v.explanation

import json
import sys
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component2_hate_speech_detection.src.component2 import analyze_text

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
    """Guard: schema rejects values outside the agreed enum."""
    bad = analyze_text()
    bad["risk_category"] = "not_a_real_category"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=_schema())

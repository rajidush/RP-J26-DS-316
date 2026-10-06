import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component4_profiling_xai.src.behavioral_anomaly import load_artifacts, predict_behavioral_anomaly
from component4_profiling_xai.src.component4 import detect_behavioral_anomaly, generate_report

MOCK_INPUTS = Path(__file__).resolve().parents[1] / "mock_inputs"


def test_report_contains_key_fields():
    with open(MOCK_INPUTS / "sample_session_output.json") as f:
        payload = json.load(f)
    report = generate_report(payload)
    assert "Risk level:       high" in report
    assert payload["session_id"] in report


# --- Function 2 through the Component 4 interface ---------------------------
# Records come from the saved scaler's training statistics (see
# test_behavioral_anomaly.py); detailed Function 2 checks live there.

@pytest.fixture(scope="module")
def behavioral_records():
    _, scaler, feature_names = load_artifacts()
    normal = {n: float(m) for n, m in zip(feature_names, scaler.mean_)}
    unusual = {
        n: float(m + 4 * s)
        for n, m, s in zip(feature_names, scaler.mean_, scaler.scale_)
    }
    return normal, unusual, feature_names


def test_detect_behavioral_anomaly_returns_four_field_result(behavioral_records):
    normal, _, _ = behavioral_records
    result = detect_behavioral_anomaly(normal)
    assert list(result) == ["status", "anomaly_flag", "decision_score", "anomaly_score"]
    assert result == predict_behavioral_anomaly(normal)


def test_detect_behavioral_anomaly_normal_and_anomaly_mapping(behavioral_records):
    normal, unusual, _ = behavioral_records
    normal_result = detect_behavioral_anomaly(normal)
    unusual_result = detect_behavioral_anomaly(unusual)
    assert (normal_result["status"], normal_result["anomaly_flag"]) == ("Normal", 0)
    assert (unusual_result["status"], unusual_result["anomaly_flag"]) == ("Anomaly", 1)
    for result in (normal_result, unusual_result):
        assert result["anomaly_score"] == -result["decision_score"]


def test_detect_behavioral_anomaly_preserves_validation_errors(behavioral_records):
    normal, _, feature_names = behavioral_records
    missing = feature_names[0]
    with pytest.raises(ValueError, match=missing):
        detect_behavioral_anomaly({k: v for k, v in normal.items() if k != missing})
    with pytest.raises(ValueError, match=feature_names[1]):
        detect_behavioral_anomaly({**normal, feature_names[1]: float("nan")})
    with pytest.raises(TypeError):
        detect_behavioral_anomaly([1.0] * len(feature_names))


def test_report_output_unchanged_by_function2_integration():
    with open(MOCK_INPUTS / "sample_session_output.json") as f:
        payload = json.load(f)
    # Exact output of generate_report before Function 2 was integrated.
    expected = (
        "=== Parent-Facing Summary ===\n"
        "Session:          sess-demo-0001\n"
        "Risk level:       high\n"
        "Emotional state:  calm\n"
        "Self-regulation shown: True\n"
        "Escalated to you: True\n"
        "Conversation length: 3 turns\n"
        "Summary: 3 turns; risk_category=explicit_visual"
    )
    assert generate_report(payload) == expected

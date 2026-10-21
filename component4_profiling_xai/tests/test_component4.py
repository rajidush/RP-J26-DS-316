import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component4_profiling_xai.src.component4 import generate_report

MOCK_INPUTS = Path(__file__).resolve().parents[1] / "mock_inputs"


def test_report_contains_key_fields():
    with open(MOCK_INPUTS / "sample_session_output.json") as f:
        payload = json.load(f)
    report = generate_report(payload)
    assert "Risk level:       high" in report
    assert payload["session_id"] in report


def test_report_matches_expected_output():
    with open(MOCK_INPUTS / "sample_session_output.json") as f:
        payload = json.load(f)
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

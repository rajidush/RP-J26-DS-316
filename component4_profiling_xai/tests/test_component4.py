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

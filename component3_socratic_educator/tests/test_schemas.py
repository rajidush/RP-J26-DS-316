"""
Schema validation of Component 3's outbound record (comp3_to_comp4.schema.json).
Run with: pytest component3_socratic_educator/tests -v
"""
import sys
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component3_socratic_educator.src.schemas import EmotionalState, EvaluateOutput, RiskLevel


def _record(timestamp):
    return EvaluateOutput(session_id="S-1", risk_level=RiskLevel.LOW, emotional_state=EmotionalState.CALM,
                          self_regulation_shown=False, escalation_flag=False, dialogue_turns=1, timestamp=timestamp)


def test_a_record_with_an_iso_timestamp_validates():
    _record("2026-10-08T16:00:00+00:00").validate()


@pytest.mark.parametrize("timestamp", ["yesterday", "2026-10-08", "08/10/2026 16:00"])
def test_a_timestamp_that_is_not_a_date_time_is_rejected(timestamp):
    # 8 Oct finding: without a format checker jsonschema ignores "format": "date-time",
    # and Component 4 then fails parsing the timestamp. The check also needs the
    # rfc3339-validator package; without it this test fails instead of passing silently.
    with pytest.raises(jsonschema.ValidationError):
        _record(timestamp).validate()

"""
Component 3 -> Component 4 adapter (plan, Thu 8 Oct).

Component 3 emits one record per session, shaped by
docs/interface-contracts/comp3_to_comp4.schema.json. Component 4 Function 1
(emotional pattern clustering, component4_profiling_xai/src/emotional_pattern.py)
groups sessions per child, so it also needs child_id, which the schema does not
carry. Rather than change the shared schema, the adapter adds child_id from
device configuration: one device runs one child profile. This interface gap is
logged in docs/interface-contracts/CHANGELOG.md for the group to decide.

The row is the record plus child_id, the same shape as the educator_sessions.csv
Component 4 was trained on (risk_level and dialogue_summary ride along unread).
Pure dicts, so Component 3 needs no pandas: build the DataFrame on Component 4's side.
"""
from __future__ import annotations

from component3_socratic_educator.src.schemas import EvaluateOutput

# The child profile this device belongs to. A hardcoded placeholder until there
# is a real device-configuration source (interface gap, CHANGELOG 2026-10-11).
DEVICE_CHILD_ID = "device-child-01"


def to_emotional_pattern_row(evaluate_output: EvaluateOutput, child_id: str = DEVICE_CHILD_ID) -> dict:
    """Validate the record against the comp3_to_comp4 schema, then add child_id.

    Raises jsonschema.ValidationError for a record that breaks the schema and
    ValueError for a blank child_id, so neither reaches Component 4.
    """
    evaluate_output.validate()
    if not isinstance(child_id, str) or not child_id.strip():
        raise ValueError("child_id is required: Component 4 groups sessions by child")
    return {"child_id": child_id, **evaluate_output.to_dict()}

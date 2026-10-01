"""
Component 2 (Hate-Speech Detection) -- owned by IT23209152.

Step 1: stub loads a mock trigger and validates it against
docs/interface-contracts/comp2_to_comp3.schema.json before returning.
Real classifier comes in later steps; keep the return shape identical.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

MOCK_INPUTS = Path(__file__).resolve().parents[1] / "mock_inputs"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "interface-contracts"
    / "comp2_to_comp3.schema.json"
)


def _load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def analyze_text(mock_file: str = "sample_analysis.json") -> dict:
    with open(MOCK_INPUTS / mock_file) as f:
        payload = json.load(f)
    # fresh session id + timestamp each run, so integration demos don't collide
    payload["session_id"] = str(uuid.uuid4())
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    jsonschema.validate(instance=payload, schema=_load_schema())
    return payload

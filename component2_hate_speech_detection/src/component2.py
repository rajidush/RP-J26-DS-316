"""
Component 2 (Hate-Speech Detection) -- STUB owned by IT23209152.

Same pattern as Component 1's stub: emits a schema-valid TriggerPayload
dict from a mock file so the full system is demonstrable before the real
NLP classifier is finished.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

MOCK_INPUTS = Path(__file__).resolve().parents[1] / "mock_inputs"


def analyze_text(mock_file: str = "sample_analysis.json") -> dict:
    with open(MOCK_INPUTS / mock_file) as f:
        payload = json.load(f)
    payload["session_id"] = str(uuid.uuid4())
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    return payload

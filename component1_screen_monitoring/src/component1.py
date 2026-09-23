"""
Component 1 (Screen Monitoring) -- STUB owned by IT23377844.

This stub exists so the whole system can be demonstrated end-to-end on a
laptop before Component 1's real detection model is finished. It reads a
fixed mock frame description and emits a TriggerPayload-shaped dict that
matches docs/interface-contracts/comp1_to_comp3.schema.json exactly.

Replace `simulate_screen_capture()` with a real capture+classification
call when Component 1 is ready -- Component 3 does not need to change at
all, as long as the returned dict still matches the schema.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

MOCK_INPUTS = Path(__file__).resolve().parents[1] / "mock_inputs"


def simulate_screen_capture(mock_file: str = "sample_trigger.json") -> dict:
    with open(MOCK_INPUTS / mock_file) as f:
        payload = json.load(f)
    # fresh session id + timestamp each run, so integration demos don't collide
    payload["session_id"] = str(uuid.uuid4())
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    return payload

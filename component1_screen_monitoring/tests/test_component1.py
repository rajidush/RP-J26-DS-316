import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.component1 import simulate_screen_capture


def test_stub_emits_required_fields():
    payload = simulate_screen_capture()
    for field in ("session_id", "timestamp", "source_component", "content_type", "risk_category", "confidence_score"):
        assert field in payload
    assert payload["source_component"] == "component1_screen_monitoring"

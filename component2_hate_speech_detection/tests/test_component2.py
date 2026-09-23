import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component2_hate_speech_detection.src.component2 import analyze_text


def test_stub_emits_required_fields():
    payload = analyze_text()
    for field in ("session_id", "timestamp", "source_component", "content_type", "risk_category", "confidence_score"):
        assert field in payload
    assert payload["source_component"] == "component2_hate_speech_detection"

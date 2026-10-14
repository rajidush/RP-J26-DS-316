"""
Tests for Function 1's output path: RawFrame + Detection → schema-valid payload,
and the continuous monitoring loop.

Uses fake detections and fake frames, so the real model and screen are not needed.
"""
import copy
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft7Validator, ValidationError
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.capture import RawFrame
from component1_screen_monitoring.src.component1 import (
    build_payload,
    capture_and_classify,
    run_monitoring_loop,
    simulate_screen_capture,
    validate_payload,
)
from component1_screen_monitoring.src.config import SCHEMA_PATH, THRESHOLD

T0 = datetime(2026, 10, 7, 10, 0, 0, tzinfo=timezone.utc)


def _frame(seconds=0.0):
    ts = (T0 + timedelta(seconds=seconds)).isoformat()
    return RawFrame(image=Image.new("RGB", (64, 40)), timestamp=ts, frame_reference=f"frame_{seconds}")


def _detection(p_violent, threshold=THRESHOLD):
    flagged = p_violent >= threshold
    return SimpleNamespace(category="violence" if flagged else "safe",
                           confidence=p_violent, flagged=flagged, inference_ms=12.0,
                           region="full", region_box=None, n_crops=1)


def _independent_validator():
    """Validate with a fresh validator built here, not the one under test."""
    schema = json.loads(SCHEMA_PATH.read_text())
    return Draft7Validator(schema, format_checker=Draft7Validator.FORMAT_CHECKER)


# ---------------------------------------------------------------------------
# build_payload / validate_payload
# ---------------------------------------------------------------------------

def test_flagged_payload_matches_schema():
    payload = build_payload(_frame(), _detection(0.93), session_id="abc", duration_visible_ms=4000)
    _independent_validator().validate(payload)
    assert payload == {
        "session_id": "abc",
        "timestamp": T0.isoformat(),
        "source_component": "component1_screen_monitoring",
        "content_type": "image",
        "risk_category": "violence",
        "confidence_score": 0.93,
        "context_metadata": {"duration_visible_ms": 4000},
    }


def test_flagged_payload_gets_fresh_session_id_by_default():
    a = build_payload(_frame(), _detection(0.9))
    b = build_payload(_frame(), _detection(0.9))
    assert a["session_id"] != b["session_id"]


def test_not_flagged_frame_gives_no_payload():
    assert build_payload(_frame(), _detection(0.2)) is None


def test_threshold_boundary_is_inclusive():
    assert build_payload(_frame(), _detection(THRESHOLD)) is not None


def test_mock_payload_still_valid():
    validate_payload(simulate_screen_capture())


def _valid():
    return build_payload(_frame(), _detection(0.9), session_id="s")


@pytest.mark.parametrize("mutate", [
    lambda p: p.pop("session_id"),
    lambda p: p.pop("confidence_score"),
    lambda p: p.update(risk_category="safe"),
    lambda p: p.update(content_type="screenshot"),
    lambda p: p.update(source_component="component2_hate_speech_detection"),
    lambda p: p.update(confidence_score=1.5),
    lambda p: p.update(confidence_score=-0.1),
    lambda p: p.update(confidence_score="0.9"),
    lambda p: p.update(timestamp="not-a-date"),
    lambda p: p["context_metadata"].update(duration_visible_ms=1.5),
], ids=["no_session_id", "no_confidence", "safe_category", "bad_content_type",
        "wrong_source", "confidence_gt_1", "confidence_lt_0", "confidence_str",
        "bad_timestamp", "duration_not_int"])
def test_invalid_payloads_rejected(mutate):
    payload = copy.deepcopy(_valid())
    mutate(payload)
    with pytest.raises(ValidationError):
        validate_payload(payload)


# ---------------------------------------------------------------------------
# capture_and_classify
# ---------------------------------------------------------------------------

class FakeDetector:
    """Returns p_violent values from a list, one per predict()/predict_regions() call."""

    threshold = THRESHOLD

    def __init__(self, probs):
        self._probs = iter(probs)
        self.region_calls = []                 # (prev_image, strategy) per predict_regions call

    def predict(self, image):
        return _detection(next(self._probs))

    def predict_regions(self, image, prev_image=None, strategy="tiles+motion"):
        self.region_calls.append((prev_image, strategy))
        return self.predict(image)


def test_capture_and_classify_flagged_and_not():
    assert capture_and_classify(FakeDetector([0.1]), capture_fn=_frame) is None
    payload = capture_and_classify(FakeDetector([0.8]), capture_fn=_frame)
    _independent_validator().validate(payload)


# ---------------------------------------------------------------------------
# run_monitoring_loop
# ---------------------------------------------------------------------------

def _clock_frames():
    """capture_fn yielding frames 2 s apart."""
    n = {"i": 0}

    def capture():
        f = _frame(seconds=2 * n["i"])
        n["i"] += 1
        return f
    return capture


def test_loop_stops_after_max_iterations():
    sleeps, payloads = [], []
    stats = run_monitoring_loop(FakeDetector([0.1] * 10), interval=2.0, max_iterations=4,
                                on_payload=payloads.append, capture_fn=_clock_frames(),
                                sleep_fn=sleeps.append)
    assert stats == {"frames": 4, "payloads": 0, "errors": 0}
    assert len(sleeps) == 3                    # no sleep after the last frame
    assert payloads == []


def test_loop_groups_consecutive_flagged_frames_into_incidents():
    payloads = []
    probs = [0.9, 0.95, 0.1, 0.8, 0.85]       # incident A (2 frames), safe, incident B (2)
    stats = run_monitoring_loop(FakeDetector(probs), max_iterations=5,
                                on_payload=payloads.append, capture_fn=_clock_frames(),
                                sleep_fn=lambda s: None)
    assert stats["payloads"] == 4
    v = _independent_validator()
    for p in payloads:
        v.validate(p)
    a1, a2, b1, b2 = payloads
    assert a1["session_id"] == a2["session_id"] != b1["session_id"] == b2["session_id"]
    assert [p["context_metadata"]["duration_visible_ms"] for p in payloads] == [0, 2000, 0, 2000]


def test_loop_logs_not_flagged_at_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="component1_screen_monitoring.src.component1"):
        run_monitoring_loop(FakeDetector([0.2]), max_iterations=1,
                            capture_fn=_clock_frames(), sleep_fn=lambda s: None)
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("not flagged" in r.getMessage() for r in debug)


def test_loop_exits_cleanly_on_ctrl_c():
    calls = {"n": 0}

    def capture():
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return _frame(calls["n"])

    stats = run_monitoring_loop(FakeDetector([0.1] * 5), max_iterations=None,
                                capture_fn=capture, sleep_fn=lambda s: None)
    assert stats["frames"] == 2


def test_loop_survives_a_failing_frame():
    class Flaky(FakeDetector):
        def predict(self, image):
            p = next(self._probs)
            if p is None:
                raise RuntimeError("boom")
            return _detection(p)

    stats = run_monitoring_loop(Flaky([0.1, None, 0.9]), max_iterations=3,
                                on_payload=lambda p: None, capture_fn=_clock_frames(),
                                sleep_fn=lambda s: None)
    assert stats == {"frames": 3, "payloads": 1, "errors": 1}


def test_loop_passes_small_previous_frame_and_strategy():
    def capture():
        return RawFrame(image=Image.new("RGB", (1440, 900), "white"),
                        timestamp=T0.isoformat(), frame_reference="f")

    det = FakeDetector([0.1, 0.1, 0.1])
    run_monitoring_loop(det, max_iterations=3, capture_fn=capture, sleep_fn=lambda s: None,
                        strategy="tiles")
    prevs = [p for p, _ in det.region_calls]
    assert prevs[0] is None                    # first frame has no previous frame
    assert all(p.mode == "L" and p.width <= 160 for p in prevs[1:])   # downscaled copy only
    assert {s for _, s in det.region_calls} == {"tiles"}

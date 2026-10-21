"""
Tests for Function 2's decision logic (response_policy.py) and the overlay
state machine / latency log / mock demo (response.py). No GUI: a fake overlay
and a fake clock stand in for Tk and time.
"""
import json
import logging
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.config import (BLOCK_THRESHOLD, CLEAR_FRAMES,
                                                      DISMISS_GRACE_S, MIN_HOLD_S)
from component1_screen_monitoring.src.response import (MOCK_SCRIPT, E2ELatencyLog,
                                                       ResponseController, mock_payload,
                                                       play_mock)
from component1_screen_monitoring.src.response_policy import decide, stronger

T0 = datetime(2026, 10, 14, 10, 0, 0, tzinfo=timezone.utc)


def _payload(category="violence", conf=0.70, sid="incident-a"):
    return mock_payload(category, conf, sid, T0.isoformat())


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category, conf, expected", [
    ("explicit_visual", 0.51, "block"),
    ("self_harm_imagery", 0.60, "block"),
    ("violence", BLOCK_THRESHOLD, "block"),          # boundary is inclusive
    ("violence", 0.99, "block"),
    ("violence", BLOCK_THRESHOLD - 0.01, "blur"),
    ("violence", 0.50, "blur"),
    ("unknown_flagged", 0.95, "blur"),
])
def test_decide_rules(category, conf, expected):
    assert decide(_payload(category, conf)) == expected


@pytest.mark.parametrize("override", ["blur", "block"])
def test_decide_override_forces_action(override):
    for category, conf in [("explicit_visual", 0.9), ("violence", 0.6), ("unknown_flagged", 0.7)]:
        assert decide(_payload(category, conf), override) == override


def test_decide_rejects_bad_override():
    with pytest.raises(ValueError):
        decide(_payload(), "maybe")


@pytest.mark.parametrize("mutate", [
    lambda p: p.pop("risk_category"),
    lambda p: p.update(risk_category="safe"),
    lambda p: p.update(confidence_score=1.4),
    lambda p: p.update(source_component="someone_else"),
    lambda p: p.update(timestamp="yesterday"),
], ids=["missing_field", "safe_category", "confidence_gt_1", "wrong_source", "bad_timestamp"])
def test_invalid_payload_is_rejected_and_logged(mutate, caplog):
    p = _payload()
    mutate(p)
    with caplog.at_level(logging.WARNING):
        assert decide(p) is None
        assert decide(p, "block") is None                # override never bypasses validation
    assert "invalid payload" in caplog.text


def test_decide_non_dict_is_rejected():
    assert decide(None) is None and decide("violence") is None


def test_stronger():
    assert stronger("blur", "block") == stronger("block", "blur") == "block"
    assert stronger(None, "blur") == "blur" and stronger(None, None) is None


# ---------------------------------------------------------------------------
# Controller (fake overlay + fake clock)
# ---------------------------------------------------------------------------

class FakeOverlay:
    def __init__(self, wall):
        self.calls, self.wall = [], wall

    def show_blur(self, image, caption="", on_visible=None):
        self.calls.append("blur")
        if on_visible:
            on_visible(self.wall())

    def show_block(self, message="", on_visible=None):
        self.calls.append("block")
        if on_visible:
            on_visible(self.wall())

    def hide(self):
        self.calls.append("hide")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


FRAME = SimpleNamespace(image=Image.new("RGB", (32, 20)), timestamp=T0.isoformat())


@pytest.fixture
def ctl():
    clock = Clock()
    overlay = FakeOverlay(lambda: T0.timestamp() + clock.t)
    c = ResponseController(overlay, clock=clock, wall_clock=lambda: T0.timestamp() + clock.t)
    return c, overlay, clock


def test_defaults_come_from_config(ctl):
    c, _, _ = ctl
    assert (c.min_hold_s, c.clear_frames, c.dismiss_grace_s, c.block_threshold) == \
        (MIN_HOLD_S, CLEAR_FRAMES, DISMISS_GRACE_S, BLOCK_THRESHOLD)


def test_flag_shows_overlay_in_decided_mode(ctl):
    c, overlay, _ = ctl
    assert c.on_payload(_payload("violence", 0.7), FRAME) == "blur"
    assert overlay.calls == ["blur"] and c.visible


def test_hide_needs_min_hold_and_clear_frames(ctl):
    c, overlay, clock = ctl
    c.on_payload(_payload(), FRAME)                       # t=0
    clock.t = 1; assert not c.on_clear_frame()
    clock.t = 2; assert not c.on_clear_frame()            # 2 clear frames, but only 2 s
    clock.t = 4.9; assert not c.on_clear_frame()
    clock.t = 5.0; assert c.on_clear_frame()              # ≥ MIN_HOLD_S and ≥ CLEAR_FRAMES
    assert overlay.calls[-1] == "hide" and not c.visible


def test_hold_long_enough_but_not_enough_clear_frames(ctl):
    c, _, clock = ctl
    c.on_payload(_payload(), FRAME)
    clock.t = 10
    assert not c.on_clear_frame()                         # only 1 clear frame
    clock.t = 11
    assert c.on_clear_frame()


def test_new_payload_extends_hold_and_resets_clear_count(ctl):
    c, overlay, clock = ctl
    c.on_payload(_payload(), FRAME)                       # t=0
    clock.t = 1; c.on_clear_frame()
    clock.t = 4; c.on_payload(_payload(), FRAME)          # extends: hold now until t=9
    clock.t = 6; assert not c.on_clear_frame()
    clock.t = 8; assert not c.on_clear_frame()            # 2 clear, but < 5 s since t=4
    clock.t = 9; assert c.on_clear_frame()
    assert overlay.calls.count("blur") == 2 and overlay.calls[-1] == "hide"   # blur refreshed once


def test_escalates_blur_to_block_but_never_downgrades(ctl):
    c, overlay, _ = ctl
    assert c.on_payload(_payload("violence", 0.70), FRAME) == "blur"
    assert c.on_payload(_payload("violence", 0.92), FRAME) == "block"
    assert c.on_payload(_payload("violence", 0.60), FRAME) == "block"
    assert overlay.calls == ["blur", "block"]


def test_forced_mode_override(ctl):
    c, overlay, _ = ctl
    c.override = "block"
    assert c.on_payload(_payload("violence", 0.55), FRAME) == "block"
    c2 = ResponseController(FakeOverlay(lambda: 0.0), override="blur", clock=Clock())
    assert c2.on_payload(_payload("explicit_visual", 0.99), FRAME) == "blur"


def test_invalid_payload_ignored_by_controller(ctl):
    c, overlay, _ = ctl
    bad = _payload()
    bad["confidence_score"] = 7
    assert c.on_payload(bad, FRAME) is None and overlay.calls == []


def test_dismiss_grace_suppresses_same_incident(ctl):
    c, overlay, clock = ctl
    c.on_payload(_payload(sid="A"), FRAME)
    clock.t = 1; c.on_dismiss()
    assert overlay.calls == ["blur", "hide"] and not c.visible
    clock.t = 5; assert c.on_payload(_payload(sid="A", conf=0.7), FRAME) is None     # suppressed
    clock.t = 10.9; assert c.on_payload(_payload(sid="A", conf=0.8), FRAME) is None  # still in grace
    clock.t = 11.0; assert c.on_payload(_payload(sid="A", conf=0.7), FRAME) == "blur"  # grace over


def test_dismiss_grace_overridden_by_high_confidence(ctl):
    c, _, clock = ctl
    c.on_payload(_payload(sid="A"), FRAME)
    clock.t = 1; c.on_dismiss()
    clock.t = 2; assert c.on_payload(_payload(sid="A", conf=BLOCK_THRESHOLD), FRAME) == "block"


def test_dismiss_grace_does_not_apply_to_new_incident(ctl):
    c, _, clock = ctl
    c.on_payload(_payload(sid="A"), FRAME)
    clock.t = 1; c.on_dismiss()
    clock.t = 2; assert c.on_payload(_payload(sid="B"), FRAME) == "blur"


def test_dismiss_when_hidden_is_harmless(ctl):
    c, overlay, _ = ctl
    c.on_dismiss()
    assert overlay.calls == []


def test_clear_frames_while_hidden_do_nothing(ctl):
    c, overlay, _ = ctl
    assert not c.on_clear_frame() and overlay.calls == []


# ---------------------------------------------------------------------------
# Latency log
# ---------------------------------------------------------------------------

def test_latency_log_fields_and_append(tmp_path, ctl):
    _, _, clock = ctl
    path = tmp_path / "e2e.json"
    path.write_text(json.dumps([{"previous": "run"}]))
    lat = E2ELatencyLog(path, source="mock", strategy="tiles+motion")
    overlay = FakeOverlay(lambda: T0.timestamp() + 0.250)            # visible 250 ms after capture
    c = ResponseController(overlay, latency=lat, clock=clock)
    c.on_payload(_payload("violence", 0.7, sid="A"), FRAME, emitted_at=T0.timestamp() + 0.200)
    c.on_payload(_payload("violence", 0.95, sid="A"), FRAME, emitted_at=T0.timestamp() + 0.200)

    assert lat.save() == 2
    saved = json.loads(path.read_text())
    assert saved[0] == {"previous": "run"} and len(saved) == 3     # appended, not overwritten
    rec = saved[1]
    assert rec["action"] == "blur" and saved[2]["action"] == "block"
    assert rec["session_id"] == "A" and rec["source"] == "mock" and rec["strategy"] == "tiles+motion"
    assert rec["frame_captured_at"] == T0.isoformat()
    assert rec["capture_to_payload_ms"] == 200.0
    assert rec["payload_to_visible_ms"] == 50.0
    assert rec["capture_to_visible_ms"] == 250.0
    for key in ("payload_emitted_at", "overlay_visible_at", "risk_category", "confidence_score"):
        assert rec[key] is not None
    assert "2 appearance(s)" in lat.summary() and "250 ms" in lat.summary()


def test_latency_log_skips_records_never_shown(tmp_path):
    lat = E2ELatencyLog(tmp_path / "e2e.json")
    lat.start(_payload(), "blur", T0.isoformat(), T0.timestamp())   # no visible() call
    assert lat.save() == 0 and not (tmp_path / "e2e.json").exists()
    assert "no overlay" in E2ELatencyLog(tmp_path / "x.json").summary()


# ---------------------------------------------------------------------------
# Mock demo through the same controller
# ---------------------------------------------------------------------------

def test_mock_script_demonstrates_blur_extend_hide_block():
    clock = Clock()
    overlay = FakeOverlay(lambda: clock.t)
    c = ResponseController(overlay, clock=clock)

    def advance(s):
        clock.t += s

    def frame():
        return SimpleNamespace(image=Image.new("RGB", (32, 20)),
                               timestamp=(T0 + timedelta(seconds=clock.t)).isoformat())

    play_mock(c, frame, sleep_fn=advance)
    shows = [x for x in overlay.calls if x != "hide"]
    assert overlay.calls[0] == "blur"                      # violence 0.70 → blur
    assert "block" in shows                                 # 0.92 / explicit → block
    assert overlay.calls.count("hide") == 2                 # both incidents end
    assert overlay.calls[-1] == "hide" and not c.visible
    first_hide = overlay.calls.index("hide")
    assert overlay.calls[:first_hide] == ["blur", "blur"]  # shown + refreshed (extended), then hidden
    assert len(MOCK_SCRIPT) * 1.0 == clock.t


def test_mock_payloads_are_schema_valid():
    from component1_screen_monitoring.src.component1 import validate_payload
    sid = str(uuid.uuid4())
    for step in MOCK_SCRIPT:
        if step:
            validate_payload(mock_payload(*step, sid))

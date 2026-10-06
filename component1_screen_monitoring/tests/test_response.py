import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.response import ResponseController


class FakeOverlay:
    def __init__(self):
        self.calls = []

    def show_blur(self, image, caption=""):
        self.calls.append(("blur", caption))

    def show_block(self, message=""):
        self.calls.append(("block", message))

    def hide(self):
        self.calls.append(("hide", None))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _result(flagged, category="violence", confidence=0.95):
    return {"flagged": flagged, "category": category, "confidence": confidence,
            "timestamp": "2026-10-06T00:00:00+00:00", "frame_reference": "frame_x"}


def _make(results, mode="blur", hold=5.0):
    """Controller whose detector returns *results* in order."""
    it = iter(results)
    detected = []

    def detector(frame):
        detected.append(frame)
        return next(it)

    overlay, clock = FakeOverlay(), FakeClock()
    ctrl = ResponseController(overlay, detector, mode=mode, hold_seconds=hold, clock=clock)
    return ctrl, overlay, clock, detected


FRAME = SimpleNamespace(image=object())


def test_safe_frame_does_nothing():
    ctrl, overlay, _, _ = _make([_result(False, "non_violence")])
    assert ctrl.on_frame(FRAME)["flagged"] is False
    assert overlay.calls == []
    assert not ctrl.active


@pytest.mark.parametrize("mode", ["blur", "block"])
def test_flagged_frame_shows_overlay_in_mode(mode):
    ctrl, overlay, _, _ = _make([_result(True)], mode=mode)
    ctrl.on_frame(FRAME)
    assert ctrl.active
    assert overlay.calls[0][0] == mode
    assert "violence" in overlay.calls[0][1]


def test_detection_skipped_while_holding_then_hides():
    ctrl, overlay, clock, detected = _make([_result(True), _result(False)], hold=5.0)
    ctrl.on_frame(FRAME)                    # t=0  flagged → show
    clock.t = 2.0
    assert ctrl.on_frame(FRAME) is None     # t=2  still holding, no detection
    assert len(detected) == 1
    clock.t = 5.0
    assert ctrl.on_frame(FRAME) is None     # t=5  hold over → hide, frame skipped
    assert overlay.calls[-1] == ("hide", None)
    assert not ctrl.active
    clock.t = 6.0
    assert ctrl.on_frame(FRAME)["flagged"] is False   # detection resumed
    assert len(detected) == 2


def test_retriggers_if_still_harmful_after_hold():
    ctrl, overlay, clock, _ = _make([_result(True), _result(True)], hold=1.0)
    ctrl.on_frame(FRAME)
    clock.t = 1.0
    ctrl.on_frame(FRAME)                    # hide
    clock.t = 2.0
    ctrl.on_frame(FRAME)                    # flagged again → show
    assert [c[0] for c in overlay.calls] == ["blur", "hide", "blur"]
    assert ctrl.active


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        ResponseController(FakeOverlay(), lambda f: {}, mode="explode")

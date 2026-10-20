"""
Cross-platform behaviour without needing Windows/Linux machines:
* capture_exclusion dispatch + the Windows WDA_* logic with a fake user32;
* no module in src/ touches platform libraries at import time;
* macos_window / capture_exclusion import and degrade cleanly as win32/linux;
* the "pause" fallback (CaptureGate + controller) through the real monitoring loop.
"""
import ast
import importlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src import capture_exclusion as cx
from component1_screen_monitoring.src import macos_window
from component1_screen_monitoring.src.capture import RawFrame
from component1_screen_monitoring.src.component1 import run_monitoring_loop
from component1_screen_monitoring.src.config import MIN_HOLD_S
from component1_screen_monitoring.src.response import CaptureGate, ResponseController, mock_payload

SRC = Path(__file__).resolve().parents[1] / "src"
T0 = datetime(2026, 10, 20, 10, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# capture_exclusion
# ---------------------------------------------------------------------------

class FakeUser32:
    def __init__(self, accepts):
        self.accepts, self.calls = set(accepts), []

    def SetWindowDisplayAffinity(self, hwnd, affinity):
        self.calls.append((hwnd, affinity))
        return 1 if affinity in self.accepts else 0


def test_windows_exclude_from_capture_success():
    u = FakeUser32({cx.WDA_EXCLUDEFROMCAPTURE})
    assert cx._exclude_windows(1234, u) == (True, "windows_exclude_from_capture")
    assert u.calls == [(1234, 0x11)]


def test_windows_old_version_falls_back_to_monitor_and_returns_false():
    u = FakeUser32({cx.WDA_MONITOR})
    assert cx._exclude_windows(1234, u) == (False, "windows_monitor_only")
    assert u.calls == [(1234, 0x11), (1234, 0x1)]


def test_windows_both_refused_or_no_hwnd():
    assert cx._exclude_windows(1234, FakeUser32(set())) == (False, "failed")
    assert cx._exclude_windows(0, FakeUser32({0x11})) == (False, "failed")


def test_dispatch_windows(monkeypatch):
    monkeypatch.setattr(cx.sys, "platform", "win32")
    monkeypatch.setattr(cx, "_hwnd", lambda w: 99)
    monkeypatch.setattr(cx, "_user32", lambda: FakeUser32({0x11}))
    assert cx.exclude_from_capture(object()) is True
    assert cx.last_method == "windows_exclude_from_capture"


def test_dispatch_linux_unsupported(monkeypatch, caplog):
    monkeypatch.setattr(cx.sys, "platform", "linux")
    with caplog.at_level("INFO"):
        assert cx.exclude_from_capture(object()) is False
    assert cx.last_method == "unsupported" and "fallback" in caplog.text


def test_dispatch_macos_uses_macos_window(monkeypatch):
    monkeypatch.setattr(cx.sys, "platform", "darwin")
    seen = []
    monkeypatch.setattr(cx, "_exclude_macos", lambda title: seen.append(title) or True)
    assert cx.exclude_from_capture(SimpleNamespace(title=lambda: "Overlay")) is True
    assert seen == ["Overlay"] and cx.last_method == "macos_sharing_none"


def test_dispatch_never_raises(monkeypatch):
    monkeypatch.setattr(cx.sys, "platform", "darwin")
    def boom():
        raise RuntimeError("no window")
    assert cx.exclude_from_capture(SimpleNamespace(title=boom)) is False
    assert cx.last_method == "failed"


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------

PLATFORM_ONLY = {"objc", "AppKit", "Quartz", "Foundation", "Cocoa", "macos_window", "winreg", "msvcrt"}


def _top_level_imports(path: Path) -> set[str]:
    names = set()
    for node in ast.parse(path.read_text()).body:              # module level only
        nodes = [node]
        if isinstance(node, ast.Try):                         # try/except ImportError blocks
            nodes = node.body + [h for hd in node.handlers for h in hd.body]
        for n in nodes:
            if isinstance(n, ast.Import):
                names |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                names |= {(n.module or "").split(".")[0]} | {a.name for a in n.names}
    return names


@pytest.mark.parametrize("path", sorted(SRC.glob("*.py")), ids=lambda p: p.name)
def test_no_platform_specific_import_at_module_level(path):
    if path.name == "macos_window.py":
        pytest.skip("the macOS module itself (guarded internally — see the reload test)")
    assert not (_top_level_imports(path) & PLATFORM_ONLY)


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_macos_window_and_capture_exclusion_import_on_other_platforms(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    try:
        mw = importlib.reload(macos_window)
        assert mw.IS_MACOS is False and mw.available() is False
        assert mw.find_window("x") is None and mw.make_key(None) is False
        importlib.reload(cx)
    finally:
        monkeypatch.undo()
        importlib.reload(macos_window)                        # restore the real platform state
        importlib.reload(cx)


# ---------------------------------------------------------------------------
# "pause" fallback
# ---------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeOverlay:
    """Records calls; capture_excluded controls which path CaptureGate takes."""

    def __init__(self, capture_excluded):
        self.capture_excluded, self.calls = capture_excluded, []

    def show_blur(self, image, caption="", on_visible=None):
        self.calls.append("blur")

    def show_block(self, message="", on_visible=None):
        self.calls.append("block")

    def hide(self):
        self.calls.append("hide")

    def hide_and_wait(self):
        self.calls.append("peek_hide")
        return True

    def restore(self):
        self.calls.append("restore")


def _payload(conf=0.7, sid="A"):
    return mock_payload("violence", conf, sid, T0.isoformat())


FRAME = SimpleNamespace(image=Image.new("RGB", (32, 20)), timestamp=T0.isoformat())


def test_frame_policy():
    clock = Clock()
    c = ResponseController(FakeOverlay(False), clock=clock)
    assert c.frame_policy(capture_excluded=False) == "classify"          # nothing shown
    c.on_payload(_payload(), FRAME)
    assert c.frame_policy(capture_excluded=True) == "classify"           # macOS path unchanged
    assert c.frame_policy(capture_excluded=False, fallback="off") == "classify"
    clock.t = MIN_HOLD_S - 0.1
    assert c.frame_policy(capture_excluded=False) == "skip"
    clock.t = MIN_HOLD_S
    assert c.frame_policy(capture_excluded=False) == "peek"
    with pytest.raises(ValueError):
        c.frame_policy(False, fallback="maybe")


def test_peek_safe_clears_and_peek_flagged_extends():
    clock = Clock()
    ov = FakeOverlay(False)
    c = ResponseController(ov, clock=clock)
    c.on_payload(_payload(), FRAME)
    clock.t = 6
    assert c.on_peek_result(_payload(), FRAME) is False                   # still violent → extend
    assert c.visible and c.frame_policy(False) == "skip"                   # hold restarted at t=6
    clock.t = 11
    assert c.frame_policy(False) == "peek"
    assert c.on_peek_result(None, FRAME) is True                          # one safe peek clears
    assert not c.visible and ov.calls[-1] == "hide"


def test_peek_with_invalid_payload_keeps_overlay():
    c = ResponseController(FakeOverlay(False), clock=Clock())
    c.on_payload(_payload(), FRAME)
    bad = _payload()
    bad["confidence_score"] = 3
    assert c.on_peek_result(bad, FRAME) is False and c.visible


class ScriptedDetector:
    threshold = 0.5

    def __init__(self, probs):
        self.probs, self.calls = list(probs), 0

    def predict_regions(self, image, prev_image=None, strategy="tiles+motion"):
        self.calls += 1
        p = self.probs.pop(0)
        return SimpleNamespace(category="violence" if p >= 0.5 else "safe", confidence=p,
                               flagged=p >= 0.5, inference_ms=1.0, region="full",
                               region_box=None, n_crops=1)


def _run(capture_excluded, probs, iterations):
    clock = Clock()
    ov = FakeOverlay(capture_excluded)
    ctl = ResponseController(ov, clock=clock)
    gate = CaptureGate(ctl, ov)
    det = ScriptedDetector(probs)

    def capture():
        return RawFrame(image=Image.new("RGB", (64, 40)),
                        timestamp=(T0 + timedelta(seconds=clock.t)).isoformat())

    def sleep(s):
        clock.t += 1.0                                                  # 1 s per iteration

    stats = run_monitoring_loop(det, interval=1.0, max_iterations=iterations,
                                capture_fn=gate.wrap(capture), on_payload=lambda p: None,
                                on_result=gate.on_result, frame_gate=gate.allow, sleep_fn=sleep)
    return stats, det, ov, gate, ctl


def test_pause_fallback_skips_classification_then_peeks_and_clears():
    # frame 1 flagged (t=0) → overlay; t=1..4 skipped (< MIN_HOLD_S); t=5 peek → safe → cleared
    stats, det, ov, gate, ctl = _run(capture_excluded=False, probs=[0.7, 0.1], iterations=2)
    assert det.calls == 2                                       # only the flag + the peek were classified
    assert gate.skipped == 4 and gate.peeks == 1 and stats["skipped"] == 4
    assert ov.calls == ["blur", "peek_hide", "restore", "hide"]
    assert not ctl.visible


def test_pause_fallback_peek_still_violent_keeps_overlay():
    stats, det, ov, gate, ctl = _run(capture_excluded=False, probs=[0.7, 0.75], iterations=2)
    assert gate.peeks == 1 and ctl.visible
    assert ov.calls[:3] == ["blur", "peek_hide", "restore"] and "hide" not in ov.calls


def test_excluded_overlay_keeps_current_behaviour():
    # macOS path: every frame classified, no skips, no peeks
    stats, det, ov, gate, ctl = _run(capture_excluded=True, probs=[0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                     iterations=7)
    assert det.calls == 7 and gate.skipped == 0 and gate.peeks == 0 and stats["skipped"] == 0
    assert "peek_hide" not in ov.calls and ov.calls[-1] == "hide"   # hidden by MIN_HOLD_S + CLEAR_FRAMES

import sys
import tkinter as tk
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.overlay import ResponseOverlay, blur_image


@pytest.fixture
def overlay(monkeypatch):
    try:
        o = ResponseOverlay(exclude_from_capture=False)
    except tk.TclError as exc:
        pytest.skip(f"no display available: {exc}")
    monkeypatch.setattr(o, "take_keyboard_focus", lambda: False)   # don't steal focus in tests
    yield o
    o.stop()
    o.process_pending()


def _apply(o):
    o.process_pending()
    o.root.update()


def test_overlay_starts_hidden(overlay):
    assert not overlay.is_visible
    assert overlay.mode is None
    assert overlay.root.state() == "withdrawn"


def test_block_then_hide(overlay):
    overlay.show_block("blocked for test")
    _apply(overlay)
    assert overlay.is_visible
    assert overlay.mode == "block"
    assert overlay.root.winfo_ismapped()
    assert overlay._label.cget("text") == "blocked for test"

    overlay.hide()
    _apply(overlay)
    assert not overlay.is_visible
    assert overlay.mode is None
    assert overlay.root.state() == "withdrawn"


def test_blur_shows_image(overlay):
    overlay.show_blur(Image.new("RGB", (640, 480), "red"), caption="blurred")
    _apply(overlay)
    assert overlay.mode == "blur"
    assert overlay._label.cget("image")          # an image is attached
    assert overlay._label.cget("text") == "blurred"


def test_switching_modes_while_visible(overlay):
    overlay.show_block("x")
    overlay.show_blur(Image.new("RGB", (64, 64)))
    _apply(overlay)
    assert overlay.is_visible
    assert overlay.mode == "blur"


def test_overlay_is_topmost(overlay):
    assert overlay.root.attributes("-topmost") == 1


def test_blur_image_resizes_and_removes_detail():
    # Sharp black/white checkerboard → blur should pull pixels towards grey.
    img = Image.new("L", (256, 256))
    img.putdata([255 * ((x // 8 + y // 8) % 2) for y in range(256) for x in range(256)])
    out = blur_image(img.convert("RGB"), (320, 200))
    assert out.size == (320, 200)
    lo, hi = out.convert("L").getextrema()
    assert hi - lo < 100


def test_on_visible_called_with_timestamp(overlay):
    seen = []
    overlay.show_block("x", on_visible=seen.append)
    _apply(overlay)
    assert len(seen) == 1 and isinstance(seen[0], float)


def test_dismiss_button_hides_and_notifies(overlay):
    calls = []
    overlay.on_dismiss = lambda: calls.append("dismissed")
    overlay.show_blur(Image.new("RGB", (64, 64)), caption="c")
    _apply(overlay)
    assert overlay.dismiss_button.winfo_ismapped()
    overlay.dismiss_button.invoke()
    overlay.root.update()
    assert calls == ["dismissed"] and not overlay.is_visible
    assert overlay.root.state() == "withdrawn"


def test_escape_binding_dismisses(overlay):
    calls = []
    overlay.on_dismiss = lambda: calls.append(1)
    overlay.show_block("x")
    _apply(overlay)
    assert overlay.root.bind("<Escape>")                       # binding installed
    overlay.root.focus_force()                                # Tk delivers keys to the focus widget
    overlay.root.update()
    overlay.root.event_generate("<KeyPress-Escape>", when="now")
    overlay.root.update()
    assert calls == [1] and not overlay.is_visible


def test_dismiss_when_hidden_does_nothing(overlay):
    calls = []
    overlay.on_dismiss = lambda: calls.append(1)
    overlay._dismiss()
    assert calls == []

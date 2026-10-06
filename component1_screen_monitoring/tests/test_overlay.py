import sys
import tkinter as tk
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.overlay import ResponseOverlay, blur_image


@pytest.fixture
def overlay():
    try:
        o = ResponseOverlay()
    except tk.TclError as exc:
        pytest.skip(f"no display available: {exc}")
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

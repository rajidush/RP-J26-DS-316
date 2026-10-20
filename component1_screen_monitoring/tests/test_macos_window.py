"""macos_window.py must degrade gracefully: no crash, falsy results without a window."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src import macos_window as macwin


def test_helpers_are_safe_without_a_window():
    assert macwin.find_window("no such window title 123") is None
    assert macwin.set_sharing_none(None) is False
    assert macwin.sharing_type(None) is None
    assert macwin.make_key(None) is False
    assert macwin.is_key(None) is False
    assert macwin.allow_key_focus(None) is False
    assert macwin.window_number(None) is None
    assert macwin.capture_below(None) is None


def test_available_matches_platform():
    assert macwin.available() == (sys.platform == "darwin" and macwin._appkit is not None)


def test_helpers_never_create_nsapp_before_tk():
    """Regression: calling helpers before Tk starts must not break tk.Tk() (it used to abort)."""
    import tkinter as tk
    import pytest
    macwin.find_window("anything")
    macwin.make_key(None)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    root.destroy()

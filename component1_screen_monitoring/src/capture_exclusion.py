"""
capture_exclusion.py — Keep the Function 2 overlay out of screen captures, per platform.

    exclude_from_capture(window) -> bool

True means screen captures (mss) now show what is UNDER the overlay, so
Function 1 can keep classifying while the overlay is up. False means captures
will contain the overlay — the monitor then switches to the
config.CAPTURE_EXCLUSION_FALLBACK behaviour ("pause").

    macOS    NSWindow.sharingType = none            (macos_window.py, ctypes)
             verified on macOS 15.3.1
    Windows  SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
             Windows 10 2004+; on older Windows WDA_MONITOR is applied instead
             (the overlay is captured as black) and False is returned
    Linux    not supported → False

Imports nothing platform-specific at module import time, so it is safe to
import on any OS.
"""
from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)

WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011     # Windows 10 version 2004 and later

#: Last method used — "macos_sharing_none", "windows_exclude_from_capture",
#: "windows_monitor_only", "unsupported" or "failed". For logs / the README evidence.
last_method: str | None = None


def exclude_from_capture(window) -> bool:
    """
    Exclude a Tk toplevel (*window*, e.g. ``tk.Tk()``) from screen capture.
    Must be called on the Tk (main) thread after the window has been created.
    """
    global last_method
    platform = sys.platform
    try:
        if platform == "darwin":
            ok = _exclude_macos(window.title())
            last_method = "macos_sharing_none" if ok else "failed"
        elif platform == "win32":
            ok, last_method = _exclude_windows(_hwnd(window), _user32())
        else:
            ok, last_method = False, "unsupported"
    except Exception as e:                      # never let this break the overlay
        log.warning("capture exclusion raised %s: %s", type(e).__name__, e)
        ok, last_method = False, "failed"

    log.info("capture exclusion on %s: %s (%s)", platform,
             "ACTIVE — captures show the screen under the overlay" if ok
             else "NOT available — monitor will use the fallback", last_method)
    return ok


# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------

def _exclude_macos(title: str) -> bool:
    try:
        from . import macos_window as macwin
    except ImportError:
        import macos_window as macwin
    win = macwin.find_window(title)
    return win is not None and macwin.set_sharing_none(win)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def _user32():
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
    user32.GetParent.argtypes = [wintypes.HWND]
    user32.GetParent.restype = wintypes.HWND
    return user32


def _hwnd(window) -> int:
    """
    The Win32 handle of the Tk toplevel's OUTER window. winfo_id() is Tk's inner
    child window; wm_frame() (or GetParent of winfo_id) is the real top-level.
    """
    window.update_idletasks()
    try:
        frame = int(window.wm_frame(), 16)
        if frame:
            return frame
    except (ValueError, TypeError, AttributeError):
        pass
    return _user32().GetParent(window.winfo_id()) or window.winfo_id()


def _exclude_windows(hwnd: int, user32) -> tuple[bool, str]:
    """Try WDA_EXCLUDEFROMCAPTURE; fall back to WDA_MONITOR (returns False)."""
    if not hwnd:
        return False, "failed"
    if user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
        return True, "windows_exclude_from_capture"
    log.warning("WDA_EXCLUDEFROMCAPTURE refused (needs Windows 10 2004+); trying WDA_MONITOR")
    if user32.SetWindowDisplayAffinity(hwnd, WDA_MONITOR):
        return False, "windows_monitor_only"      # captured as black → still not usable
    return False, "failed"

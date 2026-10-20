"""
macos_window.py — Minimal macOS window helpers for the Function 2 overlay, via ctypes.

No third-party dependency: talks to the Objective-C runtime (libobjc) and
CoreGraphics directly — the same approach mss uses for screen capture.
Every function returns None / False on other platforms or on failure, so
callers can fall back gracefully.

    find_window(title)            → NSWindow pointer of our Tk window
    set_sharing_none(win)         → exclude the window from screen capture
    sharing_type(win)             → current NSWindowSharingType (0 = none)
    allow_key_focus(win)          → let a borderless window become key (receive keys)
    make_key(win)                 → activate app + make window key (keyboard focus)
    is_key(win)                   → True if the window has keyboard focus
    window_number(win)            → CGWindowID (for capture_below)
    capture_below(window_id)      → PIL image of the screen BELOW that window

All NSWindow calls must run on the main (Tk) thread.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import sys
from ctypes import c_bool, c_char_p, c_double, c_long, c_uint32, c_ulong, c_void_p

IS_MACOS = sys.platform == "darwin"
NS_WINDOW_SHARING_NONE = 0
K_CG_WINDOW_LIST_OPTION_ON_SCREEN_BELOW_WINDOW = 1 << 2
K_CG_WINDOW_IMAGE_DEFAULT = 0

_objc = _cg = _cf = _appkit = None
if IS_MACOS:
    try:
        _objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        _objc.objc_getClass.restype = c_void_p
        _objc.objc_getClass.argtypes = [c_char_p]
        _objc.sel_registerName.restype = c_void_p
        _objc.sel_registerName.argtypes = [c_char_p]
        _objc.object_getClass.restype = c_void_p
        _objc.object_getClass.argtypes = [c_void_p]
        _objc.class_replaceMethod.restype = c_void_p
        _objc.class_replaceMethod.argtypes = [c_void_p, c_void_p, c_void_p, c_char_p]
        _cg = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreGraphics"))
        _cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        _appkit = ctypes.cdll.LoadLibrary(ctypes.util.find_library("AppKit"))
    except (OSError, AttributeError):
        _objc = _cg = _cf = _appkit = None


def available() -> bool:
    return _objc is not None and _cg is not None and _appkit is not None


def _send(obj, selector: str, *args, restype=c_void_p, argtypes=()):
    """objc_msgSend with an explicit prototype (required on Apple silicon)."""
    fn = ctypes.CFUNCTYPE(restype, c_void_p, c_void_p, *argtypes)(
        ctypes.cast(_objc.objc_msgSend, c_void_p).value)
    return fn(obj, _objc.sel_registerName(selector.encode()), *args)


def _ns_app():
    """
    The EXISTING NSApp (Tk's TKApplication), or None. Never creates one:
    [NSApplication sharedApplication] would create a plain NSApplication before
    Tk starts, and Tk then aborts when it finds an app of the wrong class.
    """
    try:
        return c_void_p.in_dll(_appkit, "NSApp").value
    except (ValueError, AttributeError):
        return None


def _nsstring_to_str(ns) -> str:
    if not ns:
        return ""
    raw = _send(ns, "UTF8String", restype=c_char_p)
    return raw.decode("utf-8", "replace") if raw else ""


# ---------------------------------------------------------------------------
# NSWindow
# ---------------------------------------------------------------------------

def find_window(title: str):
    """Pointer to this process's NSWindow with *title*, or None."""
    if not available():
        return None
    app = _ns_app()
    if not app:
        return None                                   # Tk not started yet
    try:
        windows = _send(app, "windows")
        for i in range(_send(windows, "count", restype=c_ulong)):
            win = _send(windows, "objectAtIndex:", i, argtypes=(c_ulong,))
            if _nsstring_to_str(_send(win, "title")) == title:
                return win
    except Exception:
        return None
    return None


def set_sharing_none(win) -> bool:
    if not win:
        return False
    _send(win, "setSharingType:", NS_WINDOW_SHARING_NONE, restype=None, argtypes=(c_ulong,))
    return sharing_type(win) == NS_WINDOW_SHARING_NONE


def sharing_type(win) -> int | None:
    return None if not win else int(_send(win, "sharingType", restype=c_ulong))


#: Keeps the ctypes callback alive for the lifetime of the process.
_YES_IMP = None
_patched_classes: set[int] = set()


def allow_key_focus(win) -> bool:
    """
    Borderless NSWindows answer NO to canBecomeKeyWindow, so they never get
    keyboard events (Esc). Replace that method on the window's class with one
    returning YES. Only affects windows of that class in this process.
    """
    global _YES_IMP
    if not win:
        return False
    cls = _objc.object_getClass(win)
    if cls not in _patched_classes:
        if _YES_IMP is None:
            _YES_IMP = ctypes.CFUNCTYPE(c_bool, c_void_p, c_void_p)(lambda self, cmd: True)
        for sel in (b"canBecomeKeyWindow", b"canBecomeMainWindow"):
            _objc.class_replaceMethod(cls, _objc.sel_registerName(sel),
                                      ctypes.cast(_YES_IMP, c_void_p), b"c@:")
        _patched_classes.add(cls)
    return can_become_key(win)


def post_key_to_self(keycode: int) -> bool:
    """Send a key down+up to THIS process (CGEventPostToPid) — used to verify Esc."""
    if not available():
        return False
    import os
    _cg.CGEventCreateKeyboardEvent.restype = c_void_p
    _cg.CGEventCreateKeyboardEvent.argtypes = [c_void_p, ctypes.c_uint16, c_bool]
    _cg.CGEventPostToPid.argtypes = [ctypes.c_int, c_void_p]
    _cf.CFRelease.argtypes = [c_void_p]
    for down in (True, False):
        ev = _cg.CGEventCreateKeyboardEvent(None, keycode, down)
        if not ev:
            return False
        _cg.CGEventPostToPid(os.getpid(), ev)
        _cf.CFRelease(ev)
    return True


def make_key(win) -> bool:
    app = _ns_app()
    if not win or not app:
        return False
    _send(app, "activateIgnoringOtherApps:", True, restype=None, argtypes=(c_bool,))
    _send(win, "makeKeyAndOrderFront:", None, restype=None, argtypes=(c_void_p,))
    return is_key(win)


def is_key(win) -> bool:
    return bool(win) and bool(_send(win, "isKeyWindow", restype=c_bool))


def can_become_key(win) -> bool:
    return bool(win) and bool(_send(win, "canBecomeKeyWindow", restype=c_bool))


def window_number(win) -> int | None:
    return None if not win else int(_send(win, "windowNumber", restype=c_long))


# ---------------------------------------------------------------------------
# Capture what is below a window (CoreGraphics)
# ---------------------------------------------------------------------------

class _CGPoint(ctypes.Structure):
    _fields_ = [("x", c_double), ("y", c_double)]


class _CGSize(ctypes.Structure):
    _fields_ = [("width", c_double), ("height", c_double)]


class _CGRect(ctypes.Structure):
    _fields_ = [("origin", _CGPoint), ("size", _CGSize)]


def capture_below(window_id: int, display_bounds: tuple[float, float, float, float] | None = None):
    """
    PIL RGB image of the main display with *window_id* and everything ABOVE it
    left out (kCGWindowListOptionOnScreenBelowWindow). None on failure.
    """
    if not available() or window_id is None:
        return None
    from PIL import Image

    _cg.CGMainDisplayID.restype = c_uint32
    _cg.CGDisplayBounds.restype = _CGRect
    _cg.CGDisplayBounds.argtypes = [c_uint32]
    _cg.CGWindowListCreateImage.restype = c_void_p
    _cg.CGWindowListCreateImage.argtypes = [_CGRect, c_uint32, c_uint32, c_uint32]
    for name in ("CGImageGetWidth", "CGImageGetHeight", "CGImageGetBytesPerRow"):
        getattr(_cg, name).restype = ctypes.c_size_t
        getattr(_cg, name).argtypes = [c_void_p]
    _cg.CGImageGetDataProvider.restype = c_void_p
    _cg.CGImageGetDataProvider.argtypes = [c_void_p]
    _cg.CGDataProviderCopyData.restype = c_void_p
    _cg.CGDataProviderCopyData.argtypes = [c_void_p]
    _cf.CFDataGetBytePtr.restype = c_void_p
    _cf.CFDataGetBytePtr.argtypes = [c_void_p]
    _cf.CFDataGetLength.restype = c_long
    _cf.CFDataGetLength.argtypes = [c_void_p]
    _cf.CFRelease.argtypes = [c_void_p]

    rect = (_CGRect(_CGPoint(*display_bounds[:2]), _CGSize(*display_bounds[2:]))
            if display_bounds else _cg.CGDisplayBounds(_cg.CGMainDisplayID()))
    image = _cg.CGWindowListCreateImage(rect, K_CG_WINDOW_LIST_OPTION_ON_SCREEN_BELOW_WINDOW,
                                        int(window_id), K_CG_WINDOW_IMAGE_DEFAULT)
    if not image:
        return None
    data = None
    try:
        w, h = _cg.CGImageGetWidth(image), _cg.CGImageGetHeight(image)
        stride = _cg.CGImageGetBytesPerRow(image)
        data = _cg.CGDataProviderCopyData(_cg.CGImageGetDataProvider(image))
        buf = ctypes.string_at(_cf.CFDataGetBytePtr(data), _cf.CFDataGetLength(data))
        return Image.frombuffer("RGB", (w, h), buf, "raw", "BGRX", stride, 1).copy()
    finally:
        if data:
            _cf.CFRelease(data)
        _cf.CFRelease(image)

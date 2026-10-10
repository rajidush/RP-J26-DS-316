"""Device context: foreground app, screen lock and battery saver (Windows).

The foreground app gives a verdict its context (Guardian `AppContext`): the
exe name, a coarse category, and a hash of the window title. The title itself
is never kept, because it can contain a private conversation name.

Lock and battery-saver state let the agent behave like a good guest on a
child's laptop: nothing is captured while the session is locked, and checks
slow down when Windows is saving battery.
"""
from __future__ import annotations

import ctypes
import hashlib
import sys
from ctypes import wintypes
from typing import Dict

APP_CATEGORIES = {
    "game": ("minecraft", "javaw", "robloxplayerbeta", "fortnite", "valorant", "steam", "epicgameslauncher",
             "leagueclient", "overwatch", "gta5", "apex", "cod", "rocketleague", "among us"),
    "chat": ("discord", "whatsapp", "telegram", "slack", "teams", "ms-teams", "messenger", "signal", "skype"),
    "browser": ("chrome", "msedge", "firefox", "opera", "brave", "vivaldi"),
    "video": ("vlc", "spotify", "netflix", "wmplayer", "video.ui"),
    "education": ("onenote", "winword", "excel", "powerpnt", "acrord32", "zoom"),
}
_WINDOWS = sys.platform == "win32"


def app_category(exe: str) -> str:
    stem = exe.lower().removesuffix(".exe")
    for category, names in APP_CATEGORIES.items():
        if any(stem.startswith(n) for n in names):
            return category
    return "other"


def foreground_app() -> Dict[str, str]:
    """{"exe", "category", "title_hash"} for the focused window; never the title."""
    if not _WINDOWS:
        return {"exe": "unknown", "category": "other", "title_hash": ""}
    try:
        import psutil

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        exe = psutil.Process(pid.value).name() if pid.value else "unknown"
        title_hash = hashlib.sha256(buf.value.encode("utf-8")).hexdigest()[:16] if buf.value else ""
        return {"exe": exe, "category": app_category(exe), "title_hash": title_hash}
    except Exception:
        return {"exe": "unknown", "category": "other", "title_hash": ""}


def session_locked() -> bool:
    """True when the interactive desktop is not the user's (lock screen, UAC, sign-out)."""
    if not _WINDOWS:
        return False
    user32 = ctypes.windll.user32
    desk = user32.OpenInputDesktop(0, False, 0x0100)  # DESKTOP_SWITCHDESKTOP
    if not desk:
        return True
    try:
        return not user32.SwitchDesktop(desk)
    finally:
        user32.CloseDesktop(desk)


class _PowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]


def battery_saver_on() -> bool:
    if not _WINDOWS:
        return False
    status = _PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return False
    return bool(status.SystemStatusFlag & 1)

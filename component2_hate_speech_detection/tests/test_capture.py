"""Input adapters and device context (proposal stage 1, Table 4.2, NFR5)."""
import re
import sys
import threading

import numpy as np
import pytest

from component2_hate_speech_detection.src.capture import sources
from component2_hate_speech_detection.src.capture.system import (
    app_category,
    battery_saver_on,
    foreground_app,
    session_locked,
)

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows capture APIs")


@pytest.mark.parametrize("exe,category", [
    ("Minecraft.Windows.exe", "game"), ("RobloxPlayerBeta.exe", "game"), ("Discord.exe", "chat"),
    ("msedge.exe", "browser"), ("WINWORD.EXE", "education"), ("notepad.exe", "other"),
])
def test_app_categories(exe, category):
    assert app_category(exe) == category


@windows_only
def test_foreground_app_never_exposes_the_window_title():
    app = foreground_app()
    assert set(app) == {"exe", "category", "title_hash"}
    assert app["title_hash"] == "" or re.fullmatch(r"[0-9a-f]{16}", app["title_hash"])


@windows_only
def test_lock_and_battery_state_are_booleans():
    assert isinstance(session_locked(), bool) and isinstance(battery_saver_on(), bool)


@windows_only
def test_screen_comes_from_component1_when_available():
    source = sources.default_screen_source()
    assert source.name == "component1.capture_frame"
    frame = source.grab()
    assert frame is not None and frame.mode == "RGB" and frame.width >= 640


@windows_only
def test_fallback_screen_source_works_without_component1():
    frame = sources.MssScreenSource().grab()
    assert frame is not None and frame.mode == "RGB"


@windows_only
def test_loopback_audio_delivers_16k_mono_chunks():
    got, enough = [], threading.Event()

    def on_chunk(chunk):
        got.append(chunk)
        if len(got) >= 2:
            enough.set()

    source = sources.LoopbackAudioSource()
    if not source.start(on_chunk):
        pytest.skip(f"no loopback device: {source.last_error}")
    try:
        assert enough.wait(timeout=10)
    finally:
        source.stop()
    assert got[0].dtype == np.float32 and got[0].ndim == 1
    assert len(got[0]) == int(sources.SAMPLE_RATE * sources.LoopbackAudioSource.BLOCK_S)

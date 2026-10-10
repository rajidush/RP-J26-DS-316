"""Offline speech synthesis for audio test clips (Windows System.Speech).

Test and evaluation audio is synthesised on the machine, never recorded from
children (proposal Appendix B). Same approach as the MVP demo assets.
Installed voices on the reference laptop: Microsoft David (male), Microsoft
Zira (female), both en-US.
"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np

TARGET_RATE = 16000


def available() -> bool:
    try:
        return bool(voices())
    except Exception:
        return False


def voices() -> List[str]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Add-Type -AssemblyName System.Speech; "
         "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | "
         "ForEach-Object { $_.VoiceInfo.Name }"],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return [v.strip() for v in out.stdout.splitlines() if v.strip()]


def speak_to_wav(text: str, path: Path, voice: Optional[str] = None, rate: int = 0) -> Path:
    """Render `text` to a WAV file. `rate` is -10 (slow) .. 10 (fast)."""
    safe_text = text.replace("'", "''")
    safe_path = str(path).replace("'", "''")
    select = f"$s.SelectVoice('{voice}');" if voice else ""
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"{select} $s.Rate = {int(rate)};"
        f"$s.SetOutputToWaveFile('{safe_path}'); $s.Speak('{safe_text}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True,
                   timeout=120, check=True)
    return path


def load_wav(path: Path) -> np.ndarray:
    """WAV -> float32 mono at 16 kHz (linear resample if needed)."""
    with wave.open(str(path)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        audio = (audio - 128.0) / 128.0
    else:
        audio /= float(np.iinfo(dtype).max)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != TARGET_RATE:
        n = int(len(audio) * TARGET_RATE / rate)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
    return audio.astype(np.float32)

"""Speech-to-text engine (proposal FR1, NFR5, NFR8)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from component2_hate_speech_detection.evaluation import tts
from component2_hate_speech_detection.src.readers.asr import SAMPLE_RATE, AsrEngine


def test_injected_recogniser_and_timing():
    t = AsrEngine(transcribe_fn=lambda a: " hello there ").transcribe(np.zeros(SAMPLE_RATE, np.float32))
    assert t.text == "hello there" and t.audio_s == 1.0 and t.asr_ms >= 0


def test_a_failing_recogniser_returns_empty_text_instead_of_raising():
    def boom(_):
        raise RuntimeError("model missing")

    engine = AsrEngine(transcribe_fn=boom)
    assert engine.transcribe(np.zeros(1600, np.float32)).text == ""
    assert "model missing" in engine.last_error


@pytest.mark.skipif(not tts.available(), reason="Windows speech synthesis unavailable")
def test_silence_is_not_hallucinated_into_speech():
    assert AsrEngine().transcribe(np.zeros(3 * SAMPLE_RATE, np.float32)).text == ""


@pytest.mark.skipif(not tts.available(), reason="Windows speech synthesis unavailable")
def test_synthesised_speech_is_transcribed(tmp_path):
    clip = tts.load_wav(tts.speak_to_wav("want to play minecraft later tonight", tmp_path / "c.wav"))
    assert "minecraft" in AsrEngine().transcribe(clip).text.lower()


def test_loading_faster_whisper_does_not_pull_in_torch_or_transformers():
    """ctranslate2 imports both when installed (converters only); measured +168 MB in the running app."""
    pytest.importorskip("faster_whisper")
    code = (
        "import sys, importlib.util\n"
        "from component2_hate_speech_detection.src.readers.asr import _whisper_model_class\n"
        "_whisper_model_class()\n"
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules, 'converter libraries loaded'\n"
        "if importlib.util.find_spec('torch'):\n"
        "    import torch  # hidden only for that one import: still importable afterwards\n"
        "print('ok')\n"
    )
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-500:]

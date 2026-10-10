"""App layer: the API the window calls (no window is opened in tests)."""
import base64
import io
import wave

import numpy as np
import pytest
from PIL import Image, ImageDraw

from component2_hate_speech_detection.src.agent.runtime import Agent, AgentConfig
from component2_hate_speech_detection.src.app.main import Api, exclude_from_capture, wav_bytes_to_audio
from component2_hate_speech_detection.src.engine.analyzer import Analyzer
from component2_hate_speech_detection.src.readers.asr import AsrEngine
from component2_hate_speech_detection.src.readers.screen import ScreenReader
from component2_hate_speech_detection.src.readers.speech import SpeechReader


class OneLineOcr:
    name = "fake"

    def read(self, image):
        return "you should kys", [{"box": [0.1, 0.1, 0.5, 0.2], "text": "you should kys", "conf": 0.9}]


@pytest.fixture
def api(lexicon_scorer):
    agent = Agent(config=AgentConfig(age=10, audio=False, screen=False),
                  analyzer=Analyzer(scorer=lexicon_scorer),
                  screen_reader=ScreenReader(OneLineOcr()),
                  speech_reader=SpeechReader(AsrEngine(transcribe_fn=lambda a: "nobody likes you")))
    return Api(agent)


def wav_b64(audio, rate=22050):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def test_check_text_returns_the_view_with_action_and_c3_payload(api):
    view = api.check_text("you should kys")
    assert view["rung"] == "L3" and view["recommended_action"] == "blur"
    assert view["trigger_payload"]["risk_category"] == "cyberbullying"
    assert api.feed(0)[0]["top_category"] == "threat"


def test_dropped_screenshot_is_read_and_flagged(api):
    img = Image.new("RGB", (800, 200), "white")
    ImageDraw.Draw(img).text((20, 80), "you should kys", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    views = api.check_file("chat.png", "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    assert views[0]["source"] == "screen" and views[0]["rung"] == "L3"


def test_dropped_voice_clip_is_transcribed_and_scored(api):
    rng = np.random.default_rng(3)
    speech = np.concatenate([np.zeros(22050), rng.standard_normal(33075) * 0.1, np.zeros(22050)])
    views = api.check_file("voice.wav", wav_b64(speech))
    assert views and views[0]["source"] == "audio" and views[0]["top_category"] == "bullying"


def test_unsupported_files_are_refused(api):
    with pytest.raises(ValueError):
        api.check_file("notes.txt", "data:text/plain;base64,aGk=")


def test_wav_is_resampled_to_16k_mono():
    audio = wav_bytes_to_audio(base64.b64decode(wav_b64(np.zeros(22050)).split(",")[1]))
    assert audio.dtype == np.float32 and len(audio) == 16000


def test_status_reports_engine_and_window_network_separately(api):
    status = api.status()
    assert status["network"]["engine"] == 0 and "window" in status["network"]
    assert status["persona"] == "P1_PROTECT"


def test_age_setting_returns_the_persona(api):
    assert api.set_age(14) == "P3_RESPECT"


def test_capture_exclusion_fails_safely_without_a_window():
    assert exclude_from_capture(object()) is False

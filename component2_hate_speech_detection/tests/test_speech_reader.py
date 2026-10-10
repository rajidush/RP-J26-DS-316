"""Speech reader: utterance segmentation and offline transcription (proposal SO3, FR1, NFR5)."""
import numpy as np
import pytest

from component2_hate_speech_detection.evaluation import tts
from component2_hate_speech_detection.src.engine.analyzer import Analyzer
from component2_hate_speech_detection.src.readers.asr import SAMPLE_RATE, AsrEngine
from component2_hate_speech_detection.src.readers.speech import MAX_UTTERANCE_S, Segmenter, SpeechReader

RNG = np.random.default_rng(7)


def silence(s):
    return np.zeros(int(s * SAMPLE_RATE), dtype=np.float32)


def noise(s, level):
    return (RNG.standard_normal(int(s * SAMPLE_RATE)) * level).astype(np.float32)


def speechlike(s, level=0.1):
    """Noise with a ~4 Hz syllable envelope that dips near zero, as real speech does."""
    t = np.arange(int(s * SAMPLE_RATE)) / SAMPLE_RATE
    envelope = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 2
    return (noise(s, level) * envelope).astype(np.float32)


def stream(segmenter, audio, chunk=0.25):
    step = int(chunk * SAMPLE_RATE)
    out = []
    for i in range(0, len(audio), step):
        out += segmenter.feed(audio[i:i + step])
    return out


def test_one_burst_between_silences_is_one_utterance():
    utts = stream(Segmenter(), np.concatenate([silence(1), noise(1.5, 0.1), silence(1)]))
    assert len(utts) == 1 and 1.5 <= len(utts[0]) / SAMPLE_RATE <= 2.6


def test_short_click_is_dropped():
    seg = Segmenter()
    assert stream(seg, np.concatenate([silence(1), noise(0.25, 0.2), silence(1)])) == []


def test_silence_costs_nothing():
    seg = Segmenter()
    assert stream(seg, silence(10)) == [] and not seg.in_speech


def test_long_speech_is_cut_so_a_verdict_comes_quickly():
    utts = stream(Segmenter(), np.concatenate([speechlike(20), silence(1)]))
    assert len(utts) >= 2 and all(len(u) / SAMPLE_RATE <= MAX_UTTERANCE_S + 0.2 for u in utts)


def test_noise_floor_adapts_to_steady_background():
    seg = Segmenter()
    stream(seg, noise(12, 0.02))           # steady game music / fan
    assert seg.gate > 0.03
    utts = stream(seg, np.concatenate([noise(1.5, 0.2), noise(2, 0.02)]))
    assert len(utts) == 1


def test_utterance_audio_is_wiped_after_transcription():
    reader = SpeechReader(AsrEngine(transcribe_fn=lambda a: "hello"))
    utt = noise(1, 0.1)
    assert reader.transcribe(utt).transcript.text == "hello"
    assert not utt.any()


@pytest.mark.skipif(not tts.available(), reason="Windows speech synthesis unavailable")
def test_spoken_abuse_is_transcribed_and_flagged(tmp_path, lexicon_scorer):
    clip = tts.load_wav(tts.speak_to_wav("you should kill yourself nobody likes you", tmp_path / "a.wav"))
    reader = SpeechReader()
    utts = stream(reader.segmenter, np.concatenate([silence(1), clip, silence(1)]))
    assert len(utts) == 1
    text = reader.transcribe(utts[0]).transcript.text.lower()
    assert "kill yourself" in text
    v = Analyzer(scorer=lexicon_scorer).analyze_text(text, age=10, source="audio")
    assert v.alerts and v.source == "audio"

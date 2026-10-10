"""Agent runtime: loops, de-duplication, lock pause and stats (proposal NFR1-NFR8)."""
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw

from component2_hate_speech_detection.src.agent.runtime import Agent, AgentConfig
from component2_hate_speech_detection.src.engine.analyzer import Analyzer
from component2_hate_speech_detection.src.readers.asr import SAMPLE_RATE, AsrEngine
from component2_hate_speech_detection.src.readers.screen import ScreenReader
from component2_hate_speech_detection.src.readers.speech import SpeechReader


class ScriptedOcr:
    """Fake OCR: the frame's top-left pixel picks which chat line is 'on screen'."""

    name = "scripted"
    LINES = {10: "see you at practice", 20: "nobody likes you, just leave the group"}

    def read(self, image):
        text = self.LINES.get(image.getpixel((0, 0))[0], "")
        return text, ([{"box": [0.2, 0.5, 0.6, 0.55], "text": text, "conf": 0.9}] if text else [])


class FrameSource:
    name = "scripted-frames"

    def __init__(self, codes):
        self.codes = list(codes)

    def grab(self):
        code = self.codes.pop(0) if self.codes else 0
        img = Image.new("RGB", (1280, 720), (code, code, code))
        ImageDraw.Draw(img).rectangle([300, 300, 700, 340 + code], fill=(250, 250, 250))
        return img


def make_agent(lexicon_scorer, codes=(), locked=lambda: False, **config):
    return Agent(
        config=AgentConfig(audio=False, **config),
        analyzer=Analyzer(scorer=lexicon_scorer),
        screen_source=FrameSource(codes),
        screen_reader=ScreenReader(ScriptedOcr()),
        speech_reader=SpeechReader(AsrEngine(transcribe_fn=lambda a: "you should kys")),
        context=lambda: {"exe": "Discord.exe", "category": "chat", "title_hash": ""},
        locked=locked,
        battery_saver=lambda: False,
    )


def test_harmful_message_on_screen_alerts_once_not_every_check(lexicon_scorer):
    agent = make_agent(lexicon_scorer, codes=[10, 20, 20, 20], age=9)
    verdicts = [agent.screen_tick() for _ in range(4)]
    assert verdicts[0].rung == "L0"
    assert verdicts[1].alerts and verdicts[1].app["exe"] == "Discord.exe"
    assert verdicts[2] is None and verdicts[3] is None          # unchanged screen: no work at all
    assert agent.stats.counts["alerts"] == 1 and agent.stats.counts["unchanged"] == 2


def test_same_text_reappearing_within_cooldown_is_suppressed(lexicon_scorer):
    agent = make_agent(lexicon_scorer, codes=[20, 10, 20], age=9)
    [agent.screen_tick() for _ in range(3)]
    assert agent.stats.counts["alerts"] == 1 and agent.stats.counts["repeats_suppressed"] == 1


def test_nothing_is_captured_while_the_session_is_locked(lexicon_scorer):
    grabbed = []
    agent = make_agent(lexicon_scorer, locked=lambda: True)
    agent.screen_source.grab = lambda: grabbed.append(1)
    assert agent.screen_tick() is None and not grabbed
    assert agent.stats.counts["paused_locked"] == 1


def test_age_change_applies_to_the_next_check(lexicon_scorer):
    agent = make_agent(lexicon_scorer, age=15)
    assert not agent.check_text("nobody likes you").alerts
    agent.set_age(9)
    assert agent.check_text("nobody likes you").alerts


def test_spoken_abuse_is_flagged_from_an_utterance(lexicon_scorer):
    agent = make_agent(lexicon_scorer, age=10)
    v = agent.process_utterance(np.ones(SAMPLE_RATE, dtype=np.float32) * 0.1)
    assert v.alerts and v.source == "audio" and agent.stats.counts["transcribed"] == 1


def test_audio_backlog_drops_instead_of_growing_memory(lexicon_scorer):
    agent = make_agent(lexicon_scorer)
    rng = np.random.default_rng(1)
    burst = np.concatenate([(rng.standard_normal(SAMPLE_RATE) * 0.1).astype(np.float32),
                            np.zeros(SAMPLE_RATE, np.float32)])
    for _ in range(8):  # nobody is draining the queue
        for i in range(0, len(burst), 4000):
            agent.feed_audio(burst[i:i + 4000])
    assert agent._utterances.qsize() == 4 and agent.stats.counts["audio_dropped"] == 4


def test_snapshot_reports_resources_sources_and_timings(lexicon_scorer):
    agent = make_agent(lexicon_scorer, codes=[10, 20])
    agent.screen_tick(), agent.screen_tick()
    snap = agent.snapshot()
    assert snap["rss_mb"] > 0 and snap["sources"]["screen"] == "scripted-frames"
    assert snap["timings"]["screen_check_ms"]["n"] == 2 and snap["counts"]["screen_checks"] == 2


def test_background_loop_runs_and_stops_cleanly(lexicon_scorer):
    agent = make_agent(lexicon_scorer, codes=[10, 20, 20, 20, 20, 20], screen_interval_s=0.05, age=9)
    agent.start()
    try:
        deadline = time.time() + 10
        while agent.stats.counts["alerts"] == 0 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        agent.stop()
    assert agent.stats.counts["alerts"] == 1 and agent.state == "stopped"

"""
The demo page's confidence slider (PP1 close-out, 21 Oct).

The slider is the trigger's confidence_score, and each detector means something
different by it. The page says how Component 3 reads the number and warns when the
chosen detector would never send it. The page may only describe what the FSM does:
the level it shows must be the level the FSM records.
Run with: pytest component3_socratic_educator/tests -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from component3_socratic_educator.src.fsm_controller import FSMController  # noqa: E402
from component3_socratic_educator.src.schemas import RiskCategory, TriggerPayload  # noqa: E402

PAGE = Path(__file__).resolve().parents[1] / "src" / "demo_dashboard.py"
SOURCE = {"component1": "component1_screen_monitoring", "component2": "component2_hate_speech_detection"}


def _page(src="component1", category=None, confidence=None):
    app = AppTest.from_file(str(PAGE), default_timeout=30).run()
    if src != "component1":
        app.radio(key="c_src").set_value(src).run()
    if category:
        app.selectbox(key=f"c_cat_{src}").set_value(category).run()
    if confidence is not None:
        app.slider(key=f"c_conf_{src}").set_value(confidence).run()
    assert not app.exception
    return app


def _caption(app):
    return next(c.value for c in app.caption if c.value.startswith("Component 3 reads this as"))


def _fsm_level(src, confidence):
    category = RiskCategory("violence" if src == "component1" else "cyberbullying")
    trigger = TriggerPayload.new(SOURCE[src], "image" if src == "component1" else "text_message", category, confidence)
    record, _ = FSMController().run(trigger, lambda question: "I was just watching videos")
    return record.risk_level.value


@pytest.mark.parametrize("src", ["component1", "component2"])
@pytest.mark.parametrize("confidence", [0.3, 0.49, 0.5, 0.84, 0.85, 0.99])
def test_the_level_the_page_shows_is_the_level_the_fsm_records(src, confidence):
    assert f"**{_fsm_level(src, confidence)}**" in _caption(_page(src, confidence=confidence))


def test_a_high_score_says_it_escalates_whatever_the_child_says():
    caption = _caption(_page(confidence=0.9))
    assert "**high**" in caption and "escalates to a parent whatever the child says" in caption


def test_a_moderate_score_says_only_a_distressed_child_is_escalated():
    assert "Only a distressed child is escalated" in _caption(_page(confidence=0.6))


def test_each_detector_says_what_its_score_means():
    assert "probability that the frame is violent" in _caption(_page("component1", confidence=0.7))
    assert "fused probability of its top category" in _caption(_page("component2", confidence=0.7))


def test_component_2_explains_its_age_thresholds():
    caption = _caption(_page("component2", confidence=0.7))
    assert "0.55 (ages 8-10), 0.65 (11-13), 0.75 (14-15)" in caption


def _start(app):
    return next(b for b in app.button if b.label == "Start session")


@pytest.mark.parametrize("src, category, confidence", [
    ("component1", "violence", 0.0),               # the slider minimum
    ("component1", "violence", 0.4),               # Component 1 flags from 0.50
    ("component2", "cyberbullying", 0.0),
    ("component2", "cyberbullying", 0.52),         # lowest C2 threshold for bullying is 0.55
    ("component2", "self_harm_language", 0.45),    # self-harm and grooming use 0.50
])
def test_a_score_the_detector_would_never_send_starts_no_dialogue(src, category, confidence):
    app = _page(src, category, confidence)
    assert any("would not send this" in w.value for w in app.warning)
    assert _start(app).disabled
    assert any("Component 3 is not running" in i.value for i in app.info)
    assert not app.chat_message


@pytest.mark.parametrize("src, category, confidence", [
    ("component1", "violence", 0.5),
    ("component2", "cyberbullying", 0.55),
    ("component2", "self_harm_language", 0.5),
])
def test_a_score_at_the_detectors_lowest_trigger_starts_a_dialogue(src, category, confidence):
    app = _page(src, category, confidence)
    assert not app.warning
    assert not _start(app).disabled
    _start(app).click().run()
    assert app.chat_message and not app.exception

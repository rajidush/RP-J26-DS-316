"""Self-harm specialist: counted only for the writer's own statement (proposal FR2, invariant I-3).

The suicidality model is faked: these tests pin the gate and the scorer wiring,
not the model. Real-model recall is measured by evaluation/self_harm.py.
"""
import pytest

from component2_hate_speech_detection.src.engine.analyzer import Analyzer
from component2_hate_speech_detection.src.engine.text_scorer import TextScorer


class FakeHead:
    """A model head with a fixed reading."""

    def __init__(self, name, score):
        self.name, self.loaded, self._score = name, True, score

    def read(self, text):
        return self._score, None, {}


def scorer(self_harm=0.95, **kwargs):
    quiet = [FakeHead("fake:a", 0.02), FakeHead("fake:b", 0.02)]
    return TextScorer(heads=quiet, self_harm_head=FakeHead("fake:suicidality", self_harm), **kwargs)


def test_the_writers_own_risk_counts_as_self_harm():
    d = scorer().score("i have pills saved up for tonight")
    assert d.category == "self_harm" and d.score == 0.95 and d.self_harm_counted and not d.self_harm_gate


def test_a_first_person_self_harm_reading_is_urgent_for_every_age():
    analyzer = Analyzer(scorer=scorer())
    for age in (9, 12, 15):
        v = analyzer.analyze_text("i've been hurting myself again", age)
        assert v.top_category == "self_harm" and v.urgent and v.rung == "L4"
        assert any(e.producer == "fake:suicidality" for e in v.evidence)


@pytest.mark.parametrize("text, gate", [
    ("everyone would be happier if you were gone", "not_first_person"),
    ("i think you should just disappear", "aimed_at_someone_else"),
    ("someone in the group chat told me to kill myself and i'm scared", "reported_speech"),
    ("i'd kill for some sleep", "idiom"),
    ("i'm so tired i could sleep forever", "figurative_context"),
    ("this level is impossible im gonna lose it", "gaming_context"),
])
def test_readings_not_about_the_writer_do_not_count(text, gate):
    d = scorer(self_harm=0.99).score(text)
    assert d.self_harm_gate == gate and not d.self_harm_counted and d.category != "self_harm"


def test_the_reporting_headline_case_stays_below_every_threshold():
    """SO2: a child reporting abuse must not be flagged, even when the model reads suicide language."""
    v = Analyzer(scorer=scorer(self_harm=0.99)).analyze_text(
        "someone in the group chat told me to kill myself and i'm scared", 9)
    assert v.rung == "L0" and "not counted (reported speech)" in v.explanation


def test_a_lone_reading_below_trust_is_ignored(decision_params):
    trust = decision_params["self_harm_model"]["trust"]
    d = scorer(self_harm=trust - 0.05).score("i keep thinking about not waking up")
    assert d.category != "self_harm" and not d.self_harm_counted


def test_no_model_reading_changes_nothing():
    with_none = scorer(self_harm=None).score("i keep thinking about not waking up")
    without = TextScorer(heads=[FakeHead("fake:a", 0.02), FakeHead("fake:b", 0.02)]).score(
        "i keep thinking about not waking up")
    assert (with_none.score, with_none.category) == (without.score, without.category)
    assert with_none.self_harm_score is None


def test_explicit_intent_still_needs_no_model():
    d = scorer(self_harm=0.0).score("i want to kill myself")
    assert d.category == "self_harm" and d.score >= 0.88 and d.self_harm_score is None


def test_custom_heads_load_no_self_harm_model_and_the_ablation_switch_removes_it():
    assert TextScorer(heads=[FakeHead("fake:a", 0.1)]).self_harm_head is None
    assert TextScorer(use_self_harm=False).self_harm_head is None
    assert TextScorer(use_heads=False).self_harm_head is None

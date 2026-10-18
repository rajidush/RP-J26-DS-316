"""Grooming: one-message tactics alert as grooming; ordinary secrets between friends do not (proposal FR2).

The lexicon rules and the specialist's gates are pinned here, with the
zero-shot model faked. Recall on realistic messages through the full engine
is measured by evaluation/grooming.py.
"""
import pytest

from component2_hate_speech_detection.src.engine.analyzer import Analyzer
from component2_hate_speech_detection.src.engine.lexicon import Lexicon
from component2_hate_speech_detection.src.engine.text_scorer import TextScorer


class FakeTacticReader:
    """Stands in for the zero-shot model: a fixed tactic reading, and a count of calls."""

    def __init__(self, score, tactic="secrecy"):
        self.name, self.loaded, self._reading, self.calls = "fake:zeroshot", True, (score, tactic, {tactic: score}), 0

    def read(self, text):
        self.calls += 1
        return self._reading


class Quiet:
    def __init__(self, name):
        self.name, self.loaded = name, True

    def read(self, text):
        return 0.02, None, {}


def scorer(reading=0.95, tactic="secrecy"):
    reader = FakeTacticReader(reading, tactic)
    return TextScorer(heads=[Quiet("fake:a"), Quiet("fake:b")], grooming_head=reader), reader


@pytest.mark.parametrize("text, family", [
    ("don't tell your parents we talk", "grooming:hide_contact"),
    ("tell your mum you're at a friend's house", "grooming:hide_contact"),
    ("your parents don't understand you like i do", "grooming:isolation"),
    ("you're so mature for your age", "grooming:isolation"),
    ("are your parents asleep yet? turn your camera on", "grooming:risk_check"),
    ("if you send me a photo i'll give you robux", "grooming:inducement"),
    ("let's meet at the park, come alone", "grooming:secret_meeting"),
    ("send one just for me", "grooming:exclusive_request"),
])
def test_each_tactic_alerts_as_grooming(text, family, decision_params):
    hit = Lexicon().score(text)
    assert hit.category == "grooming" and family in hit.hits
    assert hit.score >= decision_params["critical_theta"]


@pytest.mark.parametrize("text", [
    "send me pics and don't tell your parents",  # tactic
    "send me pics, it's our little secret",      # cue
])
def test_a_sexual_request_with_a_grooming_tactic_or_cue_is_grooming(text):
    hit = Lexicon().score(text)
    assert hit.category == "grooming" and "sexual:coercion" in hit.hits


def test_a_sexual_request_alone_stays_sexual_harassment():
    assert Lexicon().score("send me nudes").category == "sexual_harassment"


@pytest.mark.parametrize("text", [
    "send me pics of the homework sheet please",
    "don't tell mum about her birthday present",
    "are your parents home? i want to drop off your jumper",
    "let's meet at the park after school with everyone",
    "can you gift me robux for my birthday",
])
def test_the_same_words_in_ordinary_chat_do_not_alert(text):
    assert Lexicon().score(text).category == "none"


def test_checking_who_reads_the_chat_is_only_a_cue(decision_params):
    hit = Lexicon().score("does your mum ever check your phone?")
    assert hit.category == "grooming" and hit.score < decision_params["critical_theta"]


def test_grooming_is_urgent_for_every_age_and_blocked_for_the_youngest(make_scorer):
    """Relabelled from sexual harassment, so ages 8-10 keep its block (L4) rather than drop to a blur."""
    analyzer = Analyzer(scorer=make_scorer((0.02, None), (0.02, None)))
    for age, rung in ((9, "L4"), (12, "L3"), (15, "L3")):
        v = analyzer.analyze_text("send me pics and don't tell your parents", age)
        assert v.top_category == "grooming" and v.urgent and v.rung == rung
        assert v.to_trigger_payload()["risk_category"] == "grooming_language"


# -- the zero-shot specialist ------------------------------------------------

def test_a_tactic_the_rules_miss_is_read_by_the_model():
    s, reader = scorer(0.95, "secrecy")
    d = s.score("make sure your parents never find out about our chats")
    assert d.category == "grooming" and d.grooming_counted and d.grooming_tactic == "secrecy"
    assert reader.calls == 1


def test_the_model_is_not_asked_about_messages_off_topic():
    s, reader = scorer(0.99)
    d = s.score("gg that last round was insane")
    assert reader.calls == 0 and d.grooming_score is None and d.category == "none"


def test_a_surprise_being_kept_quiet_does_not_count():
    s, _ = scorer(0.99, "secrecy")
    d = s.score("keep the surprise party a secret from dad ok")
    assert d.grooming_gate == "surprise_context" and not d.grooming_counted and d.category != "grooming"


def test_a_reading_below_trust_is_ignored(decision_params):
    s, _ = scorer(decision_params["grooming_model"]["trust"] - 0.05)
    d = s.score("make sure your parents never find out about our chats")
    assert not d.grooming_counted and d.category != "grooming"


def test_no_model_reading_changes_nothing():
    s, _ = scorer(None)
    d = s.score("make sure your parents never find out about our chats")
    without = TextScorer(heads=[Quiet("fake:a"), Quiet("fake:b")]).score(
        "make sure your parents never find out about our chats")
    assert (d.score, d.category) == (without.score, without.category) and d.grooming_score is None


def test_the_verdict_names_the_tactic_and_carries_no_text():
    s, _ = scorer(0.95, "meeting")
    v = Analyzer(scorer=s).analyze_text("i'll wait for you outside school, don't tell anyone", 12)
    assert v.top_category == "grooming" and v.urgent and "reads as meeting" in v.explanation
    families = v.to_trigger_payload()["context_metadata"]["matched_families"]
    assert "grooming_model:meeting" in families and not any("school" in f for f in families)


def test_custom_heads_load_no_grooming_model_and_the_ablation_switch_removes_it():
    assert TextScorer(heads=[Quiet("fake:a")]).grooming_head is None
    assert TextScorer(use_grooming=False).grooming_head is None
    assert TextScorer(use_heads=False).grooming_head is None

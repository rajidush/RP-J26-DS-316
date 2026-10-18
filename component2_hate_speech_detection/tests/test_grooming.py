"""Grooming: one-message tactics alert as grooming; ordinary secrets between friends do not (proposal FR2).

The lexicon rules are pinned here. Recall on realistic messages through the
full engine is measured by evaluation/grooming.py.
"""
import pytest

from component2_hate_speech_detection.src.engine.analyzer import Analyzer
from component2_hate_speech_detection.src.engine.lexicon import Lexicon


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

"""Lexicon (proposal SO1 keyword layer). Cases carried from the MVP, plus the new families."""
import pytest

from component2_hate_speech_detection.src.engine.lexicon import Lexicon


@pytest.mark.parametrize("text,category", [
    ("you should kys", "threat"),
    ("k y s", "threat"),
    ("kiiill yourself", "threat"),
    ("shut up faggot", "hate_identity"),
    ("they breed like vermin", "hate_identity"),
    ("send me nudes", "sexual_harassment"),
    ("go back toyour country", "bullying"),  # OCR swallowed a space
])
def test_lexicon_flags_explicit_abuse(text, category):
    hit = Lexicon().score(text)
    assert hit.score >= 0.55 and hit.category == category and hit.hits


@pytest.mark.parametrize("text", [
    "gg ez noob", "want to play minecraft after school?", "i hate all vegetables honestly",
    "this homework is killing me",
])
def test_lexicon_clears_ordinary_chat(text):
    assert Lexicon().score(text).score < 0.35


def test_threat_patterns_are_suppressed_in_gaming_context():
    assert Lexicon().score("im gonna destroy you in this match").score < 0.35
    assert Lexicon().score("im gonna destroy you after school").category == "threat"


def test_gaming_context_is_detected():
    lex = Lexicon()
    assert lex.in_gaming_context("rip my whole squad wiped")
    assert not lex.in_gaming_context("im going to hurt you after school")


def test_self_harm_hyperbole_is_suppressed_but_intent_is_not():
    lex = Lexicon()
    assert lex.score("ugh this homework makes me want to die lol").category == "none"
    assert lex.score("i want to die").category == "self_harm"
    assert lex.score("i'm going to kill myself tonight").score >= 0.88


def test_grooming_secrecy_is_a_cue_that_never_alerts_alone(decision_params):
    hit = Lexicon().score("this is our little secret ok")
    assert hit.category == "grooming"
    assert decision_params["stage1_theta"] <= hit.score < decision_params["critical_theta"]


def test_every_hit_names_its_rule():
    hit = Lexicon().score("nobody likes you go back to your country")
    assert "go back to your country" in hit.hits and "bullying:exclusion" in hit.hits

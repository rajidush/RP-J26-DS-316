"""No words, no model reading: OCR garbage gets the keyword layer alone (decision.json `model_needs_words`).

The Dynabench head scored "OLMPCCHW" (an OCR misread at a meme's edge) 0.998
and alone raised an L3 hate alert. Text with no word-shaped token is now read
by the keyword layer only. Words are judged by shape, not a dictionary: a
first version used an English word list and silenced slur-only messages.
The heads are faked here.
"""
import pytest

from component2_hate_speech_detection.src.engine.analyzer import Analyzer
from component2_hate_speech_detection.src.engine.lexicon import Lexicon
from component2_hate_speech_detection.src.engine.text_scorer import TextScorer


class CountingHead:
    """A model head that always reads 0.99 hate and counts how often it is asked."""

    def __init__(self, name):
        self.name, self.loaded, self.calls = name, True, 0

    def read(self, text):
        self.calls += 1
        return 0.99, "hate_identity", {}


def scorer(**kwargs):
    heads = [CountingHead("fake:a"), CountingHead("fake:b")]
    return TextScorer(heads=heads, **kwargs), heads


@pytest.mark.parametrize("garbage", ["OLMPCCHW", "A-A∨±S-", "RKonsern", "1-800 |||"])
def test_ocr_garbage_is_not_given_to_the_models(garbage):
    s, heads = scorer()
    d = s.score(garbage)
    assert d.no_real_words and d.model_score is None and d.category == "none"
    assert all(h.calls == 0 for h in heads)


def test_text_with_a_real_word_is_still_read_by_the_models():
    s, heads = scorer()
    d = s.score("those people again")
    assert not d.no_real_words and d.model_score == 0.99 and all(h.calls == 1 for h in heads)


@pytest.mark.parametrize("text, category", [("k y s", "threat"), ("g o  d i e", "threat")])
def test_the_keyword_layer_still_reads_text_with_no_real_words(text, category):
    s, _ = scorer()
    d = s.score(text)
    assert d.no_real_words and d.category == category and d.score >= 0.55


@pytest.mark.parametrize("text", ["@Zwaaad fag", "youre a fag", "coons", "OKC nigguh", "Bitchez be stoopid", "LOL"])
def test_slurs_and_slang_are_words_even_though_no_dictionary_has_them(text):
    s, heads = scorer()
    d = s.score(text)
    assert not d.no_real_words and all(h.calls == 1 for h in heads)


def test_word_shape_rules():
    lex = Lexicon()
    assert lex.has_word("h4te you") and lex.has_word("strengths") and lex.has_word("FAG")
    assert not lex.has_word("OLMPCCHW")   # seven consonants in a row
    assert not lex.has_word("RKonsern")   # capitals no word has


def test_the_ablation_switch_gives_garbage_back_to_the_models():
    s, heads = scorer(use_word_check=False)
    assert s.score("OLMPCCHW").model_score == 0.99 and all(h.calls == 1 for h in heads)


def test_the_explanation_says_why_no_model_was_asked():
    s, _ = scorer()
    v = Analyzer(scorer=s).analyze_text("k y s", 12)
    assert "No words to read" in v.explanation

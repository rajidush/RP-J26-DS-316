"""No real words, no model reading: OCR garbage gets the keyword layer alone (packs/policy/words.json).

The Dynabench head scored "OLMPCCHW" (an OCR misread at a meme's edge) 0.998
and alone raised an L3 hate alert. Text with no English word of 3+ letters is
now read by the keyword layer only. The heads are faked here.
"""
import json
import shutil

import pytest

from component2_hate_speech_detection.src.engine.analyzer import Analyzer
from component2_hate_speech_detection.src.engine.lexicon import Lexicon
from component2_hate_speech_detection.src.engine.policy import PACK_DIR, PolicyIntegrityError, load_policy
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


@pytest.mark.parametrize("garbage", ["OLMPCCHW", "A-A∨±S-", "RKonsern", "FEME EOMEsME"])
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


def test_leetspeak_is_normalised_before_looking_for_words():
    assert Lexicon().has_word("h4te you") and not Lexicon().has_word("OLMPCCHW")


def test_the_ablation_switch_gives_garbage_back_to_the_models():
    s, heads = scorer(use_word_check=False)
    assert s.score("OLMPCCHW").model_score == 0.99 and all(h.calls == 1 for h in heads)


def test_the_explanation_says_why_no_model_was_asked():
    s, _ = scorer()
    v = Analyzer(scorer=s).analyze_text("k y s", 12)
    assert "No real words to read" in v.explanation


def test_the_word_list_is_part_of_the_verified_pack(tmp_path):
    pack = tmp_path / "policy"
    shutil.copytree(PACK_DIR, pack)
    words = json.loads((pack / "words.json").read_text(encoding="utf-8"))
    words["words"].append("olmpcchw")  # someone quietly lets garbage through
    (pack / "words.json").write_text(json.dumps(words), encoding="utf-8")
    with pytest.raises(PolicyIntegrityError):
        load_policy(pack)


def test_an_older_pack_without_a_word_list_still_loads_with_the_rule_off(tmp_path):
    pack = tmp_path / "policy"
    shutil.copytree(PACK_DIR, pack)
    (pack / "words.json").unlink()
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    del manifest["files"]["words.json"]
    (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    policy = load_policy(pack)
    assert policy.words == frozenset() and Lexicon(policy).has_word("OLMPCCHW")

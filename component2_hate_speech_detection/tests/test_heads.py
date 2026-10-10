"""Head label handling and windowing: by label name, never by index (proposal SO1)."""
from component2_hate_speech_detection.src.engine.heads import (
    WINDOW_CHARS,
    category_from_labels,
    hate_score_from_labels,
    windows,
)


def test_softmax_head_score_is_the_hate_probability():
    rows = [{"label": "nothate", "score": 0.2}, {"label": "hate", "score": 0.8}]
    assert hate_score_from_labels(rows) == 0.8


def test_multi_label_head_score_is_the_strongest_harmful_label():
    rows = [{"label": "toxic", "score": 0.6}, {"label": "threat", "score": 0.9}, {"label": "obscene", "score": 0.1}]
    assert hate_score_from_labels(rows) == 0.9


def test_unknown_label_vocabulary_abstains_instead_of_guessing():
    assert hate_score_from_labels([{"label": "POSITIVE", "score": 0.99}]) is None


def test_category_prefers_a_specific_label_over_bare_toxic():
    rows = [{"label": "toxic", "score": 0.95}, {"label": "insult", "score": 0.7}]
    assert category_from_labels(rows) == "bullying"
    rows = [{"label": "toxic", "score": 0.95}, {"label": "identity_hate", "score": 0.6}]
    assert category_from_labels(rows) == "hate_identity"


def test_weak_labels_claim_no_category():
    assert category_from_labels([{"label": "threat", "score": 0.3}]) is None


def test_long_screen_text_is_windowed_with_overlap():
    text = "x" * (WINDOW_CHARS * 3)
    chunks = windows(text)
    assert len(chunks) > 3 and all(len(c) <= WINDOW_CHARS for c in chunks)
    assert windows("short") == ["short"]

"""Shared fixtures for C2 tests. Model heads are faked so tests stay fast and offline."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class FakeHead:
    """Stands in for a pretrained head: fixed score, category and labels."""

    def __init__(self, name, score, category=None, labels=None):
        self.name, self._reading, self.loaded = name, (score, category, labels or {}), True

    def read(self, text):
        return self._reading


@pytest.fixture
def decision_params():
    from component2_hate_speech_detection.src.engine.policy import default_policy

    return default_policy().decision


@pytest.fixture
def make_scorer():
    """make_scorer((score, category), (score, category), **flags) -> TextScorer with fake heads."""
    from component2_hate_speech_detection.src.engine.text_scorer import TextScorer

    def _make(*readings, **kwargs):
        return TextScorer(heads=[FakeHead(f"fake:{i}", *r) for i, r in enumerate(readings)], **kwargs)

    return _make


@pytest.fixture
def lexicon_scorer():
    from component2_hate_speech_detection.src.engine.text_scorer import TextScorer

    return TextScorer(use_heads=False)

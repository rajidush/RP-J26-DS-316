"""
Keeps the unit tests deterministic since 5 Oct, when INTERCEPT started calling
the live model (src/model_client.py). By default every test runs with the model
call blocked, so GrammarConstrainedGenerator.generate() takes its template
fallback whether or not LM Studio happens to be running.

Tests that really want the live model are marked `@pytest.mark.live_model`;
they skip when LM Studio is not reachable.
"""
import sys
import urllib.request
from pathlib import Path

import pytest

# Repo root on sys.path so `component3_socratic_educator.*` imports work no
# matter which folder pytest is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component3_socratic_educator.src.model_client import LM_STUDIO_URL


def pytest_configure(config):
    config.addinivalue_line("markers", "live_model: needs LM Studio running on localhost:1234")


def _lm_studio_up() -> bool:
    try:
        urllib.request.urlopen(LM_STUDIO_URL.split("/v1/")[0] + "/v1/models", timeout=2)
        return True
    except OSError:
        return False


@pytest.fixture(autouse=True)
def _block_live_model(request, monkeypatch):
    if request.node.get_closest_marker("live_model"):
        if not _lm_studio_up():
            pytest.skip("LM Studio not reachable on localhost:1234")
        return

    def _offline(*_a, **_k):
        raise ConnectionError("live model blocked in unit tests (mark the test live_model to allow it)")

    # The decoder is imported under two names: component3_socratic_educator.src.*
    # by most tests, src.* by the dataset test. Patch whichever are loaded.
    for name, module in list(sys.modules.items()):
        if name.endswith("src.grammar_decoder"):
            monkeypatch.setattr(module, "call_local_model", _offline)

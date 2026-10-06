"""
Function 2 tests -- this is your adversarial-batch harness, scaled down
to run instantly for PP1. Run with:
pytest component3_socratic_educator/tests -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component3_socratic_educator.src.grammar_decoder import GrammarConstrainedGenerator, run_adversarial_batch
from component3_socratic_educator.src.schemas import RiskCategory


def test_all_generated_intercept_text_matches_grammar():
    gen = GrammarConstrainedGenerator()
    for rc in RiskCategory:
        text = gen.generate("INTERCEPT", rc)
        assert gen.validate("INTERCEPT", text), f"Violation for {rc}: {text!r}"


def test_forbidden_substrings_never_appear():
    gen = GrammarConstrainedGenerator()
    for state in ("INTERCEPT", "INQUIRE"):
        for rc in RiskCategory:
            text = gen.generate(state, rc)
            assert "kill" not in text.lower()
            assert "hate" not in text.lower()


def test_adversarial_batch_reports_zero_violations_on_current_pool():
    gen = GrammarConstrainedGenerator()
    categories = list(RiskCategory) * 5  # stand-in for a larger adversarial set
    report = run_adversarial_batch(gen, "INTERCEPT", categories)
    assert report["violation_rate"] == 0.0
    assert report["total_prompts"] == len(categories)


# --- 5 Oct: real SLM wired into INTERCEPT (unconstrained) --------------------
# Plumbing check only. Grammar compliance of the live output is NOT asserted
# yet -- constraining it is the 6 Oct task. Skips when LM Studio is down.

@pytest.mark.live_model
def test_live_model_reaches_intercept_and_session_completes():
    from component3_socratic_educator.src.fsm_controller import FSMController
    from component3_socratic_educator.src.schemas import TriggerPayload

    gen = GrammarConstrainedGenerator()
    text = gen.generate("INTERCEPT", RiskCategory.VIOLENCE)
    assert text and text != gen._candidate("INTERCEPT", RiskCategory.VIOLENCE), \
        "expected live model output, got the template fallback"

    trigger = TriggerPayload.new("component1_screen_monitoring", "image",
                                 RiskCategory.VIOLENCE, 0.7)
    result, transcript = FSMController(generator=gen).run(trigger, lambda _p: "I was just scrolling my feed")
    result.validate()
    assert transcript.turns[0].system_text.strip()

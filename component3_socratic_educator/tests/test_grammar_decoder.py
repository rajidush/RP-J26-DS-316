"""
Function 2 tests -- this is your adversarial-batch harness, scaled down
to run instantly for PP1. Run with:
pytest component3_socratic_educator/tests -v
"""
import sys
from pathlib import Path

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

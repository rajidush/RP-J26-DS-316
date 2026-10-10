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


# --- 6 Oct: constrained decoding for INTERCEPT (ADR 0004) --------------------
# The model is faked: each test hands the generator a `model_call` that records
# what it was sent and returns a canned reply, so no LM Studio is needed.
import json

from component3_socratic_educator.src import grammar_decoder as gd

OPENER = gd.INTERCEPT_OPENERS[0]


def _fake_model(reply):
    calls = []

    def call(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return (reply if isinstance(reply, str) else json.dumps(reply)), 0.5
    call.calls = calls
    return call


def _line(reply, rc=RiskCategory.VIOLENCE):
    model = _fake_model(reply)
    return GrammarConstrainedGenerator(model_call=model).generate_with_source("INTERCEPT", rc), model


def test_intercept_asks_the_model_with_the_json_schema_and_no_category():
    _, model = _line({"opener": OPENER, "question": "What were you doing just now?"})
    [call] = model.calls
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["schema"] == gd.INTERCEPT_SCHEMA
    assert "violence" not in call["prompt"].lower()          # Q7: the category is never sent


def test_a_valid_constrained_reply_becomes_the_line():
    line, _ = _line({"opener": OPENER, "question": "What were you doing just now?"})
    assert line.text == f"{OPENER} What were you doing just now?"
    assert (line.source, line.reason) == ("model", None)
    assert GrammarConstrainedGenerator().validate("INTERCEPT", line.text)


@pytest.mark.parametrize("reply, reason", [
    ("not json at all", "invalid_json"),
    ({"opener": "Hey there.", "question": "What were you doing?"}, "grammar"),
    ({"opener": OPENER, "question": "What were you doing"}, "grammar"),                  # no ?
    ({"opener": OPENER, "question": "What’s going on?"}, "grammar"),                     # curly apostrophe
    ({"opener": OPENER, "question": "Do you hate what you saw?"}, "forbidden_word"),
    ({"opener": OPENER, "question": "Did that alert worry you?"}, "alert_language"),
    ({"opener": OPENER, "question": "Was the violence upsetting to see?"}, "category_named"),
])
def test_a_bad_reply_falls_back_to_the_template_line(reply, reason):
    line, _ = _line(reply)
    assert (line.source, line.reason) == ("fallback", reason)
    assert line.text == GrammarConstrainedGenerator()._candidate("INTERCEPT", RiskCategory.VIOLENCE)


def test_an_unreachable_model_falls_back_to_the_template_line():
    # conftest blocks the live model, so the default model call raises
    line = GrammarConstrainedGenerator().generate_with_source("INTERCEPT", RiskCategory.VIOLENCE)
    assert (line.source, line.reason) == ("fallback", "model_unreachable")


def test_category_names_are_matched_as_whole_words_only():
    line, _ = _line({"opener": OPENER, "question": "How are you feeling about yourself right now?"},
                    rc=RiskCategory.SELF_HARM_IMAGERY)
    assert line.source == "model"                             # "yourself" is not "self"


def test_inquire_never_calls_the_model():
    model = _fake_model({"opener": OPENER, "question": "Unused?"})
    line = GrammarConstrainedGenerator(model_call=model).generate_with_source("INQUIRE", RiskCategory.VIOLENCE)
    assert model.calls == [] and line.source == "template"


def test_the_grammar_regex_accepts_every_template_line():
    gen = GrammarConstrainedGenerator()
    for line in gen._CANDIDATES["INTERCEPT"]:
        assert gen.validate("INTERCEPT", line), line


@pytest.mark.live_model
def test_live_constrained_intercept_line_meets_the_grammar():
    gen = GrammarConstrainedGenerator()
    for rc in (RiskCategory.VIOLENCE, RiskCategory.GROOMING_LANGUAGE):
        line = gen.generate_with_source("INTERCEPT", rc)
        assert gen.validate("INTERCEPT", line.text), line
        assert line.source == "model" or line.reason != "grammar", line   # a grammar fallback means the schema leaked

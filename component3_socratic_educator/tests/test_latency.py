"""
src/latency: how long the child waits for each line. A fake clock stands in
for time, so the numbers are exact and no server is needed.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component3_socratic_educator.src import latency
from component3_socratic_educator.src.fsm_controller import FSMController
from component3_socratic_educator.src.grammar_decoder import GrammarConstrainedGenerator
from component3_socratic_educator.src.schemas import RiskCategory, TriggerPayload

MODEL_LINE = json.dumps({"opener": "I noticed something on your screen just now.",
                         "question": "What were you doing just before this?"})


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _trigger():
    return TriggerPayload.new(source_component="component1_screen_monitoring", content_type="image",
                              risk_category=RiskCategory.EXPLICIT_VISUAL, confidence_score=0.9,
                              app_or_window="browser")


def _controller(clock, model_seconds=0.6):
    def model_call(prompt, **options):
        clock.now += model_seconds
        return MODEL_LINE, model_seconds
    return FSMController(generator=GrammarConstrainedGenerator(model_call=model_call))


def test_each_wait_is_the_time_before_a_line_and_excludes_the_childs_reply_time():
    clock = FakeClock()
    replies = ["idk", "no", "I was just watching videos with my friend", "ok"]   # INTERCEPT, then the three moves

    def child(_line, it=iter(replies)):
        clock.now += 5.0                                             # the child thinking: not system time
        return next(it)

    result = latency.time_session(_controller(clock), _trigger(), child, clock=clock)

    assert [w.state for w in result.waits] == ["INTERCEPT", "INQUIRE", "INQUIRE", "INQUIRE", "EVALUATE+CONTRACT"]
    assert [w.seconds for w in result.waits] == pytest.approx([0.6, 0.0, 0.0, 0.0, 0.0])
    assert result.record.dialogue_turns == 4


def test_the_intercept_model_call_is_timed_separately_from_the_whole_step():
    clock = FakeClock()
    result = latency.time_session(_controller(clock, model_seconds=0.25), _trigger(),
                                  lambda _line: "I should close it next time instead", clock=clock)
    assert result.model_call_seconds == pytest.approx([0.25])
    assert result.waits[0].source == "model"


def test_a_fallback_intercept_line_is_marked_as_such():
    clock = FakeClock()

    def down(prompt, **options):
        raise ConnectionError("no server")
    controller = FSMController(generator=GrammarConstrainedGenerator(model_call=down))
    result = latency.time_session(controller, _trigger(), lambda _line: "I should close it next time instead",
                                  clock=clock)
    assert result.waits[0].source == "fallback" and result.waits[0].reason == "model_unreachable"
    assert result.model_call_seconds == pytest.approx([0.0])


def test_summarise_reports_percentiles_and_the_count_over_the_target():
    s = latency.summarise([0.5, 0.6, 0.7, 3.5], target=3.0)
    assert s == {"n": 4, "min": 0.5, "median": 0.65, "p95": 3.5, "max": 3.5, "over_target": 1}


def test_summarise_of_nothing_is_empty_not_an_error():
    assert latency.summarise([], target=3.0) == {"n": 0}


def test_the_generator_is_left_as_it_was():
    clock = FakeClock()
    controller = _controller(clock)
    call = controller.generator.model_call
    latency.time_session(controller, _trigger(), lambda _line: "I should close it next time instead", clock=clock)
    assert controller.generator.model_call is call
    assert "generate_with_source" not in vars(controller.generator)


def test_with_no_model_call_of_its_own_the_live_server_is_looked_up_per_call_so_tests_can_block_it():
    clock = FakeClock()
    controller = FSMController(generator=GrammarConstrainedGenerator())   # conftest blocks the live model
    result = latency.time_session(controller, _trigger(), lambda _line: "I should close it next time instead",
                                  clock=clock)
    assert result.waits[0].reason == "model_unreachable"
    assert controller.generator.model_call is None


def test_an_existing_override_of_the_generator_is_put_back():
    clock = FakeClock()
    controller = _controller(clock)
    override = controller.generator.generate_with_source
    controller.generator.generate_with_source = override          # e.g. another wrapper already in place
    latency.time_session(controller, _trigger(), lambda _line: "I should close it next time instead", clock=clock)
    assert vars(controller.generator)["generate_with_source"] is override

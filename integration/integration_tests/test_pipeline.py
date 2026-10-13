"""
Automated version of the integration pipeline -- same chain as
end_to_end_pipeline.py, but with scripted child responses instead of
interactive input(), so it can run unattended (CI, or just `pytest`).

The model is always faked here (offline, or a canned constrained reply), so the
tests give the same result whether or not LM Studio is running. The recorded
online and offline runs are made by component3_socratic_educator/evidence/run_end_to_end.py.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from component3_socratic_educator.src import grammar_decoder as gd                     # noqa: E402
from component3_socratic_educator.src.schemas import _validate_against_schema         # noqa: E402
from integration.comp3_to_comp4 import DEVICE_CHILD_ID                                 # noqa: E402
from integration.end_to_end_pipeline import build_trigger_from, offline_model, run_session  # noqa: E402

# Short replies keep INQUIRE asking (a reply under 4 words is not complete), so all three INQUIRE turns run.
SCRIPT = ["Hmm", "idk", "not sure", "I should've closed it instead, next time I will"]


def _model_replying(question):
    def call(_prompt, **_kwargs):
        return json.dumps({"opener": gd.INTERCEPT_OPENERS[0], "question": question}), 0.5
    return call


def _run(source="component1", model_call=offline_model, script=SCRIPT):
    replies = iter(script)
    return run_session(build_trigger_from(source), lambda _p: next(replies), model_call=model_call)


def test_full_pipeline_runs_end_to_end_with_scripted_responses():
    log = _run()
    record = log["record"]
    assert "Parent-Facing Summary" in log["report"]
    assert record["session_id"] in log["report"]
    assert record["dialogue_turns"] >= 1
    _validate_against_schema(record, "comp3_to_comp4.schema.json")


@pytest.mark.parametrize("source, category", [("component1", "explicit_visual"), ("component2", "cyberbullying")])
def test_each_upstream_component_can_trigger_a_session(source, category):
    log = _run(source)
    assert log["trigger"]["source_component"].startswith(source)
    assert log["trigger"]["risk_category"] == category
    assert log["record"]["session_id"] == log["trigger"]["session_id"]


def test_offline_the_intercept_line_is_the_fallback_line_and_says_why():
    intercept = _run()["turns"][0]
    assert intercept["state"] == "INTERCEPT"
    assert (intercept["source"], intercept["reason"]) == ("fallback", "model_unreachable")


def test_online_the_intercept_line_is_the_models_constrained_line():
    intercept = _run(model_call=_model_replying("What were you doing just now?"))["turns"][0]
    assert intercept["system_text"] == f"{gd.INTERCEPT_OPENERS[0]} What were you doing just now?"
    assert (intercept["source"], intercept["reason"]) == ("model", None)


def test_a_model_line_the_post_check_blocks_is_logged_with_its_reason():
    intercept = _run(model_call=_model_replying("Did that alert worry you?"))["turns"][0]
    assert (intercept["source"], intercept["reason"]) == ("fallback", "alert_language")
    assert intercept["model_output"].endswith("Did that alert worry you?")


def test_every_turn_is_logged_with_where_its_line_came_from():
    turns = _run()["turns"]
    assert [t["state"] for t in turns] == ["INTERCEPT", "INQUIRE", "INQUIRE", "INQUIRE", "EVALUATE", "CONTRACT"]
    assert [t["child_response"] for t in turns[:4]] == SCRIPT
    assert all(t["source"] == "template" for t in turns[1:4])
    assert all(t["source"] is None for t in turns[4:])           # internal / fixed CONTRACT lines


def test_the_record_reaches_component4_as_its_input_row():
    log = _run()
    assert log["component4_row"] == {"child_id": DEVICE_CHILD_ID, **log["record"]}

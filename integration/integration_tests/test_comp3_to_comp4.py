"""
Component 3 -> Component 4 handoff (plan, Thu 8 Oct).

Component 3's record (comp3_to_comp4.schema.json) is validated, then adapted to
the row Component 4 Function 1 (emotional pattern clustering) really reads.
The schema has no child_id; Function 1 groups sessions by it. The adapter adds
it from device configuration (one device = one child profile), so the schema
stays unchanged and the gap is logged in docs/interface-contracts/CHANGELOG.md.

The tests that call Component 4's real code skip when pandas is missing or when
its trained artifacts (component4_profiling_xai/models/, gitignored) are absent.
"""
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from component3_socratic_educator.src.fsm_controller import FSMController                    # noqa: E402
from component3_socratic_educator.src.grammar_decoder import GrammarConstrainedGenerator      # noqa: E402
from component3_socratic_educator.src.schemas import (                                         # noqa: E402
    EmotionalState, EvaluateOutput, RiskCategory, RiskLevel, TriggerPayload)
from integration.comp3_to_comp4 import DEVICE_CHILD_ID, to_emotional_pattern_row              # noqa: E402

C4_MODELS = ROOT / "component4_profiling_xai" / "models"


def _record(**changes):
    fields = dict(session_id="S-1", risk_level=RiskLevel.MODERATE, emotional_state=EmotionalState.CALM,
                  self_regulation_shown=True, escalation_flag=False, dialogue_turns=3,
                  dialogue_summary="3 turns", timestamp="2026-10-08T16:00:00+00:00")
    return EvaluateOutput(**{**fields, **changes})


def _offline(*_a, **_k):
    raise ConnectionError("no model in this test: INTERCEPT takes its fallback line")


def _fsm_sessions(n):
    """n real FSM sessions with scripted child replies and no live model."""
    controller = FSMController(generator=GrammarConstrainedGenerator(model_call=_offline))
    scripts = [["I'm not sure"] * 3,
               ["I guess I was curious", "I should've closed it instead, next time I will"],
               ["Leave me alone, it's none of your business"] * 3]
    visual = [RiskCategory.EXPLICIT_VISUAL, RiskCategory.VIOLENCE, RiskCategory.SELF_HARM_IMAGERY,
              RiskCategory.UNKNOWN_FLAGGED]   # the categories comp1_to_comp3.schema.json allows
    outputs = []
    for i in range(n):
        trigger = TriggerPayload.new("component1_screen_monitoring", "image", visual[i % len(visual)], 0.8)
        replies = iter(scripts[i % len(scripts)])
        output, _ = controller.run(trigger, respond=lambda _p, r=replies: next(r, "ok"))
        outputs.append(output)
    return outputs


# --- the adapter --------------------------------------------------------------

def test_the_row_is_the_schema_record_plus_child_id():
    record = _record()
    row = to_emotional_pattern_row(record, child_id="C-TEST")
    assert row == {"child_id": "C-TEST", **record.to_dict()}


def test_the_default_child_id_comes_from_device_configuration():
    assert to_emotional_pattern_row(_record())["child_id"] == DEVICE_CHILD_ID


def test_a_record_that_breaks_the_schema_never_reaches_component4():
    with pytest.raises(jsonschema.ValidationError):
        to_emotional_pattern_row(_record(dialogue_turns=0))
    with pytest.raises(jsonschema.ValidationError):
        to_emotional_pattern_row(_record(timestamp="yesterday"))


@pytest.mark.parametrize("child_id", ["", "   ", None])
def test_a_missing_child_id_is_refused(child_id):
    with pytest.raises(ValueError):
        to_emotional_pattern_row(_record(), child_id=child_id)


# --- against Component 4's real Function 1 -----------------------------------

def _component4():
    pytest.importorskip("pandas")
    pytest.importorskip("joblib")
    from component4_profiling_xai.src import emotional_pattern
    return emotional_pattern


def test_the_row_has_every_column_component4_requires():
    c4 = _component4()
    assert set(c4.REQUIRED_COLUMNS) <= set(to_emotional_pattern_row(_record()))


def test_without_the_adapter_component4_rejects_the_schema_record():
    # What breaks when Component 4's real input differs from the stub's: the
    # stub reads the schema dict as is; Function 1 needs child_id as well.
    c4 = _component4()
    import pandas as pd
    with pytest.raises(ValueError, match="child_id"):
        c4.clean_sessions(pd.DataFrame([_record().to_dict()]))


def test_component4_cleans_adapted_rows_without_changing_their_meaning():
    c4 = _component4()
    import pandas as pd
    outputs = _fsm_sessions(3)
    cleaned = c4.clean_sessions(pd.DataFrame([to_emotional_pattern_row(o) for o in outputs]))
    by_id = cleaned.set_index("session_id")
    for o in outputs:
        row = by_id.loc[o.session_id]
        assert row["emotional_state"] == o.emotional_state.value
        assert row["self_regulation_shown"] == o.self_regulation_shown     # bools survive C4's string parsing
        assert row["escalation_flag"] == o.escalation_flag
        assert row["dialogue_turns"] == o.dialogue_turns


def test_fsm_sessions_reach_component4_and_get_an_emotional_pattern():
    c4 = _component4()
    if not (C4_MODELS / "emotional_pattern_kmeans.pkl").exists():
        pytest.skip("Component 4's trained artifacts are not in component4_profiling_xai/models/")
    import pandas as pd
    outputs = _fsm_sessions(6)
    patterns = c4.run_emotional_pattern(pd.DataFrame([to_emotional_pattern_row(o) for o in outputs]))
    _, _, metadata = c4.load_artifacts()
    assert list(patterns.index) == [DEVICE_CHILD_ID]
    assert patterns.loc[DEVICE_CHILD_ID, "sessions"] == 6
    assert patterns.loc[DEVICE_CHILD_ID, "pattern"] in metadata["cluster_names"].values()

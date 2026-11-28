"""Presentation tests for evidence that answers different questions."""
import pandas as pd
import pytest

from component4_profiling_xai.src.behavioral_explanation_reliability import Review
from component4_profiling_xai.src.behavioral_profile import ANCHORS
from component4_profiling_xai.src.behavioral_temporal_explain import TemporalExplanation
from ui.pages._behavioral_upgrade import adaptive_summary, explanation_text, guidance, recent_behavior, temporal_text


def profile(feature, deviation, *, available=True, ready=True):
    values = dict.fromkeys(ANCHORS, 0.0)
    values[feature] = deviation
    return {"available": available, "median": (0.0 if ready else float("nan"),) * len(ANCHORS),
            "abrupt": True, "shift": False, "robust_z": tuple(values.values())}


@pytest.mark.parametrize(("deviation", "flagged", "personal", "model"), [
    (4, False, "More", "less unusual"),
    (-4, True, "Fewer", "more unusual"),
])
def test_personal_and_model_directions_are_distinct(deviation, flagged, personal, model):
    adaptive = adaptive_summary(profile("language_flags", deviation))
    explanation = explanation_text({}, flagged, Review("supported", "Language activity"))
    assert adaptive.startswith(personal) and "usual for your child" in adaptive
    assert model in explanation and explanation.startswith("In the AI’s assessment")
    assert "safe" not in explanation and "improving" not in explanation
    assert "Both explanation checks" not in explanation


@pytest.mark.parametrize("review", [None, Review("invalid"), Review("limited"), Review("conflicting")])
def test_missing_or_rejected_reviews_never_name_a_raw_factor(review):
    raw = {"shap": {"Language activity": 8}, "lime": {"Language activity": 7}}
    result = explanation_text(raw, True, review)
    assert "language" not in result and "more unusual" not in result
    advice, question = guidance(pd.Series({"language_flags": 1}), review, False)
    assert "conversation" not in advice and question


def test_incomplete_inputs_and_historical_shift_remain_separate():
    assert "too little activity" in adaptive_summary(profile("language_flags", 4, available=False))
    assert "too few usable earlier records" in adaptive_summary(profile("language_flags", 4, ready=False))
    sustained = TemporalExplanation("sustained", "Language activity", detectors=("cusum",))
    assert "Across several days" in temporal_text(sustained, True, Review("supported"))
    assert "clear factor" in temporal_text(sustained, True, Review("limited"))
    isolated = TemporalExplanation("isolated", "Screen activity")
    assert temporal_text(isolated, False, Review("supported")).startswith("On one recent day")
    records = pd.DataFrame({"date": pd.date_range("2026-03-01", periods=6).date, "flag": False,
                            "ewma_alert": False, "cusum_alert": False, "ewma_active": False,
                            "cusum_active": False})
    day = records.at[5, "date"]
    records.at[5, "flag"] = True
    assert recent_behavior(records, day, 5)[0].startswith("No continuing change")
    records.at[0, "flag"] = True
    assert recent_behavior(records, day, 5)[0].startswith("One earlier day stood out")


def test_urgent_priority_does_not_depend_on_explanation_reliability():
    observation = pd.Series({"screen_escalated_incidents": 1, "language_bullying_flags": 1})
    advice, prompt = guidance(observation, Review("supported", "Language activity"), True, True)
    assert "safety alert above" in advice and prompt is None
    advice, prompt = guidance(observation, Review("limited"), False)
    assert "records include language flagged" in advice and prompt

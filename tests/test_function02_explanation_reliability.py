"""Focused policy and deployed scoring checks for Step 02."""
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from component4_profiling_xai.src.behavioral_explain import SCORE_ATOL
from component4_profiling_xai.src.behavioral_explanation_reliability import (
    Policy, assess_daily, assess_temporal, daily_evidence, perturb, temporal_evidence,
)
from component4_profiling_xai.src.behavioral_profile import BUNDLE, profiles, score_history
from component4_profiling_xai.src.behavioral_temporal_explain import TemporalExplanation
from ui import services
from ui.legacy_pages._behavioral_upgrade import explanation_text, temporal_text


@pytest.fixture(scope="module")
def deployed():
    bundle = joblib.load(BUNDLE)
    data = services.behavioral_daily_source()["daily"]
    data = data.loc[data["child_id"].isin(["C001", "C009"])]
    matrix, _ = profiles(data)
    return bundle, matrix, score_history(data, matrix, bundle)


@pytest.fixture
def daily(deployed):
    bundle, matrix, history = deployed
    index = history.index[history["flag"].eq(True).fillna(False)][-1]
    score = history.at[index, "score"]
    original, variants = daily_evidence(matrix.loc[[index], bundle["features"]], score, bundle)
    return original, score, bundle, variants


def test_actual_shap_and_held_out_lime(daily):
    result, score, bundle, probes = daily
    review = assess_daily(result, score, True, bundle, probes)
    assert review.status == "supported"
    assert review.diagnostics["shap_error"] < SCORE_ATOL
    assert result["lime_variance"] > 0
    assert result["lime_r2"] == pytest.approx(1 - result["lime_rmse"] ** 2 / result["lime_variance"])
    assert 0 <= result["lime_mae"] <= result["lime_rmse"]


@pytest.mark.parametrize("failure", ["shap", "score", "reference", "groups"])
def test_invalid_shap_is_neutral(daily, failure):
    result, score, bundle, probes = daily
    altered = dict(result)
    if failure == "shap":
        altered["shap"] = dict.fromkeys(result["shap"], np.nan)
    elif failure == "groups":
        altered["shap"] = {"Unknown": sum(result["shap"].values())}
    else:
        altered[failure] += 0.001
    review = assess_daily(altered, score, True, bundle, probes)
    assert review.status == "invalid"
    assert "unavailable" in explanation_text(altered, True, review)


def test_lime_quality_and_disagreement_do_not_veto_stable_shap(daily):
    result, score, bundle, probes = daily
    review = assess_daily(result, score, True, bundle, probes)
    group = review.group
    high = {**result, "lime": result["shap"], "lime_r2": 0.99}
    low = {**high, "lime_r2": -0.25}
    disagree = {**high, "lime": {key: (1 if key != group else -1) for key in high["lime"]}}
    for variant, agreement in [(high, True), (low, False), (disagree, False)]:
        decision = assess_daily(variant, score, True, bundle, probes, Policy(1, 0.7))
        assert decision.status == "supported"
        assert decision.group == group and decision.corroborated == agreement
        assert ("Both explanation checks" in explanation_text(variant, True, decision)) == agreement


def test_unstable_rank_and_near_boundary_are_neutral(daily):
    result, score, bundle, probes = daily
    initial = assess_daily(result, score, True, bundle, probes)
    group = initial.group
    other = next(key for key in result["shap"] if key != group)
    reordered = dict(result["shap"])
    reordered[other], reordered[group] = reordered[group], reordered[other]
    swapped = {**result, "shap": reordered}
    review = assess_daily(result, score, True, bundle, [(result, score), (swapped, score)], Policy(1, 0.7))
    assert review.status in {"limited", "conflicting"}
    assert review.diagnostics["rank_stability"] == 0.5
    assert review.group is None
    boundary = {**bundle, "threshold": score + 0.001}
    moved = {**result, "score": score + 0.002, "shap": dict(result["shap"])}
    moved["shap"][group] += 0.002
    review = assess_daily(result, score, False, boundary, [(moved, score + 0.002)])
    assert review.diagnostics["decision_flip"] and review.group is None
    assert boundary["threshold"] == score + 0.001


def test_missing_explanation_data(daily):
    _, score, bundle, _ = daily
    assert assess_daily(None, score, True, bundle, []).status == "invalid"
    assert assess_daily(None, np.nan, False, bundle, []).status == "insufficient_data"


def test_perturbation_keeps_counts_missing_and_probability_relationships():
    matrix = pd.DataFrame({"language_flags": [0, 1, 2, 2], "language_max_risk": [0, 1, .8, np.nan],
                           "language_mean_risk": [0, 1, .5, np.nan], "other": [4, 3, 2, 1]})
    for amount in (-.02, .02):
        changed = perturb(matrix, amount)
        pd.testing.assert_frame_equal(changed[["language_flags", "other"]], matrix[["language_flags", "other"]])
        assert changed.iloc[0].eq(matrix.iloc[0]).all()
        assert changed.iloc[3].isna().equals(matrix.iloc[3].isna())
        assert changed.at[1, "language_max_risk"] == changed.at[1, "language_mean_risk"]
        assert changed.at[2, "language_mean_risk"] <= changed.at[2, "language_max_risk"] <= 1


def test_temporal_fallback_and_current_detector_wording():
    raw = TemporalExplanation("sustained", "Language activity", reconstruction_error=0.0, detectors=("ewma",))
    review = assess_temporal(raw, [raw, raw])
    assert review.status == "supported"
    assert "also active on the selected day" in temporal_text(raw, False, review)
    changed = replace(raw, status="isolated")
    review = assess_temporal(raw, [raw, changed], Policy(1, 0.7))
    assert review.status == "limited" and "clear factor" in temporal_text(raw, True, review)
    assert assess_temporal(replace(raw, reconstruction_error=1), [raw]).status == "invalid"
    assert assess_temporal(TemporalExplanation("insufficient"), []).status == "insufficient_data"


def test_temporal_reliability_replays_only_selected_child_and_date(deployed):
    bundle, matrix, history = deployed
    day = history["date"].max() - pd.Timedelta(days=5)
    before = history.copy(deep=True)
    threshold = bundle["threshold"]
    original = temporal_evidence(history, matrix, bundle, "C001", day)
    selected = history["date"].le(day) & history["child_id"].eq("C001")
    assert original == temporal_evidence(history.loc[selected], matrix.loc[selected], bundle, "C001", day)
    assert original != temporal_evidence(history, matrix, bundle, "C009", day)
    pd.testing.assert_frame_equal(history, before)
    assert bundle["threshold"] == threshold

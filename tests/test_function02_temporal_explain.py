"""Temporal evidence rules and replay against the actual deployed detector."""
import joblib
import numpy as np
import pandas as pd
import pytest

from component4_profiling_xai.src.behavioral_profile import BUNDLE, profiles, score_history
from component4_profiling_xai.src.behavioral_temporal_explain import (
    SCORE_ATOL, compare_contributions, explain_temporal,
)
from ui import services
from ui.pages._behavioral_upgrade import temporal_text


@pytest.fixture
def contributions():
    dates = pd.date_range("2026-10-01", periods=14)
    values = pd.DataFrame({"Language activity": 0.0, "Screen activity": -0.25}, index=range(len(dates)))
    history = pd.DataFrame({"child_id": "demo", "date": dates, "flag": False})
    return history, values


def compare(history, values, day=None):
    scored = history.assign(score=values.sum(axis=1, min_count=values.shape[1]) + 2.5)
    return compare_contributions(scored, values, "demo", day or history["date"].iloc[-1], 1, window=7)


@pytest.mark.parametrize(("recent", "status"), [
    ([0, 0, 0, 0, 0, 0, 0], "stable"),
    ([0, 0, 0, 0.2, 0.4, 0.6, 0.8], "sustained"),
    ([0, 0, 0, 0, 0.5, 0.5, 0.5], "sustained"),
    ([0, 0, 0, 1, 0, 0, 0], "isolated"),
    ([-1, -1, -1, -1, -1, -1, -1], "unclear"),
    ([1, 1, 1, 0, 0, 0, 0], "unclear"),
])
def test_signed_temporal_patterns(contributions, recent, status):
    rows, values = contributions
    values.loc[7:, "Language activity"] = recent
    result = compare(rows, values)
    assert result.status == status
    assert result.deltas["Language activity"] == pytest.approx(np.mean(recent))
    assert result.reconstruction_error <= SCORE_ATOL
    assert result.group == ("Language activity" if status in {"sustained", "isolated"} else None)


def test_negative_shap_can_rise_without_pushing_score_above_reference(contributions):
    rows, values = contributions
    values.loc[7:, "Screen activity"] = -0.05
    result = compare(rows, values)
    assert result.deltas["Screen activity"] == pytest.approx(0.20)
    assert result.status == "unclear"


def test_missing_days_excluded_and_break_sustained_run(contributions):
    rows, values = contributions
    values.loc[7:, "Language activity"] = 1
    rows = rows.drop(index=12)
    result = compare(rows, values)
    assert result.recent_count == 6
    assert result.deltas["Language activity"] == pytest.approx(1)
    assert result.status == "unclear"
    assert compare(rows.drop(index=[8, 9]), values).status == "insufficient"
    values.loc[13] = np.nan
    assert compare(rows, values).status == "insufficient"


def test_only_current_overlapping_detector_evidence(contributions):
    rows, values = contributions
    values.loc[7:, "Language activity"] = 1
    rows["ewma_active"] = False
    rows["ewma_alert"] = False
    rows.loc[7:9, "ewma_active"] = True
    rows.loc[7, "ewma_alert"] = True
    rows["cusum_alert"] = False
    rows["cusum_active"] = True
    rows.loc[0, "cusum_alert"] = True
    assert compare(rows, values).detectors == ()
    rows.loc[11:, "ewma_active"] = True
    rows.loc[11, "ewma_alert"] = True
    assert compare(rows, values).detectors == ("ewma",)
    assert "also active" in temporal_text(compare(rows, values), False)


def test_tied_and_unverified_results_have_neutral_wording(contributions):
    rows, values = contributions
    values.loc[7:] = 1
    values.loc[:6] = 0
    assert compare(rows, values).status == "unclear"
    rows["score"] = values.sum(axis=1)
    rows.loc[13, "score"] += 1
    result = compare_contributions(rows, values, "demo", rows["date"].iloc[-1], 1, window=7)
    assert result.status == "unsupported"
    assert "unavailable" in temporal_text(result, False)


def test_deployed_scores_reconstruct_and_replay_as_of_selected_child_and_date():
    dataset = services.behavioral_daily_source()["daily"]
    rows = dataset.loc[dataset["child_id"].isin(["C001", "C009"])]
    bundle = joblib.load(BUNDLE)
    matrix, _ = profiles(rows)
    history = score_history(rows, matrix, bundle)
    day = rows["date"].max() - pd.Timedelta(days=7)
    for window in (7, 14):
        original = explain_temporal(history, matrix, bundle, "C001", day, window)
        assert original.deltas and original.reconstruction_error <= SCORE_ATOL
        eligible = history["date"].le(day) & history["child_id"].eq("C001")
        assert original == explain_temporal(history.loc[eligible], matrix.loc[eligible], bundle, "C001", day, window)
        assert original != explain_temporal(history, matrix, bundle, "C009", day, window)
        assert original != explain_temporal(history, matrix, bundle, "C001", rows["date"].max(), window)
    corrupted = history.copy()
    corrupted.loc[corrupted["date"].eq(day), "score"] += 0.01
    assert explain_temporal(corrupted, matrix, bundle, "C001", day).status == "unsupported"

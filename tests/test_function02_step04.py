"""Research-only challenger feature, calibration, and lock checks."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from component4_profiling_xai import evaluate_anomaly_optimization as experiment
from component4_profiling_xai.src.behavioral_anomaly import anomaly_scores, assessable_days
from component4_profiling_xai.src.behavioral_profile import BUNDLE, profiles
from ui import services


@pytest.fixture
def daily():
    dataset = services.behavioral_daily_source()["daily"]
    children = dataset["child_id"].unique()[:2]
    return dataset.loc[dataset["child_id"].isin(children)].groupby("child_id").head(12).reset_index(drop=True)


def test_new_features_distinguish_zero_missing_and_invalid_shares(daily):
    daily.loc[:2, ["screen_incidents", "screen_night_incidents"]] = [0, 0]
    daily.at[1, "screen_monitor_observed"] = 0
    daily.at[2, "screen_night_incidents"] = 1
    frame = experiment.features(daily)
    assert frame.at[0, "screen_night_share"] == 0
    assert pd.isna(frame.at[1, "screen_night_share"])
    assert pd.isna(frame.at[2, "screen_night_share"])
    assert not assessable_days(daily, frame).at[1]


def test_later_records_child_identity_and_labels_do_not_supply_predictors(daily):
    first_child = daily.at[0, "child_id"]
    first = daily["child_id"].eq(first_child)
    cut = daily.loc[first, "date"].iloc[8]
    original = experiment.features(daily)
    changed = daily.assign(is_anomaly=1, anomaly_type="audit", episode_id="metadata")
    changed.loc[~first | daily["date"].gt(cut), "screen_incidents"] = 9999
    scope = first & daily["date"].le(cut)
    pd.testing.assert_frame_equal(original.loc[scope], experiment.features(changed).loc[scope])
    deployed = joblib.load(BUNDLE)
    reference, _ = profiles(daily)
    cols = deployed["features"]
    np.testing.assert_array_equal(anomaly_scores(deployed["pipeline"], reference[cols]),
                                  anomaly_scores(deployed["pipeline"], original[cols]))
    for columns in experiment.build_designs(cols).values():
        assert {"child_id", "date", "episode_id", "anomaly_type", "is_anomaly", "persona"}.isdisjoint(columns)


def test_strict_budget_respects_ties():
    y = np.asarray([0, 0, 0, 1, 1])
    scores = np.asarray([0., 1., 2., 2., 3.])
    assert (scores > experiment.cutoff(y, scores, 0)).tolist() == [False, False, False, False, True]
    assert (scores > experiment.cutoff(y, scores, 1 / 3)).tolist() == [False, False, True, True, True]


def test_holdout_refuses_a_mismatched_lock_before_building_data(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "OUT", tmp_path)
    (tmp_path / "selection_lock.json").write_text(json.dumps({"script_sha256": "mismatched"}))

    def forbidden(*args):
        pytest.fail("Holdout generation must follow lock verification")

    monkeypatch.setattr(experiment, "build_dataset", forbidden)
    with pytest.raises(RuntimeError, match="Lock mismatch"):
        experiment.holdout()

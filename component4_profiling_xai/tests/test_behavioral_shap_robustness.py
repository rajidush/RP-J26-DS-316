from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component4_profiling_xai import behavioral_shap_robustness_synthetic as design
from component4_profiling_xai import evaluate_behavioral_shap_robustness as evaluation
from component4_profiling_xai.src.behavioral_anomaly import FEATURE_SET_A, assessable_days, feature_matrix
from component4_profiling_xai.src.behavioral_shap import normal_training_rows


@pytest.fixture(scope="module")
def daily():
    return design.generate()


def test_replication_invariants_and_complete_outage_distribution(daily):
    train = normal_training_rows(pd.read_csv(evaluation.DATA))
    previous = pd.read_csv(evaluation.HOLDOUT, usecols=["child_id"])
    evaluation.validate_design(daily, train, previous)
    assert set(daily.groupby("arm").size()) == {240}
    assert set(daily.groupby(["arm", "pair_id"]).is_anomaly.sum()) == {3}
    coverage = daily.loc[daily.arm.eq("coverage_loss")]
    assert coverage.outage.value_counts().to_dict() == {
        "none": 180, "screen": 20, "language": 20, "both": 20,
    }
    features = feature_matrix(daily, FEATURE_SET_A)
    assert assessable_days(daily, features).equals(daily.outage.eq("none"))


def test_calendar_is_only_model_input_changed_and_night_counts_are_nested(daily):
    keys = ["pair_id", "day_index"]
    base = daily.loc[daily.arm.eq("reference")].set_index(keys)
    weekend = daily.loc[daily.arm.eq("weekend")].set_index(keys)
    night = daily.loc[daily.arm.eq("night_40pct")].set_index(keys)
    assert not base.is_weekend.any() and weekend.is_weekend.all()
    pd.testing.assert_frame_equal(base[FEATURE_SET_A[:-1]], weekend[FEATURE_SET_A[:-1]])
    assert night.night_risk_events.ge(base.night_risk_events).all()
    assert night.night_risk_events.gt(base.night_risk_events).any()


def test_cluster_metric_keeps_undefined_rates_and_valid_coverage():
    vec = np.zeros(len(evaluation.OUTCOMES), dtype=int)
    vec[evaluation.OUTCOMES.index("TP")] = 2
    vec[evaluation.OUTCOMES.index("FN")] = 1
    vec[evaluation.OUTCOMES.index("insufficient_data")] = 1
    metrics = evaluation.calculate(vec)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["coverage"] == pytest.approx(.75)
    assert metrics["positive_detection_all_days"] == pytest.approx(.50)
    assert np.isnan(metrics["fpr"])
    assert evaluation.interval(np.array([np.nan])) is None

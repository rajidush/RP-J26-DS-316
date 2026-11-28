"""Regression coverage for optimization selection and holdout boundaries."""
import numpy as np
import pandas as pd
import pytest

from component4_profiling_xai.optimize_function02 import bootstrap, score_curves, temporal_outcomes
from component4_profiling_xai.src.behavioral_optimization import Hybrid, calibration, development
from component4_profiling_xai.train_function02 import metrics


def test_precision_recall_cutoff_respects_strict_ties():
    labels = [0, 0, 1, 1]
    scores = np.array([0.1, 0.2, 0.2, 0.9])
    result = calibration(labels, scores)
    assert result["threshold"] < 0.2
    assert (scores > result["threshold"]).tolist() == [False, True, True, True]
    actual = metrics(labels, scores, result["threshold"])
    assert actual["f1"] == pytest.approx(result["validation_f1"])
    assert actual["fpr"] == pytest.approx(result["validation_fpr"])


def test_development_discards_consumed_test_and_checks_dates():
    rows = pd.DataFrame({"date": ["2026-07-01", "2026-08-01", "2026-09-01"],
                         "split": ["train", "val", "test"], "is_anomaly": [0, 1, 99]})
    assert development(rows)["is_anomaly"].tolist() == [0, 1]
    with pytest.raises(ValueError, match="overlap"):
        development(rows.assign(date="2026-07-01"))


def test_hybrid_personal_rarity_has_safe_cold_start_and_monotonic_tail():
    scored = Hybrid.personal_scores(pd.DataFrame({"a": [None, -3, 1, 4], "b": [None, 0, 2, 3]}))
    assert scored.tolist() == [0, 0, 2, 4]
    tail = Hybrid.rarity(np.array([0, 0, 1, 2, 3]), scored)
    assert tail[0] == tail[1] == 0
    assert 0 < tail[2] < tail[3] < np.inf


def test_temporal_credit_requires_observed_labeled_day_and_counts_false_alerts():
    daily = pd.DataFrame({"date": ["2026-07-01", "2026-07-02", "2026-07-03"],
                          "episode_id": ["", "", "E1"], "is_anomaly": [0, 0, 1]})
    episodes = pd.DataFrame({"episode_id": ["E1"], "anomaly_type": ["spike"], "intensity": ["subtle"],
                             "start_date": ["2026-07-02"], "observed_days": [1]})
    changes = pd.Series([False, True, True])
    result, details = temporal_outcomes(daily, pd.Series([True] * 3), episodes, pd.Series([0, 1, 3]), changes)
    assert result["false_alert_rate"] == 0.5
    assert result["detected"] == 1 and details["delay_days"].tolist() == [1]


def test_cluster_bootstrap_preserves_pairing_and_pr_curves_are_labeled():
    rows = pd.DataFrame({"child_id": ["a", "a", "b", "b"], "label": [0, 1, 0, 1],
                         "global": [False, True, False, True], "same": [False, True, False, True]})
    assert bootstrap(rows, "same") == {"delta_f1_95": [0.0, 0.0], "delta_fpr_95": [0.0, 0.0]}
    curves = score_curves(rows[["global", "same", "label"]])
    assert set(curves["design"]) == {"global", "same"}

def test_research_interface_shows_measured_optimization_without_removing_original_panels():
    import json
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from component4_profiling_xai.optimize_function02 import OUTPUT

    report = OUTPUT / "evaluation.json"
    if not report.is_file():
        pytest.skip("Generate the sealed optimization study before checking published results")
    selected = json.loads(report.read_text(encoding="utf-8"))["selected"]["design"]
    app = Path(__file__).resolve().parents[1] / "component4_profiling_xai/experiments/archived_research_function02.py"
    page = AppTest.from_file(str(app), default_timeout=45).run()
    assert not page.exception
    assert len(page.header) == 6
    assert any("Optimization and sealed synthetic holdout" == item.label for item in page.expander)
    assert any(selected in item.value for item in page.markdown)

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component4_profiling_xai import behavioral_shap_independent_synthetic as simulator
from component4_profiling_xai import evaluate_behavioral_shap_independent as evaluation
from component4_profiling_xai.src.behavioral_anomaly import FEATURE_SET_A, assessable_days, feature_matrix
from component4_profiling_xai.src.behavioral_shap import mask_violations


def test_independent_cohort_is_reproducible_and_preserves_coverage_semantics():
    daily = simulator.generate()
    pd.testing.assert_frame_equal(daily, simulator.generate())
    assert len(daily) == 672 and daily.child_id.nunique() == 24
    assert not daily.duplicated(["child_id", "date"]).any()
    assert daily.child_id.str.startswith("XAI-IV1-").all()
    assert set(daily.groupby("scenario").size()) == {336}
    assert set(daily.groupby("scenario").profile.nunique()) == {3}

    missing = daily.screen_outage | daily.language_outage
    assert missing.any() and (~missing).any()
    assert (daily.screen_outage & daily.language_outage).any()
    features = feature_matrix(daily, FEATURE_SET_A)
    assert assessable_days(daily, features).equals(~missing)
    assert not mask_violations(features.loc[~missing]).any(axis=None)
    assert daily.loc[missing, "total_risk_events"].isna().all()
    assert daily.loc[missing, "is_anomaly"].eq(1).any()


def test_metrics_exclude_missing_activity_from_confusion_counts():
    rows = [{"is_flagged": flag, "is_anomaly": label} for flag, label in (
        (True, 1), (True, 0), (False, 1), (False, 0), (None, 1),
    )]
    summary = evaluation.model_metrics(rows)
    assert [summary[key] for key in ("tp", "fp", "fn", "tn")] == [1, 1, 1, 1]
    assert summary["precision_assessed"] == summary["recall_assessed"] == summary["f1_assessed"] == .5
    assert summary["positive_detection_all_days"] == pytest.approx(1 / 3)


def test_freeze_refuses_mutated_payload(tmp_path, monkeypatch):
    file = tmp_path / "payload"
    file.write_bytes(b"original")
    manifest = {"sha256": {"payload": evaluation.digest(file)}}
    file.write_bytes(b"mutated")
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="frozen input changed"):
        evaluation.verify_freeze(manifest)


def test_cutoff_slices_both_sides_and_ignores_missing():
    rows = [
        {"child_id": "X", "date": f"date-{i}", "margin": margin, "is_flagged": flag,
         "status": "insufficient_data" if flag is None else "uncertain" if flag else "not_flagged",
         "reasons": []}
        for i, (margin, flag) in enumerate([(0.1, True), (0.05, True), (-0.03, False), (None, None)])
    ]
    panel = evaluation.cutoff_panel(rows, .06)
    assert panel["nearest_flagged"]["case_keys"] == [["X", "date-1"], ["X", "date-0"]]
    assert panel["nearest_unflagged"]["case_keys"] == [["X", "date-2"]]
    assert panel["within_frozen_background_span_flagged"]["n"] == 1
    assert panel["within_frozen_background_span_unflagged"]["n"] == 1

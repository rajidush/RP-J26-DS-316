import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import behavioral_anomaly as BA
from src.behavioral_dataset import DATASET_BASELINE_COLUMNS, MODEL_FEATURES

DATA = Path(__file__).resolve().parents[1] / "data" / "behavioral_daily.csv"


def make_daily(n=300, seed=0, features=BA.FEATURE_SET_B):
    """Normal-looking random rows with every model feature plus keys and labels."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.poisson(1.0, size=(n, len(features))).astype(float), columns=features)
    df["is_weekend"] = rng.random(n) < 2 / 7
    df.insert(0, "child_id", [f"C{i % 5:03d}" for i in range(n)])
    df.insert(1, "date", pd.date_range("2026-07-01", periods=n, freq="D").astype(str))
    df["is_anomaly"] = 0
    df["anomaly_type"] = "none"
    df["episode_id"] = np.nan
    return df


# ------------------------------------------------------------------ feature sets

def test_feature_sets_come_from_project_definitions():
    assert BA.FEATURE_SET_B == MODEL_FEATURES
    assert len(BA.FEATURE_SET_B) == 45
    assert len(BA.FEATURE_SET_A) == 27
    assert set(BA.BASELINE_FEATURES) == set(DATASET_BASELINE_COLUMNS) - {"baseline_ready"}
    assert set(BA.FEATURE_SET_A).isdisjoint(BA.BASELINE_FEATURES)
    # order of Set A follows Set B
    assert BA.FEATURE_SET_A == [f for f in BA.FEATURE_SET_B if f in BA.FEATURE_SET_A]


def test_no_label_or_metadata_column_is_a_feature():
    for feats in BA.FEATURE_SETS.values():
        assert set(feats).isdisjoint(BA.NON_FEATURE_COLUMNS)


@pytest.mark.skipif(not DATA.is_file(), reason="dataset not present")
def test_feature_sets_exist_in_dataset():
    cols = pd.read_csv(DATA, nrows=1).columns
    assert set(BA.FEATURE_SET_B) <= set(cols)


def test_feature_matrix_casts_to_float_and_rejects_labels():
    df = make_daily(10)
    X = BA.feature_matrix(df, BA.FEATURE_SET_A)
    assert list(X.columns) == BA.FEATURE_SET_A
    assert (X.dtypes == float).all()
    with pytest.raises(ValueError, match="cannot be model input"):
        BA.feature_matrix(df, ["is_anomaly"])
    with pytest.raises(ValueError, match="missing feature"):
        BA.feature_matrix(df.drop(columns="is_weekend"), BA.FEATURE_SET_A)


def test_check_features_finds_constant_duplicate_and_correlated():
    df = make_daily(200)
    df["language_typed_flags"] = 0.0                                   # constant
    df["copy_of_flags"] = df["language_flags"]                         # exact duplicate
    df["almost_flags"] = df["language_flags"] + np.random.default_rng(1).normal(0, 0.01, len(df))
    feats = ["language_flags", "language_typed_flags", "copy_of_flags", "almost_flags", "screen_incidents"]
    rep = BA.check_features(df, feats)
    assert rep["constant"] == ["language_typed_flags"]
    assert rep["duplicates"] == [("language_flags", "copy_of_flags")]
    pairs = {frozenset(p) for p in rep["high_corr"][["feature_1", "feature_2"]].itertuples(index=False)}
    assert frozenset({"language_flags", "almost_flags"}) in pairs
    assert BA.clean_feature_list(df, feats) == ["language_flags", "almost_flags", "screen_incidents"]


# ------------------------------------------------------------------ models and thresholds

@pytest.mark.parametrize("algorithm", list(BA.ALGORITHMS))
def test_pipeline_scores_outliers_higher_and_handles_nan(algorithm):
    train = make_daily(300)
    X = BA.feature_matrix(train, BA.FEATURE_SET_B)
    pipe = BA.build_pipeline(algorithm, {"nu": 0.05} if algorithm == "one_class_svm" else {}).fit(X)

    probe = make_daily(2, seed=5)
    counts = [f for f in BA.FEATURE_SET_A if f != "is_weekend"]
    probe.loc[1, counts] = 40.0                                        # extreme day
    probe.loc[0, BA.BASELINE_FEATURES] = np.nan                        # warm-up day
    scores = BA.anomaly_scores(pipe, BA.feature_matrix(probe, BA.FEATURE_SET_B))
    assert np.isfinite(scores).all()
    assert scores[1] > scores[0]


def test_build_pipeline_is_reproducible_and_validates_name():
    X = BA.feature_matrix(make_daily(200), BA.FEATURE_SET_A)
    a = BA.anomaly_scores(BA.build_pipeline("isolation_forest", seed=7).fit(X), X)
    b = BA.anomaly_scores(BA.build_pipeline("isolation_forest", seed=7).fit(X), X)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="unknown algorithm"):
        BA.build_pipeline("kmeans")


def test_threshold_and_evaluate():
    train_scores = np.arange(100, dtype=float)
    assert BA.threshold_from_quantile(train_scores, 0.9) == pytest.approx(89.1)
    with pytest.raises(ValueError):
        BA.threshold_from_quantile(train_scores, 1.0)

    m = BA.evaluate([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2], threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5
    # nothing flagged: no division error
    assert BA.evaluate([0, 1], [0.0, 0.0], threshold=1.0)["precision"] == 0.0


def test_tune_threshold_picks_best_f1_and_prefers_higher_quantile_on_ties():
    train_scores = np.linspace(0, 1, 101)
    val_scores = np.array([0.1, 0.2, 0.3, 0.97, 0.99])
    y_val = np.array([0, 0, 0, 1, 1])
    best, table = BA.tune_threshold(train_scores, val_scores, y_val, [0.5, 0.9, 0.95])
    assert len(table) == 3
    assert best["f1"] == 1.0
    assert best["quantile"] == 0.95                                    # all three tie at F1 = 1


# ------------------------------------------------------------------ artifacts

def test_save_load_and_predict_roundtrip(tmp_path):
    train = make_daily(200)
    feats = BA.FEATURE_SET_A
    pipe = BA.build_pipeline("one_class_svm", {"nu": 0.05}).fit(BA.feature_matrix(train, feats))
    thr = BA.threshold_from_quantile(BA.anomaly_scores(pipe, BA.feature_matrix(train, feats)), 0.95)
    BA.save_artifacts(pipe, {"algorithm": "one_class_svm", "features": feats, "threshold": thr}, tmp_path)

    loaded, meta = BA.load_artifacts(tmp_path)
    assert meta["features"] == feats
    json.loads((tmp_path / BA.METADATA_FILE).read_text())

    new = make_daily(20, seed=3)
    out = BA.predict_anomalies(new, tmp_path)
    assert list(out.columns) == ["child_id", "date", "anomaly_score", "is_flagged"]
    expected = BA.anomaly_scores(pipe, BA.feature_matrix(new, feats))
    assert np.allclose(out["anomaly_score"], expected)
    assert (out["is_flagged"] == (expected > thr)).all()


def test_save_artifacts_requires_core_metadata(tmp_path):
    pipe = BA.build_pipeline("isolation_forest")
    with pytest.raises(ValueError, match="threshold"):
        BA.save_artifacts(pipe, {"algorithm": "x", "features": []}, tmp_path)


@pytest.mark.skipif(not (BA.ARTIFACT_DIR / BA.PIPELINE_FILE).is_file(), reason="model not trained yet")
def test_saved_model_loads_and_scores_dataset():
    pipe, meta = BA.load_artifacts()
    assert meta["features"] in BA.FEATURE_SETS.values()
    assert {"threshold", "threshold_quantile", "algorithm", "params", "sklearn_version"} <= meta.keys()
    if DATA.is_file():
        df = pd.read_csv(DATA)
        out = BA.predict_anomalies(df[df.split == "test"])
        assert len(out) == 680 and out["anomaly_score"].notna().all()

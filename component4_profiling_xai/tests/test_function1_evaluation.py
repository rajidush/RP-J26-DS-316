"""Tests for the read-only Function 1 evaluation runner."""

import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from component4_profiling_xai import evaluate_function1 as evaluation
from component4_profiling_xai.src import emotional_pattern as f1


def test_metrics_match_independent_four_point_calculation():
    metrics = evaluation.cluster_metrics(
        np.array([[0.0], [2.0], [8.0], [10.0]]), np.array([0, 0, 1, 1]), np.array([[1.0], [9.0]]))
    assert metrics["sizes"] == [2, 2]
    assert metrics["inertia"] == 4.0
    assert metrics["silhouette"] == pytest.approx((7 / 9 + 5 / 7) / 2)
    assert metrics["davies_bouldin"] == pytest.approx(0.25)
    assert metrics["calinski_harabasz"] == pytest.approx(32)


@pytest.mark.parametrize("labels", [[], [0, 0, 0], [0, 1, 2]])
def test_metrics_report_undefined_partition_scores_as_null(labels):
    metrics = evaluation.cluster_metrics(
        np.zeros((len(labels), 1)), np.array(labels, dtype=int), np.zeros((3, 1)))
    assert metrics["silhouette"] is metrics["davies_bouldin"] is metrics["calinski_harabasz"] is None
    assert metrics["inertia"] == 0


@pytest.mark.parametrize("k, values", [(1, [1, 2, 3]), (3, [1, 2, 3]), (3, [1, 1, 2, 2])])
def test_unsupported_cluster_counts_are_rejected(k, values):
    with pytest.raises(ValueError, match="distinct vectors"):
        evaluation.fit_experiment(np.array(values).reshape(-1, 1), k, 42)


def test_sweep_skips_counts_above_the_distinct_and_sample_limits():
    scaled = np.array([[0.0], [0.0], [1.0], [1.0]])
    with threadpool_limits(limits=1):
        results = evaluation.sweep(scaled, np.array([0, 0, 1, 1]), 2)
    assert results[0]["ari_vs_saved"] == 1
    assert all("skipped" in row for row in results[1:])


def test_empty_sessions_and_insufficient_windows_are_rejected():
    with pytest.raises(ValueError, match="requires session records"):
        evaluation.prepare(pd.DataFrame(), {})
    single = pd.DataFrame([{"child_id": "A", "session_id": "1", "timestamp": "2026-01-01",
                            "emotional_state": "calm", "self_regulation_shown": False,
                            "dialogue_turns": 2}])
    with pytest.raises(ValueError, match="reliable window"):
        evaluation.prepare(single, {"window_days": 14, "min_sessions": 3})


def test_window_vectors_match_production_timeline_and_selected_session_counts():
    data = pd.read_csv(evaluation.COMPONENT / "data" / "educator_sessions.csv")
    _, _, metadata = f1.load_artifacts()
    sample = data[data["child_id"].isin(["C001", "C004", "C031"])]
    _, windows, _ = evaluation.prepare(sample, metadata["trend_settings"])
    with threadpool_limits(limits=1):
        _, trend = f1.run_dashboard_outputs(sample)
        model, scaler, _ = f1.load_artifacts()
        labels = model.predict(scaler.transform(windows))
    assignments = dict(zip(windows.index, labels))
    expected = {(child, row["window"] - 1): row["pattern"] for child, details in trend.items()
                for row in details["windows"] if row["reliable"]}
    assert {key: metadata["cluster_names"][label] for key, label in assignments.items()} == expected
    assert np.allclose(windows[f1.FEATURES[:4]].sum(axis=1), 1)


def test_profiles_report_original_scale_mean_median_and_sample_standard_deviation():
    features = pd.DataFrame({"p_calm": [0.0, 0.5, 1.0], "mean_turns": [2.0, 3.0, 4.0]})
    results = evaluation.profiles(features, np.array([0, 0, 1]), {0: "A", 1: "B", 2: "C"})
    assert results[0]["count"] == 2
    assert results[0]["proportion"] == pytest.approx(2 / 3)
    assert results[0]["features"]["p_calm"]["mean"] == 0.25
    assert results[0]["features"]["mean_turns"]["median"] == 2.5
    assert results[0]["features"]["mean_turns"]["sd"] == pytest.approx(np.sqrt(0.5))
    assert results[1]["features"]["p_calm"]["sd"] is None
    assert results[2]["count"] == 0 and results[2]["features"] == {}


def test_grouped_split_is_reproducible_and_never_splits_a_child_across_sets():
    groups = np.repeat(np.arange(40), np.arange(40) % 4 + 1)
    holdout, samples = evaluation.grouped_split(groups)
    repeated_holdout, repeated_samples = evaluation.grouped_split(groups[::-1])
    assert np.array_equal(holdout, repeated_holdout)
    assert len(holdout) == 10
    assert len(samples) == len(evaluation.SEEDS)
    for sample, repeated in zip(samples, repeated_samples):
        assert len(sample) == 24
        assert not np.any(np.isin(groups, sample) & np.isin(groups, holdout))
        assert np.array_equal(sample, repeated)


def test_stability_is_invariant_to_cluster_id_permutations():
    reference = np.array([0, 0, 0, 1, 1, 2])
    result = evaluation.stability([reference, np.array([1, 1, 1, 2, 2, 0])], reference, 3)
    assert result["pairwise_ari"]["mean"] == 1
    assert [size["mean"] for size in result["aligned_cluster_sizes"]] == [3, 2, 1]
    assert all(size["sd"] == 0 for size in result["aligned_cluster_sizes"])


def test_research_runner_is_reproducible_and_keeps_artifacts_unchanged(tmp_path):
    from ui.services import demo_sessions

    before = evaluation.artifact_hashes(f1.ARTIFACT_DIR)
    assert len(before) == 4
    with pytest.raises(ValueError, match="outside the production artifact directory"):
        evaluation.run(f1.ARTIFACT_DIR / "invalid-output", demo_sessions())

    output = tmp_path / "research"
    with threadpool_limits(limits=1):
        first = evaluation.run(output, demo_sessions())
        original_json = (output / "metrics.json").read_bytes()
        second = evaluation.run(output, demo_sessions())
    assert first == second
    assert (output / "metrics.json").read_bytes() == original_json
    assert json.loads(original_json)["artifact_sha256"] == before
    assert evaluation.artifact_hashes(f1.ARTIFACT_DIR) == before
    assert {path.stem for path in output.glob("*.png")} == {
        "elbow", "silhouette_vs_k", "davies_bouldin_vs_k", "calinski_harabasz_vs_k",
        "cluster_sizes", "saved_assignments_pca", "window_feature_profiles",
    }

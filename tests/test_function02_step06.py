"""Exact XAI batching, per-request memory bounds, and dashboard equivalence."""
import socket

import joblib
import numpy as np
import pandas as pd
import pytest

from component4_profiling_xai import evaluate_xai_runtime as benchmark
from component4_profiling_xai.evaluate_temporal_explain import controlled
from component4_profiling_xai.src.behavioral_explain import explain
from component4_profiling_xai.src.behavioral_explanation_reliability import temporal_evidence
from component4_profiling_xai.src.behavioral_profile import BUNDLE
from component4_profiling_xai.src.behavioral_xai_runtime import ExplanationRequest
from ui import services


@pytest.fixture(scope="module")
def cases():
    bundle = joblib.load(BUNDLE)
    data = pd.read_csv(benchmark.ROOT / "data" / "behavioral_daily.csv")
    return bundle, dict(benchmark.examples(data, bundle))


@pytest.mark.parametrize("name", ["native_0", "native_1", "native_5", "controlled_stable",
                                  "controlled_gradual", "controlled_missing"])
def test_original_and_optimized_messages_and_evidence(name, cases, monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("XAI must execute offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    bundle, examples = cases
    rows = examples[name]
    prepared = benchmark.setup(rows, bundle)
    original = benchmark.functions(rows, bundle, *prepared)
    for stage, operation in original.items():
        optimized = benchmark.functions(rows, bundle, *prepared, ExplanationRequest(bundle))[stage]
        benchmark.compare(operation(), optimized(), stage)


def test_request_memo_hits_eviction_and_same_date_input_changes(cases):
    bundle, examples = cases
    rows = examples["native_5"]
    matrix, _, _ = benchmark.setup(rows, bundle)
    row = matrix.loc[[rows.index[-1]], bundle["features"]]
    request = ExplanationRequest(bundle)
    requested = request.explain_rows(row)[0]
    benchmark.compare(requested, request.explain_rows(row)[0])
    assert len(request._memo) == 1 and not ExplanationRequest(bundle)._memo
    revised = row.copy()
    revised["is_weekend"] = 1 - revised["is_weekend"]
    new = request.explain_rows(revised)[0]
    assert len(request._memo) == 2 and new["shap"] != requested["shap"]
    benchmark.compare(explain(bundle["pipeline"], revised, bundle["background"]), new)
    other = pd.concat([row] * (request.MAX_ROWS + 5), ignore_index=True)
    other["screen_incidents"] = np.arange(len(other))
    request.explain_rows(other)
    assert len(request._memo) == request.MAX_ROWS
    payload = sum(len(key) + grid.nbytes for key, (grid, _) in request._memo.items())
    assert payload <= request.MAX_ROWS * 8 * (27 + 8 * 24)


def test_temporal_chronology_child_and_monitoring_isolation(cases):
    bundle, _ = cases
    all_rows = services.behavioral_daily_source()["daily"]
    rows = all_rows.loc[all_rows["child_id"].isin(all_rows["child_id"].unique()[:2])].reset_index(drop=True)
    child = rows["child_id"].iloc[0]
    day = rows.loc[rows["child_id"].eq(child), "date"].iloc[33]
    matrix, _, history = benchmark.setup(rows, bundle)
    scope = rows["child_id"].eq(child) & rows["date"].le(day)
    expected = ExplanationRequest(bundle).temporal(history, matrix, child, day)
    benchmark.compare(expected, ExplanationRequest(bundle).temporal(history.loc[scope], matrix.loc[scope], child, day))
    changed = rows.copy()
    changed.loc[~scope, "screen_incidents"] += 400
    altered, _, scores = benchmark.setup(changed, bundle)
    benchmark.compare(expected, ExplanationRequest(bundle).temporal(scores, altered, child, day))
    index = rows.index[scope][-1]
    changed.loc[index, "language_monitor_observed"] = 0
    altered, _, scores = benchmark.setup(changed, bundle)
    result, _ = ExplanationRequest(bundle).temporal(scores, altered, child, day)
    assert result.status == "insufficient"


def test_invalid_recorded_score_is_not_hidden_by_coalition_reuse(cases):
    bundle, examples = cases
    rows = examples["native_5"]
    matrix, _, history = benchmark.setup(rows, bundle)
    index = rows.index[-1]
    child, day = rows.loc[index, ["child_id", "date"]]
    request = ExplanationRequest(bundle)
    request.temporal(history, matrix, child, day)
    corrupted = history.copy()
    corrupted.loc[index, "score"] += 0.1
    expected = temporal_evidence(corrupted, matrix, bundle, child, day)
    assert expected[0].status == "unsupported"
    benchmark.compare(expected, request.temporal(corrupted, matrix, child, day))


def test_sustained_temporal_change_preserves_detector_annotations(cases):
    bundle, _ = cases
    data = pd.read_csv(benchmark.ROOT / "data" / "behavioral_daily.csv")
    normals = data.loc[data["split"].eq("train") & data["is_anomaly"].eq(0)]
    first = normals.loc[normals["child_id"].eq(normals["child_id"].iloc[0])].iloc[-1]
    rows = controlled(first, pd.Timestamp("2026-09-11"), "sustained", "screen")
    rows["date"] = rows["date"].dt.date
    prepared = benchmark.setup(rows, bundle)
    original = benchmark.functions(rows, bundle, *prepared)["temporal_review"]()
    optimized = benchmark.functions(rows, bundle, *prepared, ExplanationRequest(bundle))["temporal_review"]()
    assert original[0].status == "sustained" and original[0].detectors
    benchmark.compare(original, optimized)

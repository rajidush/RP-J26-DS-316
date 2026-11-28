"""Reproduce pilot with python -m component4_profiling_xai.evaluate_behavioral_shap OUTPUT."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
import platform
from threading import Event, Thread
from time import perf_counter

import numpy as np
import pandas as pd
import psutil

from .src.behavioral_anomaly import (
    ARTIFACT_DIR, METADATA_FILE, PIPELINE_FILE, check_features, feature_matrix, score_observed_days,
)
from .src.behavioral_shap import (
    BehavioralShap, format_parent_explanation, normal_training_rows,
    ranked, select_background, top_overlap, verified_groups,
)

DATA = Path(__file__).resolve().parent / "data" / "behavioral_daily.csv"
CONFIGS = [(32, 17, 256), (32, 29, 256), (64, 17, 256), (64, 29, 256), (32, 17, 512)]
METHODS = ("kernel", "group_aware")


def scored_test_rows(daily, explainer):
    test = daily.loc[daily["split"].eq("test")].copy()
    features = feature_matrix(test, explainer.metadata["features"])
    scores, observed = score_observed_days(explainer.pipeline, test, features)
    test = test.assign(score=scores).loc[observed].reset_index(drop=True)
    test["margin"] = test["score"] - explainer.metadata["threshold"]
    test["outcome"] = np.select(
        [test["margin"].gt(0) & test["is_anomaly"].eq(1), test["margin"].gt(0),
         test["is_anomaly"].eq(1)], ["TP", "FP", "FN"], default="TN")
    return test.assign(absolute_margin=test["margin"].abs())


def pilot_rows(test):
    nearest = test.sort_values(["absolute_margin", "child_id", "date"]).head(2)
    yield from ((index, "near_threshold") for index in nearest.index)
    rng = np.random.default_rng(42)
    for outcome in ("TP", "FP", "FN", "TN"):
        pool = test.loc[test.outcome.eq(outcome)].drop(index=nearest.index, errors="ignore")
        yield from ((int(index), outcome) for index in rng.choice(pool.index, size=2, replace=False))


def benchmark(explain):
    """Sample process-wide resident memory every 1 ms, including native allocations."""
    process = psutil.Process()
    start_rss = process.memory_info().rss
    peak = start_rss
    done = Event()

    def sample():
        nonlocal peak
        while not done.wait(0.001):
            peak = max(peak, process.memory_info().rss)

    worker = Thread(target=sample)
    worker.start()
    start = perf_counter()
    try:
        result = explain()
        seconds = perf_counter() - start
        peak = max(peak, process.memory_info().rss)
        return result, {
            "seconds": seconds, "rss_baseline_bytes": start_rss, "rss_peak_bytes": peak,
            "rss_peak_increase_bytes": peak - start_rss,
        }
    finally:
        done.set()
        worker.join()


def compare(records):
    output = []
    for (left_config, left), (right_config, right) in combinations(records, 2):
        shared = set(left.group_values) & set(right.group_values)
        output.append({
            "left": left_config, "right": right_config,
            "feature_top3_jaccard": (top_overlap(left.feature_values, right.feature_values)
                                     if left.feature_values is not None else None),
            "group_top2_jaccard": top_overlap(left.group_values, right.group_values, 2),
            "group_sign_agreement": sum(
                np.sign(left.group_values[g]) == np.sign(right.group_values[g]) for g in shared
            ) / len(shared),
            "expected_score_difference": abs(left.expected_score - right.expected_score),
        })
    return output


def fidelity(values, deletion):
    shared = {g: delta for g, delta in values.items() if g in deletion}
    top = ranked(shared, 1)[0]
    observed = ranked(deletion, 1)[0]
    rho = pd.Series(shared).abs().corr(pd.Series(deletion).abs(), method="spearman") if (
        len(set(map(abs, shared.values()))) > 1 and len(set(map(abs, deletion.values()))) > 1
    ) else None
    return {
        "top_group": top, "largest_deletion_group": observed, "top_group_agreement": top == observed,
        "top_sign_agreement": bool(np.sign(shared[top]) == np.sign(deletion[top])),
        "absolute_rank_spearman": rho,
    }


def evaluate():
    import shap  # Warm dependency imports before measuring per-explanation costs.

    daily = pd.read_csv(DATA)
    explainer = BehavioralShap()
    paths = (DATA, ARTIFACT_DIR / PIPELINE_FILE, ARTIFACT_DIR / METADATA_FILE)
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in paths}
    if hashes[DATA.name] != explainer.metadata["dataset_sha256"]:
        raise ValueError("dataset differs from the frozen training source")
    background_keys = {(size, seed) for size, seed, _ in CONFIGS}
    backgrounds = {key: select_background(daily, *key) for key in sorted(background_keys)}
    used = {key for background in backgrounds.values() for key in background.identifiers}
    keys = pd.MultiIndex.from_frame(daily[["child_id", "date"]])
    donor = select_background(daily.loc[~keys.isin(used)], 64, 101)
    train = normal_training_rows(daily)
    correlations = check_features(train, explainer.metadata["features"], corr_limit=0.85)
    test = scored_test_rows(daily, explainer)
    results = []
    for index, stratum in pilot_rows(test):
        row = test.iloc[[index]]
        item = row[["child_id", "date", "is_anomaly", "score"]].iloc[0].to_dict()
        item.update(stratum=stratum, runs=[], failures=[], comparisons={})
        deletion = explainer.deletion_effects(row, donor)
        item["deletion_effects"] = deletion
        for method in METHODS:
            records = []
            configs = CONFIGS if method == "kernel" else CONFIGS[:4]
            for size, seed, budget in configs:
                key = f"{size}/{seed}/{budget}" if method == "kernel" else f"{size}/{seed}/exact"
                try:
                    result, timing = benchmark(lambda: explainer.explain(
                        row, backgrounds[size, seed], method=method, nsamples=budget, seed=137,
                    ))
                    if result is None or result.anomaly_score != item["score"]:
                        raise ValueError("score differs from the existing prediction path")
                    if not result.reconstructs:
                        item["failures"].append({"method": method, "config": key, "error": "reconstruction or threshold mismatch"})
                    item["runs"].append({"config": key, **asdict(result), **timing,
                                         "fidelity": fidelity(result.group_values, deletion)})
                    records.append((key, result))
                except Exception as error:
                    item["failures"].append({"method": method, "config": key, "error": repr(error)})
            item["comparisons"][method] = compare(records)
            if method == "group_aware":
                evidence = [r for _, r in records]
                if len(records) != len(configs):
                    evidence = [records[0][1], None] if records else []
                item["verified_groups"] = verified_groups(evidence, deletion)
                item["parent_text"] = format_parent_explanation(evidence, deletion)
        results.append(item)
    return {
        "versions": {name: version(name) for name in ("shap", "numpy", "pandas", "scikit-learn", "psutil")},
        "platform": platform.platform(), "processor": platform.processor(),
        "python": platform.python_version(), "shap_version": shap.__version__,
        "hashes_before": hashes,
        "hashes_after": {p.name: sha256(p.read_bytes()).hexdigest() for p in paths},
        "training_rows": len(train), "training_children": train["child_id"].nunique(),
        "high_correlations": correlations["high_corr"].to_dict("records"),
        "training_constants": correlations["constant"], "training_duplicates": correlations["duplicates"],
        "perturbation_background": donor.metadata(), "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path")
    results = evaluate()
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()

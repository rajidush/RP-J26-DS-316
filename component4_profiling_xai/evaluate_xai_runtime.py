"""Reproducible original-versus-exact XAI benchmark for Step 06."""
from dataclasses import asdict
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
import tracemalloc
from unittest.mock import patch

import joblib
import matplotlib
import numpy as np
import pandas as pd

from .evaluate_adaptive_profile import scenario_rows
from .evaluate_anomaly_optimization import digest, serializable
from .src.behavioral_explain import explain, SCORE_ATOL
from .src.behavioral_explanation_reliability import assess_daily, assess_temporal, daily_evidence, temporal_evidence
from .src.behavioral_profile import BUNDLE, ProfileConfig, profiles, score_history
from .src.behavioral_synthetic import build_dataset
from .src.urgent_alerts import urgent_alerts
from ui.pages._behavioral_upgrade import explanation_text, guidance, historical_comparison, recent_behavior, temporal_text

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "xai_runtime_optimization_step06"
SEED = 2026112633
REPETITIONS = 5
LOCK_SOURCES = ("behavioral_explain.py", "behavioral_explanation_reliability.py",
                "behavioral_temporal_explain.py", "behavioral_profile.py",
                "behavioral_adaptive_profile.py", "behavioral_anomaly.py", "urgent_alerts.py")


def save(name, value):
    (OUT / name).write_text(json.dumps(serializable(value), allow_nan=False, indent=2) + "\n", encoding="utf-8")


def examples(daily, bundle):
    daily = daily.assign(date=pd.to_datetime(daily["date"]).dt.date)
    children = list(daily.groupby("child_id", sort=True))
    for offset, (child, rows) in enumerate(children[:8]):
        rows = rows.reset_index(drop=True)
        matrix, _ = profiles(rows)
        scored = score_history(rows, matrix, bundle)
        if offset == 0:
            end = 8
        elif offset in (1, 2, 3) and scored["flag"].fillna(False).iloc[24:].any():
            end = scored.index[scored["flag"].fillna(False) & (scored.index >= 24)][-1]
        elif offset in (4, 5) and scored["flag"].eq(False).fillna(False).iloc[24:].any():
            end = scored.index[scored["flag"].eq(False).fillna(False) & (scored.index >= 24)][-1]
        else:
            end = len(rows) - 1
        yield f"native_{offset}", rows.loc[:end]
    anchor = children[0][1].iloc[0]
    for style in ("stable", "gradual", "missing"):
        rows, _ = scenario_rows(anchor, style, pd.Timestamp(daily["date"].max()), 1, 4)
        rows["date"] = rows["date"].dt.date
        if style == "missing":
            rows.loc[rows.index[-1], "language_monitor_observed"] = False
        yield "controlled_" + style, rows


def setup(rows, bundle):
    config = ProfileConfig(**bundle["profile"])
    matrix, details = profiles(rows, config)
    return matrix, details, score_history(rows, matrix, bundle)


def functions(rows, bundle, matrix, details, history, engine=None):
    index = rows.index[-1]
    child, day = rows.loc[index, ["child_id", "date"]]
    row = matrix.loc[[index], bundle["features"]]
    score = history.at[index, "score"]
    flag = history.at[index, "flag"]
    flagged = bool(flag) if pd.notna(flag) else False

    def daily_only():
        if not np.isfinite(score):
            return None
        return engine.explain_rows(row)[0] if engine else explain(bundle["pipeline"], row, bundle["background"])

    def daily_review():
        if engine:
            result, variants = engine.daily(row, score)
            review = engine.review_daily(result, score, flagged, variants)
        else:
            result, variants = daily_evidence(row, score, bundle)
            review = assess_daily(result, score, flagged, bundle, variants)
        return result, variants, review

    def temporal_review():
        result, variants = (engine.temporal(history, matrix, child, day) if engine else
                            temporal_evidence(history, matrix, bundle, child, day))
        return result, variants, assess_temporal(result, variants)

    def temporal_only():
        if engine:
            return engine.temporal(history, matrix, child, day, review=False)[0]
        from .src.behavioral_temporal_explain import explain_temporal
        return explain_temporal(history, matrix, bundle, child, day)

    def messages():
        temporal, _, tr = temporal_review()
        result, _, dr = daily_review()
        overview, active, earlier = recent_behavior(history, day, bundle["profile"]["minimum"])
        urgency = urgent_alerts(rows).loc[:, ["is_urgent", "reasons"]]
        return (flagged, score, overview, active, earlier, temporal_text(temporal, active, tr),
                explanation_text(result, flagged, dr if np.isfinite(score) else None),
                guidance(matrix.loc[index], dr, active, bool(urgency.at[index, "is_urgent"])),
                historical_comparison(rows, matrix, details, history["flag"], index, day,
                                      ProfileConfig(**bundle["profile"])),
                tuple(urgency.itertuples(index=False, name=None)))

    return {"daily": daily_only, "daily_review": daily_review, "temporal": temporal_only,
            "temporal_review": temporal_review, "messages": messages}


def engine_for(bundle, arm):
    if arm == "optimized":
        from .src.behavioral_xai_runtime import ExplanationRequest
        return ExplanationRequest(bundle)
    return None


def operation(stage, arm, rows, bundle, prepared):
    if stage == "load_model":
        return joblib.load(BUNDLE)
    if stage == "prepare_profiles":
        return profiles(rows, ProfileConfig(**bundle["profile"]))
    if stage == "score_history":
        return score_history(rows, prepared[0], bundle)
    if stage == "adaptive_comparison":
        matrix, details, history = prepared
        return historical_comparison(rows, matrix, details, history["flag"], rows.index[-1],
                                     rows["date"].iloc[-1], ProfileConfig(**bundle["profile"]))
    if stage == "end_to_end":
        prepared = setup(rows, bundle)
        stage = "messages"
    return functions(rows, bundle, *prepared, engine_for(bundle, arm))[stage]()


def seconds(function):
    start = perf_counter()
    value = function()
    return (perf_counter() - start) * 1000, value


def benchmark(data, arms):
    cold = {}
    records = []
    exemplar = list(examples(data, joblib.load(BUNDLE)))
    for arm in arms:
        def initial():
            bundle = joblib.load(BUNDLE)
            return operation("end_to_end", arm, exemplar[1][1], bundle, None)
        cold[arm] = seconds(initial)[0]
    for case, rows in exemplar:
        for arm in arms:
            bundle = joblib.load(BUNDLE)
            prepared = setup(rows, bundle)
            stages = ("daily", "daily_review", "temporal", "temporal_review", "end_to_end",
                      "prepare_profiles", "score_history", "adaptive_comparison", "load_model")
            if case != exemplar[0][0]:
                stages = stages[:-1]
            if arm == "optimized":
                stages += ("same_request_memo",)
            for stage in stages:
                if stage == "same_request_memo":
                    run = functions(rows, bundle, *prepared, engine_for(bundle, arm))["messages"]
                    run()
                else:
                    run = lambda: operation(stage, arm, rows, bundle, prepared)
                times = [seconds(run)[0] for _ in range(REPETITIONS)]
                # Instrument separately from stopwatch repetitions.
                pipeline = bundle["pipeline"]
                with patch.object(pipeline, "score_samples", wraps=pipeline.score_samples) as score:
                    run()
                    calls = score.call_count
                    prediction_rows = sum(len(call.args[0]) for call in score.call_args_list)
                tracemalloc.start()
                run()
                _, traced = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                records.append(dict(case=case, rows=len(rows), arm=arm, stage=stage, first_ms=times[0],
                                    warm_median_ms=np.median(times[1:]), p95_ms=np.quantile(times, .95),
                                    max_ms=max(times), model_calls=calls, model_rows=prediction_rows,
                                    peak_bytes=traced, samples_ms=json.dumps(times)))
    try:
        import psutil
        memory = psutil.Process().memory_info()
        process = {"rss_bytes": memory.rss, "peak_working_set_bytes": getattr(memory, "peak_wset", None)}
    except ImportError:
        process = None
    return pd.DataFrame(records), {"load_plus_first_request_ms": cold, "process_after_experiment": process,
                                   "repetitions": REPETITIONS, "artifact_bytes": BUNDLE.stat().st_size,
                                   "platform": platform.platform(), "python": platform.python_version()}


def compare(original, optimized, path="result"):
    if hasattr(original, "__dataclass_fields__"):
        original, optimized = asdict(original), asdict(optimized)
    if isinstance(original, dict):
        assert original.keys() == optimized.keys(), path
        for key in original:
            if key not in {"shap_ms", "lime_ms"}:
                compare(original[key], optimized[key], f"{path}.{key}")
    elif isinstance(original, (tuple, list)):
        assert len(original) == len(optimized), path
        for index, (left, right) in enumerate(zip(original, optimized)):
            compare(left, right, f"{path}[{index}]")
    elif isinstance(original, (float, np.floating)):
        assert isinstance(optimized, (float, np.floating)) and np.isclose(
            original, optimized, atol=SCORE_ATOL, rtol=0, equal_nan=True), (path, original, optimized)
    elif original is pd.NA:
        assert optimized is pd.NA, path
    else:
        assert original == optimized, (path, original, optimized)


def aggregate(measurements):
    output = []
    for (stage, arm), group in measurements.groupby(["stage", "arm"]):
        samples = [json.loads(row) for row in group["samples_ms"]]
        pooled = np.concatenate(samples)
        output.append(dict(stage=stage, arm=arm, first_ms=group["first_ms"].median(),
                           warm_median_ms=np.median(np.concatenate([sample[1:] for sample in samples])),
                           p95_ms=np.quantile(pooled, .95), max_ms=max(pooled),
                           model_calls=group["model_calls"].median(), model_rows=group["model_rows"].median(),
                           peak_bytes=group["peak_bytes"].median(), max_peak_bytes=group["peak_bytes"].max()))
    return pd.DataFrame(output)


def final(data):
    bundle = joblib.load(BUNDLE)
    results = []
    for case, rows in examples(data, bundle):
        prepared = setup(rows, bundle)
        left = functions(rows, bundle, *prepared)
        failures = []
        for stage in left:
            try:
                request = engine_for(bundle, "optimized")
                right = functions(rows, bundle, *prepared, request)
                compare(left[stage](), right[stage](), stage)
            except AssertionError as exc:
                failures.append(str(exc))
        data_result, _, daily_review = left["daily_review"]()
        temporal, _, review = left["temporal_review"]()
        results.append(dict(case=case, rows=len(rows), equivalent=not failures, discrepancies=failures,
                            daily=daily_review.status, temporal=temporal.status, review=review.status,
                            scored=data_result is not None, flagged=str(prepared[2]["flag"].iloc[-1]),
                            memo_entries=len(request._memo),
                            memo_payload_bytes=sum(len(key) + score_grid.nbytes
                                                   for key, (score_grid, _) in request._memo.items())))
    save("equivalence.json", results)
    if not all(row["equivalent"] for row in results):
        raise RuntimeError("An equivalence gate failed; inspect equivalence.json")
    measurements, environment = benchmark(data, ("original", "optimized"))
    environment["synthetic_data_sha256"] = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()
    measurements.to_csv(OUT / "holdout_runtime.csv", index=False)
    summary = aggregate(measurements)
    summary.to_csv(OUT / "holdout_summary.csv", index=False)
    save("holdout_environment.json", environment)
    per_stage = summary.set_index(["stage", "arm"])
    latency = {stage: float(per_stage.at[(stage, "optimized"), "warm_median_ms"]
                            / per_stage.at[(stage, "original"), "warm_median_ms"])
               for stage in ("temporal_review", "end_to_end")}
    memory = {stage: float(per_stage.at[(stage, "optimized"), "peak_bytes"]
                           / per_stage.at[(stage, "original"), "peak_bytes"])
              for stage in ("temporal", "temporal_review")}
    accepted = max(latency.values()) < 1 and max(memory.values()) <= 1.5
    save("acceptance.json", {"equivalent_cases": len(results), "warm_latency_ratios": latency,
                             "median_traced_peak_ratios": memory, "accepted": accepted})
    plot = summary.pivot(index="stage", columns="arm", values="warm_median_ms").loc[
        ["daily", "daily_review", "temporal", "temporal_review", "end_to_end"]]
    axis = plot.plot.barh(figsize=(8, 4), color=["#3c937e", "#285c8d"])
    axis.set(xlabel="Median milliseconds per request", title="Step 06 · same cases, fresh request state")
    axis.figure.tight_layout()
    axis.figure.savefig(OUT / "runtime_comparison.png", dpi=160)
    plt.close(axis.figure)
    print(f"{len(results)} equivalent cases; acceptance passed={accepted}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baseline", "lock", "holdout"))
    stage = parser.parse_args().stage
    OUT.mkdir(exist_ok=True)
    if stage == "baseline":
        rows = pd.read_csv(ROOT / "data" / "behavioral_daily.csv")
        table, environment = benchmark(rows, ("original",))
        table.to_csv(OUT / "development_baseline.csv", index=False)
        save("development_environment.json", environment)
    elif stage == "lock":
        save("selection_lock.json", {
            "seed": SEED, "model_sha256": digest(BUNDLE), "benchmark_sha256": digest(Path(__file__)),
            "optimizer_sha256": digest(ROOT / "src" / "behavioral_xai_runtime.py"),
            "baseline_sha256": digest(OUT / "development_baseline.csv"),
            "source_sha256": {name: digest(ROOT / "src" / name) for name in LOCK_SOURCES},
            "acceptance": "All cases exact categorical/message parity and numeric 1e-8 absolute parity; "
                          "both temporal review and end-to-end warm medians lower; "
                          "stage median peak traced memory no more than 1.50 times original on either temporal path.",
            "target": "Temporal review under 500 ms on desktop, exploratory. No cross-request cache.",
        })
    else:
        lock = json.loads((OUT / "selection_lock.json").read_text())
        for path, key in ((BUNDLE, "model_sha256"), (Path(__file__), "benchmark_sha256"),
                          (ROOT / "src" / "behavioral_xai_runtime.py", "optimizer_sha256"),
                          (OUT / "development_baseline.csv", "baseline_sha256")):
            if digest(path) != lock[key]:
                raise RuntimeError(f"Lock mismatch for {path.name}")
        for name in LOCK_SOURCES:
            if digest(ROOT / "src" / name) != lock["source_sha256"][name]:
                raise RuntimeError(f"Lock mismatch for {name}")
        final(build_dataset(lock["seed"]).daily)


if __name__ == "__main__":
    main()

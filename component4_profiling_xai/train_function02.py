"""Train and evaluate the Function 02 framework on the existing synthetic cohort."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter, process_time

import joblib
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import psutil
import sklearn

from .src.behavioral_anomaly import FEATURE_SET_A, anomaly_scores, assessable_days, build_pipeline
from .src.behavioral_anomaly import check_features, evaluate, load_artifacts, tune_threshold
from .src.behavioral_explain import explain
from .src.behavioral_profile import FUSED, PERSONAL, TEMPORAL, ProfileConfig, TrendConfig
from .src.behavioral_profile import BUNDLE, profiles, score_history

ROOT = Path(__file__).resolve().parent
OUTPUT = BUNDLE.parent

PARAMS = {"isolation_forest": {"n_estimators": 100, "n_jobs": 1},
          "one_class_svm": {"nu": 0.01, "gamma": 0.001}}


def metrics(labels, scores, threshold):
    result = evaluate(labels, scores, threshold)
    negatives = result["fp"] + result["tn"]
    return {**result, "fpr": result["fp"] / negatives if negatives else None}


def latency(pipeline, frame):
    anomaly_scores(pipeline, frame.iloc[[0]])
    times = []
    for i in range(min(60, len(frame))):
        start = perf_counter()
        anomaly_scores(pipeline, frame.iloc[[i]])
        times.append((perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.quantile(times, 0.95))}


def temporal_metrics(history, daily, episodes, split, method):
    selected = history[daily.loc[history.index, "split"].eq(split)]
    normal = selected["score"].notna() & daily.loc[selected.index, "is_anomaly"].eq(0)
    points = selected[selected[f"{method}_alert"]]
    delays = []
    cases = episodes[episodes["split"].eq(split)]
    for episode in cases.itertuples():
        matches = points[points["child_id"].eq(episode.child_id)
                         & points["date"].between(episode.start_date, episode.end_date)]
        if not matches.empty:
            delays.append((pd.Timestamp(matches["date"].min()) - pd.Timestamp(episode.start_date)).days)
    return {"episodes": len(cases), "detected": len(delays),
            "detection_rate": len(delays) / len(cases),
            "delay_days_detected_only": float(np.mean(delays)) if delays else None,
            "false_alert_rate": float(selected.loc[normal, f"{method}_alert"].mean()),
            "normal_scored_days": int(normal.sum())}


def explanation_metrics(pipeline, rows, background):
    local = []
    for _, row in rows.iterrows():
        row = row.to_frame().T
        start = perf_counter()
        result = explain(pipeline, row, background)
        elapsed = (perf_counter() - start) * 1000
        alternate = explain(pipeline, row, background.sample(frac=0.75, random_state=23), seed=23)
        actual = result["score"]
        local.append({"joint_ms": elapsed, "shap_ms": result["shap_ms"], "lime_ms": result["lime_ms"], "lime_heldout_weighted_r2": result["lime_r2"],
                      "shap_additivity_error": abs(actual - result["reference"] - sum(result["shap"].values())),
                      "shap_top_group_agreement": top(result["shap"]) == top(alternate["shap"]),
                      "lime_top_group_agreement": top(result["lime"]) == top(alternate["lime"]),
                      "shap_top_group_deletion": result["deletion"][top(result["shap"])],
                      "lime_top_group_deletion": result["deletion"][top(result["lime"])],
                      **{f"shap: {k}": v for k, v in result["shap"].items()}})
    table = pd.DataFrame(local)
    return table, table.filter(like="shap: ").abs().mean().to_dict()


def top(values):
    return max(values, key=lambda key: abs(values[key]))


def train(config):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data" / "behavioral_daily.csv"
    daily = pd.read_csv(path)
    episodes = pd.read_csv(ROOT / "data" / "behavioral_episodes.csv")
    matrix, _ = profiles(daily, config)
    observed = assessable_days(daily, matrix)
    masks = {split: observed & daily["split"].eq(split) for split in ("train", "validation", "test")}
    # Existing generator names its middle chronological partition "val".
    if not masks["validation"].any():
        daily["split"] = daily["split"].replace("val", "validation")
        episodes["split"] = episodes["split"].replace("val", "validation")
        masks["validation"] = observed & daily["split"].eq("validation")
    train, val, test = (masks[key] for key in ("train", "validation", "test"))
    assert daily.loc[train, "date"].max() < daily.loc[val, "date"].min()
    assert daily.loc[val, "date"].max() < daily.loc[test, "date"].min()
    normal = train & daily["is_anomaly"].eq(0)
    designs = {
        "global": FEATURE_SET_A,
        "personalized": FUSED + PERSONAL + TEMPORAL,
        "screen_only": [f for f in FUSED + PERSONAL + TEMPORAL if f.startswith("screen_")],
        "language_only": [f for f in FUSED + PERSONAL + TEMPORAL if f.startswith("language_")],
        "no_personalization": FUSED + TEMPORAL,
        "no_temporal": FUSED + PERSONAL,
        "combined_raw": FUSED,
    }
    results, candidates = [], []
    for design, features in designs.items():
        for algorithm, params in PARAMS.items():
            pipeline = build_pipeline(algorithm, params).fit(matrix.loc[normal, features])
            train_scores = anomaly_scores(pipeline, matrix.loc[normal, features])
            tuned, _ = tune_threshold(train_scores, anomaly_scores(pipeline, matrix.loc[val, features]),
                                      daily.loc[val, "is_anomaly"], [0.90, 0.95, 0.97, 0.98, 0.99, 0.995])
            bundle = {"pipeline": pipeline, "features": features, "threshold": tuned["threshold"],
                      "center": float(np.median(train_scores)),
                      "scale": float(np.std(train_scores)), "algorithm": algorithm, "design": design}
            candidates.append((bundle, tuned))
    selected, _ = max((item for item in candidates if item[0]["design"] in {"global", "personalized"}),
                      key=lambda item: (item[1]["f1"], -item[1]["fp"]))
    selected["trend"] = vars(TrendConfig())
    selected["profile"] = vars(config)
    development = daily["split"].ne("test")
    trend_candidates = []
    for threshold in (1, 2, 3, 4, 6, 8):
        selected["trend"] = vars(TrendConfig(threshold=threshold))
        history = score_history(daily.loc[development], matrix.loc[development], selected)
        for method in ("ewma", "cusum"):
            result = temporal_metrics(history, daily, episodes, "validation", method)
            trend_candidates.append((method, threshold, result))
    chosen = {}
    for method in ("ewma", "cusum"):
        _, threshold, result = max((item for item in trend_candidates if item[0] == method),
                                   key=lambda item: (item[2]["detection_rate"] - 5 * item[2]["false_alert_rate"],
                                                     -item[2]["false_alert_rate"]))
        chosen[method] = {"threshold": threshold, "validation": result}
    selected["trend"] = {**vars(TrendConfig()), "threshold": chosen["ewma"]["threshold"],
                         "cusum_threshold": chosen["cusum"]["threshold"]}
    selected["background"] = matrix.loc[normal, selected["features"]].sample(n=24, random_state=42)
    selected["sklearn_version"] = sklearn.__version__
    selected["dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    validation_history = score_history(daily.loc[development], matrix.loc[development], selected).loc[val]
    sustained = validation_history["ewma_active"] | validation_history["cusum_active"]
    priority_candidates = {}
    for count in (1, 2, 3, 4):
        rule = sustained | (validation_history["flag"] & validation_history["prior_flagged"].ge(count))
        priority_candidates[count] = metrics(daily.loc[val, "is_anomaly"], rule.astype(float), 0.5)
    selected["priority_repeated"] = max(priority_candidates, key=lambda count: (
        priority_candidates[count]["f1"], -priority_candidates[count]["fp"], count))
    with NamedTemporaryFile(dir=OUTPUT, suffix=".joblib", delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        joblib.dump(selected, pending, compress=3)
        pending.replace(BUNDLE)
    finally:
        pending.unlink(missing_ok=True)
    joblib.dump({f"{b['design']}/{b['algorithm']}": {
        **b, "profile": vars(config), "validation": tuning,
        "sklearn_version": sklearn.__version__, "dataset_sha256": selected["dataset_sha256"],
    } for b, tuning in candidates}, OUTPUT / "candidate_models.joblib", compress=3)
    for bundle, tuned in candidates:
        frame = matrix.loc[test, bundle["features"]]
        results.append({"design": bundle["design"], "algorithm": bundle["algorithm"],
                        "validation_f1": tuned["f1"], **metrics(daily.loc[test, "is_anomaly"],
                            anomaly_scores(bundle["pipeline"], frame), bundle["threshold"]),
                        **latency(bundle["pipeline"], frame)})
    legacy, metadata = load_artifacts()
    frame = matrix.loc[test, metadata["features"]]
    results.append({"design": "frozen_legacy", "algorithm": metadata["algorithm"],
                    **metrics(daily.loc[test, "is_anomaly"], anomaly_scores(legacy, frame),
                              metadata["threshold"]), **latency(legacy, frame)})
    history = score_history(daily, matrix, selected)
    for method, record in chosen.items():
        record["test"] = temporal_metrics(history, daily, episodes, "test", method)
    flagged = test & history["flag"].fillna(False)
    pool = matrix.loc[flagged, selected["features"]]
    sample = pool.sample(n=min(30, len(pool)), random_state=42) if len(pool) else matrix.loc[test, selected["features"]].head(1)
    process = psutil.Process()
    before = process.memory_info().rss
    wall, cpu = perf_counter(), process_time()
    explanations, global_importance = explanation_metrics(selected["pipeline"], sample, selected["background"])
    elapsed, cpu = perf_counter() - wall, process_time() - cpu
    resource = {"platform": platform.platform(), "python": platform.python_version(),
                "rss_before_bytes": before, "rss_after_bytes": process.memory_info().rss,
                "xai_cpu_seconds": cpu, "xai_cpu_percent_one_core": 100 * cpu / elapsed,
                "bundle_bytes": BUNDLE.stat().st_size}
    summary = {"source": "Existing synthetic cohort, previously used for research",
               "dataset_sha256": selected["dataset_sha256"], "selected": {
                   key: selected[key] for key in ("design", "algorithm", "features", "threshold",
                                                  "trend", "profile", "sklearn_version")},
               "splits": {key: {"scored_days": int(mask.sum()),
                                "first": daily.loc[mask, "date"].min(), "last": daily.loc[mask, "date"].max()}
                          for key, mask in masks.items()},
               "priority_validation": priority_candidates,
               "priority_repeated": selected["priority_repeated"],
               "temporal": chosen, "global_shap": global_importance, "resources": resource,
               "emotional_ablation": "Unavailable: no aligned labeled emotional observations",
               "xai_means": explanations.mean(numeric_only=False).to_dict()}
    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT / "model_comparison.csv", index=False)
    figure = Figure(figsize=(8, 4), layout="constrained")
    axis = figure.subplots()
    comparison[comparison["design"].isin(["global", "personalized"])].pivot(
        index="design", columns="algorithm", values="f1").plot.bar(ax=axis, rot=0)
    axis.set(title="Chronological synthetic test performance", ylabel="F1", ylim=(0, 1))
    figure.savefig(OUTPUT / "model_comparison.png", dpi=140)

    figure = Figure(figsize=(8, 4), layout="constrained")
    axis = figure.subplots()
    pd.Series({name.removeprefix("shap: "): value for name, value in global_importance.items()}).plot.barh(ax=axis)
    axis.set(title="Group attribution on sampled flagged test days", xlabel="Mean absolute SHAP")
    figure.savefig(OUTPUT / "global_shap.png", dpi=140)
    explanations.to_csv(OUTPUT / "xai_evaluation.csv", index=False)
    check_features(daily.loc[normal], FEATURE_SET_A)["high_corr"].to_csv(
        OUTPUT / "training_correlations.csv", index=False)
    (OUTPUT / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = ProfileConfig()
    parser.add_argument("--baseline-days", type=int, default=defaults.days)
    parser.add_argument("--frequency-days", type=int, default=defaults.frequency_days)
    parser.add_argument("--min-history", type=int, default=defaults.minimum)
    args = parser.parse_args()
    train(ProfileConfig(days=args.baseline_days, frequency_days=args.frequency_days, minimum=args.min_history))

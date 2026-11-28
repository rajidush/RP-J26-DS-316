"""Fit, lock and evaluate the Function 02 research challengers."""
import argparse
import hashlib
import json
from itertools import combinations, product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve

from .src.behavioral_anomaly import assessable_days, anomaly_scores
from .src.behavioral_explain import explain
from .src.behavioral_optimization import HOLDOUT_SEED, candidates, development, matrices, selection_key
from .src.behavioral_profile import ANCHORS, BUNDLE, PERSONAL, TEMPORAL, TrendConfig, temporal
from .src.behavioral_synthetic import build_dataset
from .train_function02 import latency, metrics

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "function02_optimization"
LOCK = OUTPUT / "selection_lock.json"
ARTIFACT = OUTPUT / "research_candidates.joblib"
PRIMARY = ("global", "full_personalized", "selected_personalized", "hybrid")
DAILY = ROOT / "data" / "behavioral_daily.csv"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def scores_for(daily, frames, result, model):
    matrix = frames[result["window"]]
    covered = assessable_days(daily, matrix)
    score = pd.Series(np.nan, index=daily.index)
    score.loc[covered] = anomaly_scores(model, matrix.loc[covered, result["features"]])
    return score


def trend_history(daily, score, threshold, center, scale, config):
    result = []
    for _, rows in daily.groupby("child_id", sort=False):
        rows = rows.sort_values("date")
        values = score.loc[rows.index]
        flags = values.gt(threshold).where(values.notna())
        trend = temporal(rows["date"], values, flags, center, scale, config)
        trend.index = rows.index
        result.append(trend)
    return pd.concat(result).reindex(daily.index)


def temporal_outcomes(daily, mask, episodes, score, change):
    scoped = daily.loc[mask]
    hits = scoped.loc[change.loc[scoped.index] & scoped["is_anomaly"].eq(1)]
    first = hits.groupby("episode_id")["date"].min()
    result = episodes[["episode_id", "anomaly_type", "intensity", "start_date", "observed_days"]].copy()
    result["first_detection"] = result["episode_id"].map(first)
    result["delay_days"] = (pd.to_datetime(result["first_detection"]) -
                            pd.to_datetime(result["start_date"])).dt.days
    normal = mask & score.notna() & daily["is_anomaly"].eq(0)
    delays = result["delay_days"].dropna()
    return {"episodes": len(result), "detected": int(delays.size),
            "detection_rate": float(delays.size / len(result)),
            "delay_days_detected_only": float(delays.mean()) if delays.size else None,
            "false_alerts": int(change.loc[normal].sum()),
            "normal_scored_days": int(normal.sum()),
            "false_alert_rate": float(change.loc[normal].mean())}, result


def tune_trends(daily, score, valid, episodes, threshold, center, scale):
    grid = []
    for method, smoothing, cutoff in (
        [("ewma", a, t) for a, t in product((0.15, 0.30, 0.50, 0.70), (1.0, 2.0, 3.0, 4.0))]
        + [("cusum", d, t) for d, t in product((0.25, 0.50, 0.75), (2.0, 4.0, 6.0, 8.0))]
    ):
        config = TrendConfig(threshold=cutoff, alpha=smoothing) if method == "ewma" else TrendConfig(
            cusum_threshold=cutoff, drift=smoothing)
        history = trend_history(daily, score, threshold, center, scale, config)
        stats, _ = temporal_outcomes(daily, valid, episodes, score, history[f"{method}_alert"])
        grid.append({"method": method, "parameter": smoothing, "cutoff": cutoff, **stats})
    winners = {}
    for method in ("ewma", "cusum"):
        winners[method] = max((r for r in grid if r["method"] == method),
                              key=lambda r: (r["detection_rate"] - 5 * r["false_alert_rate"],
                                             -r["false_alert_rate"], r["cutoff"]))
    return winners, grid


def fit():
    if LOCK.exists():
        raise SystemExit("Selection is already locked. Use the existing observations or a new named protocol.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    daily = development(pd.read_csv(DAILY))
    episodes = pd.read_csv(ROOT / "data" / "behavioral_episodes.csv")
    episodes = episodes[episodes["split"].isin(["val", "validation"])]
    frames = matrices(daily)
    search, per_window, picked, normal, valid, correlations = candidates(daily, frames)
    chosen, model = min((picked[name] for name in PRIMARY), key=lambda pair: selection_key(pair[0]))
    score = scores_for(daily, frames, chosen, model)
    center, scale = float(score.loc[normal].median()), float(score.loc[normal].std(ddof=0))
    trends, trend_grid = tune_trends(daily, score, valid, episodes, chosen["threshold"], center, scale)
    legacy = joblib.load(BUNDLE)
    existing = {"id": "deployed", "design": "deployed", "algorithm": legacy["algorithm"],
                "window": legacy["profile"]["days"], "features": legacy["features"],
                "threshold": legacy["threshold"]}
    picked["deployed"] = (existing, legacy["pipeline"])
    validation = pd.DataFrame({name: scores_for(daily, frames, record, scorer)
                               for name, (record, scorer) in picked.items()}).loc[valid]
    validation = validation.assign(label=daily.loc[valid, "is_anomaly"])
    score_curves(validation).to_csv(OUTPUT / "validation_pr.csv", index=False)
    pd.DataFrame(search).to_csv(OUTPUT / "validation_search.csv", index=False)
    pd.DataFrame([record for record, _ in per_window.values()]).to_csv(
        OUTPUT / "window_and_ablation_validation.csv", index=False)
    correlations.to_csv(OUTPUT / "training_spearman.csv")
    train_frame = frames[chosen["window"]].loc[normal, chosen["features"]]
    audit = []
    for feature in ANCHORS:
        for split_mask, label, name in ((normal, 0, "fit_normal"), (valid, 0, "validation_normal"),
                                        (valid, 1, "validation_anomaly")):
            rows = split_mask & daily["is_anomaly"].eq(label)
            values = frames[chosen["window"]].loc[rows, [feature, f"{feature}__robust_z", f"{feature}__active_days"]]
            for field in values:
                sample = values[field]
                audit.append({"segment": name, "feature": field, "rows": len(sample),
                              "missing_rate": float(sample.isna().mean()),
                              "zero_rate": float(sample.eq(0).mean()),
                              "median": float(sample.median()),
                              "p90": float(sample.quantile(0.9))})
    pd.DataFrame(audit).to_csv(OUTPUT / "personal_feature_audit.csv", index=False)
    pd.DataFrame(trend_grid).to_csv(OUTPUT / "temporal_validation.csv", index=False)
    joblib.dump({"models": picked, "normal_reference": train_frame, "center": center,
                 "scale": scale}, ARTIFACT, compress=3)
    write_json(LOCK, {"seed": HOLDOUT_SEED, "artifact_sha256": digest(ARTIFACT),
                      "development_sha256": digest(DAILY), "selected": chosen, "trends": trends,
                      "selected_designs": {name: pair[0] for name, pair in picked.items()},
                      "fit_rows": int(normal.sum()), "validation_rows": int(valid.sum()),
                      "validation_positive": int(daily.loc[valid, "is_anomaly"].sum()),
                      "validation_start": daily.loc[valid, "date"].min(),
                      "validation_end": daily.loc[valid, "date"].max(),
                      "candidate_count": len(search), "protocol_sha256": digest(
                          ROOT / "experiments" / "FUNCTION02_OPTIMIZATION.md"),
                      "candidate_source_sha256": digest(ROOT / "src" / "behavioral_optimization.py"),
                      "evaluation_source_sha256": digest(Path(__file__))})
    print("Locked", chosen["id"], "validation F1", chosen["validation_f1"], flush=True)


def score_curves(frame):
    parts = []
    for name in frame.columns.drop("label"):
        p, r, threshold = precision_recall_curve(frame["label"], frame[name])
        parts.append(pd.DataFrame({"design": name, "precision": p, "recall": r,
                                   "inclusive_threshold": np.r_[threshold, np.nan]}))
    return pd.concat(parts, ignore_index=True)


def bootstrap(rows, name, global_name="global"):
    counts = []
    for _, child in rows.groupby("child_id"):
        y = child["label"].to_numpy(dtype=bool)
        cells = []
        for arm in (name, global_name):
            hit = child[arm].to_numpy(dtype=bool)
            cells.append([np.sum(hit & y), np.sum(hit & ~y), np.sum(~hit & y), np.sum(~hit & ~y)])
        counts.append(cells)
    counts = np.asarray(counts)
    rng = np.random.default_rng(132)
    a = counts[rng.integers(len(counts), size=(2000, len(counts)))].sum(axis=1)
    tp, fp, fn, tn = np.moveaxis(a, -1, 0)
    f1 = 2 * tp / (2 * tp + fp + fn)
    fpr = fp / (fp + tn)
    return {f"delta_{label}_95": np.quantile(metric[:, 0] - metric[:, 1], [0.025, 0.975]).tolist()
            for label, metric in (("f1", f1), ("fpr", fpr))}


def xai_check(model, rows, reference, scores):
    results = []
    backgrounds = [reference.sample(n=24, random_state=seed) for seed in (133, 176, 228, 314, 557)]
    for index, row in rows.iterrows():
        case = row.to_frame().T
        rounds = [explain(model, case, bg, seed=i + 42) for i, bg in enumerate(backgrounds)]
        base = rounds[0]
        independent_effect = rounds[-1]["deletion"]
        item = {"case": index, "score": scores.loc[index],
                "shap_additive_absolute_error": float(np.max([
                    abs(scores.loc[index] - entry["reference"] - sum(entry["shap"].values()))
                    for entry in rounds])),
                "lime_weighted_r2": float(np.mean([r["lime_r2"] for r in rounds])),
                "lime_original_score_absolute_error": float(np.mean([
                    abs(scores.loc[index] - (r["lime_intercept"] + sum(r["lime"].values())))
                    for r in rounds]))}
        for method in ("shap", "lime"):
            values = [r[method] for r in rounds]
            ranking = lambda v: max(v, key=lambda group: abs(v[group]))
            top = ranking(base[method])
            item[f"{method}_top_agreement"] = np.mean([ranking(a) == ranking(b)
                                                       for a, b in combinations(values, 2)])
            item[f"{method}_sign_agreement"] = np.mean([
                np.mean([np.sign(a[g]) == np.sign(b[g]) for g in a])
                for a, b in combinations(values, 2)])
            rank_agreement = [float(spearmanr(list(map(abs, a.values())), list(map(abs, b.values()))).statistic)
                              for a, b in combinations(values, 2)
                              if len(set(map(abs, a.values()))) > 1 and len(set(map(abs, b.values()))) > 1]
            item[f"{method}_rank_spearman"] = float(np.mean(rank_agreement)) if rank_agreement else None
            item[f"{method}_top_drop"] = independent_effect[top]
            item[f"{method}_signed_drop_advantage"] = (independent_effect[top] -
                                                        np.mean([effect for group, effect in independent_effect.items()
                                                                 if group != top]))
            item[f"{method}_absolute_drop_advantage"] = (abs(independent_effect[top]) -
                                                          np.mean([abs(effect) for group, effect in independent_effect.items()
                                                                   if group != top]))
            item[f"{method}_mean_ms"] = float(np.mean([r[f"{method}_ms"] for r in rounds]))
        results.append(item)
    return pd.DataFrame(results)


def error_report(daily, frame, score, flag, episodes, design):
    labeled = daily[["child_id", "date", "is_anomaly", "anomaly_type", "episode_id"]].join(
        episodes.set_index("episode_id")[["intensity"]], on="episode_id")
    labeled["error"] = np.select(
        [flag & labeled["is_anomaly"].eq(0), ~flag & labeled["is_anomaly"].eq(1)], ["FP", "FN"], default="")
    labeled = labeled.assign(score=score, design=design)
    context = frame[["screen_incidents", "screen_night_incidents", "language_flags",
                     "language_urgent_flags", *PERSONAL, *TEMPORAL]]
    return labeled.join(context)


def evaluate_holdout():
    if (OUTPUT / "evaluation.json").exists():
        raise SystemExit("The sealed evaluation has already been completed. Its results are saved.")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["seed"] != HOLDOUT_SEED or lock["artifact_sha256"] != digest(ARTIFACT):
        raise ValueError("The locked candidate artifact or seed changed")
    for key, path in (("development_sha256", DAILY), ("evaluation_source_sha256", Path(__file__)),
                      ("candidate_source_sha256", ROOT / "src" / "behavioral_optimization.py"),
                      ("protocol_sha256", ROOT / "experiments" / "FUNCTION02_OPTIMIZATION.md")):
        if lock[key] != digest(path):
            raise ValueError(f"Locked input changed: {path.name}")
    payload = joblib.load(ARTIFACT)
    holdout = OUTPUT / "sealed_holdout"
    daily_path, episode_path = holdout / "daily.csv", holdout / "episodes.csv"
    if not (holdout / "manifest.json").is_file():
        holdout.mkdir(exist_ok=True)
        fresh = build_dataset(HOLDOUT_SEED)
        fresh.daily.to_csv(daily_path, index=False)
        fresh.episodes.to_csv(episode_path, index=False)
        write_json(holdout / "manifest.json", {"seed": HOLDOUT_SEED, "daily_sha256": digest(daily_path),
                   "episodes_sha256": digest(episode_path), "lock_sha256": digest(LOCK)})
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["seed"], manifest["daily_sha256"], manifest["episodes_sha256"], manifest["lock_sha256"]) != (
        HOLDOUT_SEED, digest(daily_path), digest(episode_path), digest(LOCK)
    ):
        raise ValueError("Holdout manifest does not match the locked evaluation")
    daily = pd.read_csv(daily_path)
    episodes = pd.read_csv(episode_path)
    final = daily["day_index"].gt(56)
    episodes = episodes[episodes["start_date"].ge(daily.loc[final, "date"].min())]
    frames = matrices(daily)
    eligible = final & assessable_days(daily, frames[14])
    test = daily.loc[eligible]
    labels = test["is_anomaly"]
    results, errors, curves, binary = [], [], {}, {}
    selected = lock["selected"]["design"]
    for name, (setting, model) in payload["models"].items():
        frame = frames[setting["window"]].loc[eligible]
        score = anomaly_scores(model, frame[setting["features"]])
        flag = score > setting["threshold"]
        result = {"design": name, "algorithm": setting["algorithm"], "window": setting["window"],
                  "validation_f1": setting.get("validation_f1"), **metrics(labels, score, setting["threshold"]),
                  "average_precision": float(average_precision_score(labels, score)),
                  **latency(model, frame[setting["features"]])}
        results.append(result)
        curves[name], binary[name] = score, flag
        if name in {*PRIMARY, "deployed"}:
            errors.append(error_report(test, frame, score, flag, episodes, name))
    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT / "test_comparison.csv", index=False)
    comparison[["design", "tn", "fp", "fn", "tp"]].to_csv(OUTPUT / "confusion_matrices.csv", index=False)
    error_rows = pd.concat(errors)
    error_rows[error_rows["error"].ne("")].to_csv(OUTPUT / "error_days.csv", index=False)
    positive = error_rows[error_rows["is_anomaly"].eq(1)].assign(missed=lambda df: df["error"].eq("FN"))
    positive.groupby(["design", "anomaly_type", "intensity"], dropna=False)["missed"].agg(
        positives="size", false_negatives="sum").reset_index().to_csv(OUTPUT / "false_negatives_by_scenario.csv", index=False)
    normal = error_rows[error_rows["is_anomaly"].eq(0)]
    normal.groupby(["design", "error"])[["screen_incidents", "screen_night_incidents", "language_flags",
                                        "language_urgent_flags", *PERSONAL]].mean().to_csv(
        OUTPUT / "false_positive_context.csv")
    boot = pd.DataFrame(binary).assign(label=labels.to_numpy(), child_id=test["child_id"].to_numpy())
    intervals = {name: bootstrap(boot, name) for name in (*PRIMARY[1:], "deployed")}
    pr = score_curves(pd.DataFrame(curves).assign(label=labels.to_numpy()))
    pr.to_csv(OUTPUT / "test_pr.csv", index=False)
    figure = Figure(figsize=(9, 5), layout="constrained")
    axis = figure.subplots()
    for name in (*PRIMARY, "deployed"):
        group = pr[pr["design"].eq(name)]
        ap = comparison.set_index("design").loc[name, "average_precision"]
        axis.step(group["recall"], group["precision"], where="post", label=f"{name} (AP {ap:.3f})")
    axis.axhline(labels.mean(), color="gray", linestyle=":", label="Positive prevalence")
    axis.set(xlabel="Recall", ylabel="Precision", title="Fresh synthetic holdout precision–recall")
    axis.legend()
    figure.savefig(OUTPUT / "test_precision_recall.png", dpi=150)
    setting, model = payload["models"][selected]
    score = scores_for(daily, frames, setting, model)
    flags = binary[selected]
    ewma, cusum = (lock["trends"][name] for name in ("ewma", "cusum"))
    config = TrendConfig(threshold=ewma["cutoff"], alpha=ewma["parameter"],
                         cusum_threshold=cusum["cutoff"], drift=cusum["parameter"])
    history = trend_history(daily, score, setting["threshold"], payload["center"], payload["scale"], config)
    trends = {}
    for method in ("ewma", "cusum"):
        trends[method], details = temporal_outcomes(daily, final, episodes, score, history[f"{method}_alert"])
        details.to_csv(OUTPUT / f"{method}_episodes.csv", index=False)
    xai_rows = pd.concat([test.loc[np.asarray(flags) == value].sample(
        n=min(16, int(np.sum(np.asarray(flags) == value))), random_state=784) for value in (True, False)])
    xai = xai_check(model, frames[setting["window"]].loc[xai_rows.index, setting["features"]],
                    payload["normal_reference"], score)
    xai["flagged"] = score.loc[xai["case"]].to_numpy() > setting["threshold"]
    xai.to_csv(OUTPUT / "xai_stability.csv", index=False)
    write_json(OUTPUT / "evaluation.json", {"selected": setting, "holdout": manifest,
               "test_rows": len(test), "test_positive": int(labels.sum()), "test_children": int(test["child_id"].nunique()),
               "first_test_date": test["date"].min(), "last_test_date": test["date"].max(), "intervals": intervals,
               "temporal": trends, "xai_cases": len(xai),
               "xai_means": xai.drop(columns=["case", "score", "flagged"]).mean().replace({np.nan: None}).to_dict()})
    print(comparison[comparison["design"].isin([*PRIMARY, "deployed"])].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "test"])
    args = parser.parse_args()
    {"fit": fit, "test": evaluate_holdout}[args.stage]()

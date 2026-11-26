"""Research-only Step 04 candidate selection and locked synthetic holdout."""
import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from time import perf_counter
import tracemalloc

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import FunctionTransformer

from .evaluate_temporal_explain import json_value
from .optimize_function02 import bootstrap, xai_check
from .src.behavioral_anomaly import anomaly_scores, assessable_days, build_pipeline, evaluate
from .src.behavioral_explanation_reliability import (
    assess_daily, assess_temporal, daily_evidence, temporal_evidence,
)
from .src.behavioral_optimization import Hybrid, development
from .src.behavioral_profile import BUNDLE, FUSED, PERSONAL, TEMPORAL, ProfileConfig, profiles, score_history
from .src.behavioral_synthetic import build_dataset
from .src.urgent_alerts import urgent_alerts

ROOT = Path(__file__).resolve().parent
DEV = ROOT / "data" / "behavioral_daily.csv"
OUT = ROOT / "outputs" / "anomaly_detection_optimization_step04"
SEED = 2026112619
ALERT_MARGIN = 0.005
RATIOS = {
    "screen_night_share": ("screen_night_incidents", "screen_incidents"),
    "screen_seconds_per_incident": ("screen_flagged_seconds", "screen_incidents"),
    "language_night_share": ("language_night_flags", "language_flags"),
    "language_bullying_share": ("language_bullying_flags", "language_flags"),
}
VARIABILITY = ["screen_incidents", "language_flags", "language_bullying_flags"]
DIRECT = {"language_urgent_flags", "language_grooming_flags", "language_self_harm_flags"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(serializable(value), allow_nan=False, indent=2) + "\n", encoding="utf-8")


def serializable(value):
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return json_value(value)


def features(daily):
    matrix, details = profiles(daily, ProfileConfig(days=14))
    for label, (numerator, denominator) in RATIOS.items():
        top, bottom = matrix[numerator], matrix[denominator]
        valid = top.ge(0) & bottom.ge(0) & (top.le(bottom) if label.endswith("share") else True)
        matrix[label] = top.div(bottom.where(bottom.gt(0))).where(valid)
        matrix.loc[valid & bottom.eq(0) & top.eq(0), label] = 0
    for column in VARIABILITY:
        matrix[column + "__prior_std"] = details["std"][column]
    return matrix


def metrics(labels, scores, threshold):
    result = evaluate(labels, scores, threshold)
    result["fpr"] = result["fp"] / (result["fp"] + result["tn"]) if result["fp"] + result["tn"] else None
    result["fnr"] = result["fn"] / (result["fn"] + result["tp"]) if result["fn"] + result["tp"] else None
    mixed = pd.Series(labels).nunique() == 2
    result["pr_auc_ap"] = float(average_precision_score(labels, scores)) if mixed else None
    result["roc_auc"] = float(roc_auc_score(labels, scores)) if mixed else None
    return result


def cutoff(labels, scores, budget):
    """Select a strict > cutoff using complete tie groups under the alert budget."""
    fpr, recall, levels = roc_curve(labels, scores, drop_intermediate=False)
    thresholds = np.where(np.isfinite(levels), np.nextafter(levels, -np.inf), max(scores))
    tp = recall * np.sum(labels)
    fp = fpr * np.sum(np.asarray(labels) == 0)
    f1 = 2 * tp / (np.sum(labels) + tp + fp)
    keep = np.flatnonzero(fpr <= budget)
    best = keep[np.lexsort((-thresholds[keep], fpr[keep], -f1[keep]))[0]]
    return float(thresholds[best])


def build_designs(reference_columns):
    history = PERSONAL + TEMPORAL + [column + "__prior_std" for column in VARIABILITY]
    ratios = list(RATIOS)
    return {
        "global": reference_columns, "global_without_totals": FUSED,
        "screen_only": [name for name in FUSED if name.startswith("screen_")],
        "language_only": [name for name in FUSED if name.startswith("language_")],
        "global_ratios": FUSED + ratios, "personalized": FUSED + history,
        "combined": FUSED + history + ratios,
        "combined_without_direct_safety": [name for name in FUSED if name not in DIRECT] + history + ratios,
    }


def search(daily, frame, deployed):
    covered = assessable_days(daily, frame)
    train = covered & daily["split"].eq("train") & daily["is_anomaly"].eq(0)
    valid = covered & daily["split"].eq("validation")
    y = daily.loc[valid, "is_anomaly"]
    prior = anomaly_scores(deployed["pipeline"], frame.loc[valid, deployed["features"]])
    reference = metrics(y, prior, deployed["threshold"])
    budget = reference["fpr"] + ALERT_MARGIN
    results, fitted = [], {}

    def register(name, design, algorithm, prep, params, columns, model):
        scores = anomaly_scores(model, frame.loc[valid, columns])
        threshold = cutoff(y, scores, budget)
        results.append(dict(name=name, design=design, algorithm=algorithm, preprocessing=prep,
                            parameters=json.dumps(params, sort_keys=True), features=json.dumps(columns),
                            threshold=threshold, **metrics(y, scores, threshold)))
        fitted[name] = (model, columns, threshold)

    register("deployed_recalibrated", "threshold_only", "one_class_svm", "standard", {},
             deployed["features"], deployed["pipeline"])
    grid = [
        ("one_class_svm", {"nu": nu, "gamma": gamma})
        for nu, gamma in product((.01, .05), (.001, .01))
    ] + [("isolation_forest", {"n_estimators": 64, "max_samples": size, "n_jobs": 1})
         for size in (128, 256)]
    for design, columns in build_designs(deployed["features"]).items():
        for (algorithm, params), prep in product(grid, ("standard", "asinh")):
            pipeline = build_pipeline(algorithm, params)
            if prep == "asinh":
                pipeline.steps.insert(1, ("asinh", FunctionTransformer(np.arcsinh, feature_names_out="one-to-one")))
            pipeline.fit(frame.loc[train, columns])
            register(f"{design}_{len(results):03d}", design, algorithm, prep, params, columns, pipeline)
    # Use the deployed global score as a common anchor to isolate the hybrid addition.
    normal = frame.loc[train]
    raw = deployed["features"]
    global_ref = np.sort(anomaly_scores(deployed["pipeline"], normal[raw]))
    personal_ref = np.sort(Hybrid.personal_scores(normal[PERSONAL]))
    for weight in (.15, .35, .55):
        model = Hybrid(deployed["pipeline"], raw, PERSONAL, global_ref, personal_ref, weight)
        register(f"hybrid_{weight}", "hybrid", "rarity_blend", "empirical_normal_reference",
                 {"weight": weight}, raw + PERSONAL, model)
    table = pd.DataFrame(results).sort_values(["f1", "fpr", "pr_auc_ap", "name"],
                                              ascending=[False, True, False, True])
    return table, reference, budget, fitted, train


def lock_development():
    deployed = joblib.load(BUNDLE)
    daily = development(pd.read_csv(DEV))
    frame = features(daily)
    table, reference, budget, models, train = search(daily, frame, deployed)
    best = table.iloc[0]
    pipeline, columns, threshold = models[best["name"]]
    normal = frame.loc[train, columns]
    scores = anomaly_scores(pipeline, normal)
    candidate = dict(pipeline=pipeline, features=columns, threshold=threshold,
                     center=float(np.median(scores)), scale=float(np.std(scores)),
                     trend=deployed["trend"], background=normal.sample(n=24, random_state=42))
    joblib.dump(candidate, OUT / "candidate.joblib")
    # Keep a larger training-only donor pool for a separate donor-sensitivity probe.
    donor_columns = list(dict.fromkeys(deployed["features"] + columns))
    frame.loc[train, donor_columns].sample(n=120, random_state=315).to_csv(
        OUT / "training_donors.csv", index=False, float_format="%.17g",
    )
    table.to_csv(OUT / "validation_search.csv", index=False)
    table.groupby("design", sort=False).head(1).to_csv(OUT / "validation_ablations.csv", index=False)
    recipe = table[(table["algorithm"] == "one_class_svm") & (table["preprocessing"] == "standard")
                   & table["parameters"].eq(json.dumps({"nu": .01, "gamma": .001}, sort_keys=True))]
    recipe.to_csv(OUT / "validation_matched_ablations.csv", index=False)
    consumed = ROOT / "outputs" / "function02_optimization" / "sealed_holdout" / "daily.csv"
    historical = pd.read_csv(consumed)
    old_matrix = features(historical)
    test = historical["split"].ne("train") & assessable_days(historical, old_matrix)
    previous = metrics(historical.loc[test, "is_anomaly"],
                       anomaly_scores(deployed["pipeline"], old_matrix.loc[test, deployed["features"]]),
                       deployed["threshold"])
    save("reference_audit.json", {"validation": reference, "consumed_holdout_reproduction": previous,
                                 "deployed_threshold": deployed["threshold"], "deployed_features": deployed["features"],
                                 "train_normal_rows": int(train.sum())})
    save("selection_lock.json", {
        "seed": SEED, "fit_data_sha256": digest(DEV), "deployed_sha256": digest(BUNDLE),
        "script_sha256": digest(Path(__file__)),
        "candidate_sha256": digest(OUT / "candidate.joblib"),
        "donors_sha256": digest(OUT / "training_donors.csv"),
        "candidate": best.to_dict(), "validation_fpr_limit": budget,
        "selection": "Best validation F1 within deployed validation FPR + 0.005, then lower FPR, higher AP, name.",
        "holdout": "Fresh cohort, last 34 dates, both arms on common assessable rows. Earlier dates supply history only.",
        "recommendation_gate": "Paired child-bootstrap lower delta F1 > 0; FPR increase no more than 0.005; "
                               "review XAI fidelity, reliability and runtime before proposing deployment.",
    })
    print(best[["name", "f1", "fpr", "threshold"]].to_dict(), flush=True)


def timing(model, values):
    ms = []
    for _ in range(12):
        start = perf_counter()
        anomaly_scores(model, values)
        ms.append((perf_counter() - start) * 1000)
    tracemalloc.start()
    anomaly_scores(model, values)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"rows": len(values), "median_ms": float(np.median(ms)), "p95_ms": float(np.quantile(ms, .95)),
            "peak_traced_bytes": peak}


def explain_holdout(frame, bundle, history, test, donors, name, examples):
    selected = history.loc[test]
    columns = bundle["features"]
    checked = xai_check(bundle["pipeline"], frame.loc[examples, columns], donors[columns], history["score"])
    checked.index = examples
    checked["arm"] = name
    for index in examples:
        score, flag = history.loc[index, ["score", "flag"]]
        start = perf_counter()
        result, variants = daily_evidence(frame.loc[[index], columns], score, bundle)
        review = assess_daily(result, score, flag, bundle, variants)
        checked.loc[index, "daily_ms"] = (perf_counter() - start) * 1000
        checked.loc[index, "review"] = review.status
        checked.loc[index, "corroborated"] = review.corroborated
    temporal_checks = []
    for endpoint in selected.groupby("child_id").tail(1).head(8).itertuples():
        start = perf_counter()
        result, variants = temporal_evidence(history, frame, bundle, endpoint.child_id, endpoint.date)
        review = assess_temporal(result, variants)
        temporal_checks.append(dict(arm=name, child=endpoint.child_id, date=str(endpoint.date), pattern=result.status,
                                    status=review.status, group=result.group, error=result.reconstruction_error,
                                    milliseconds=(perf_counter() - start) * 1000))
    return checked, pd.DataFrame(temporal_checks)


def holdout():
    lock = json.loads((OUT / "selection_lock.json").read_text())
    for path, field in ((Path(__file__), "script_sha256"), (BUNDLE, "deployed_sha256"),
                        (DEV, "fit_data_sha256"), (OUT / "candidate.joblib", "candidate_sha256"),
                        (OUT / "training_donors.csv", "donors_sha256")):
        if digest(path) != lock[field]:
            raise RuntimeError(f"Lock mismatch for {path.name}")
    daily = build_dataset(lock["seed"]).daily
    daily.to_csv(OUT / "fresh_holdout.csv", index=False)
    frame = features(daily)
    test = daily["split"].ne("train") & assessable_days(daily, frame)
    output = daily.loc[test, ["child_id", "date", "is_anomaly", "anomaly_type", "episode_id"]].copy()
    output["label"] = output["is_anomaly"]
    output["urgent"] = urgent_alerts(daily)["is_urgent"].loc[test]
    dau, tau, child_metrics, category, summary = [], [], [], [], {}
    donors = pd.read_csv(OUT / "training_donors.csv", float_precision="round_trip")
    for name, path in (("deployed", BUNDLE), ("candidate", OUT / "candidate.joblib")):
        bundle = joblib.load(path)
        history = score_history(daily, frame, bundle)
        scores = history.loc[test, "score"]
        output[name] = history.loc[test, "flag"].astype(bool)
        output[name + "_score"] = scores
        if name == "deployed":
            examples = pd.concat([group.sample(n=min(len(group), 8), random_state=120)
                                  for _, group in output.groupby("deployed")]).index
        summary[name] = metrics(output["label"], scores, bundle["threshold"])
        columns = bundle["features"]
        summary[name]["runtime"] = timing(bundle["pipeline"], frame.loc[scores.index[:40], columns])
        summary[name]["single_row_runtime"] = timing(bundle["pipeline"], frame.loc[scores.index[:1], columns])
        summary[name]["artifact_bytes"] = path.stat().st_size
        for child, group in output.groupby("child_id"):
            child_metrics.append(dict(arm=name, child=child, **metrics(group["label"], scores.loc[group.index],
                                                                       bundle["threshold"])))
        for kind, group in output.loc[output["label"].eq(1)].groupby("anomaly_type"):
            category.append(dict(arm=name, category=kind, days=len(group), detected=int(group[name].sum())))
        daily_xai, temporal_xai = explain_holdout(frame, bundle, history, test, donors, name, examples)
        dau.append(daily_xai)
        tau.append(temporal_xai)
        summary[name]["xai"] = {
            "daily_cases": len(daily_xai), "supported": int(daily_xai["review"].eq("supported").sum()),
            "corroborated": int(daily_xai["corroborated"].eq(True).sum()),
            "max_shap_reconstruction_error": daily_xai["shap_additive_absolute_error"].max(),
            "mean_lime_r2": daily_xai["lime_weighted_r2"].mean(),
            "mean_shap_top_agreement": daily_xai["shap_top_agreement"].mean(),
            "mean_shap_sign_agreement": daily_xai["shap_sign_agreement"].mean(),
            "mean_lime_top_agreement": daily_xai["lime_top_agreement"].mean(),
            "median_daily_review_ms": daily_xai["daily_ms"].median(),
            "temporal_supported": int(temporal_xai["status"].eq("supported").sum()),
            "max_temporal_error": temporal_xai["error"].max(),
            "median_temporal_review_ms": temporal_xai["milliseconds"].median(),
        }
        errors = output.loc[output[name].ne(output["label"].astype(bool))]
        errors.join(daily[["screen_incidents", "screen_night_incidents", "language_flags",
                           "language_bullying_flags", "language_max_risk"]]).to_csv(OUT / f"{name}_errors.csv", index=False)
        summary[name]["false_positive_urgent_overlap"] = int((errors["urgent"] & errors["label"].eq(0)).sum())
    output.to_csv(OUT / "holdout_predictions.csv", index=False)
    pd.DataFrame(child_metrics).to_csv(OUT / "per_child.csv", index=False)
    pd.DataFrame(category).to_csv(OUT / "per_category.csv", index=False)
    pd.concat(dau).to_csv(OUT / "daily_xai.csv", index_label="record")
    pd.concat(tau).to_csv(OUT / "temporal_xai.csv", index=False)
    summary["paired_child_bootstrap"] = bootstrap(output, "candidate", "deployed")
    summary["cohort"] = {"rows": len(output), "positives": int(output["label"].sum()),
                         "children": output["child_id"].nunique(), "dataset_sha256": digest(OUT / "fresh_holdout.csv")}
    save("holdout_metrics.json", summary)
    print(json.dumps(serializable(summary), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "holdout"))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (lock_development if args.stage == "validate" else holdout)()


if __name__ == "__main__":
    main()

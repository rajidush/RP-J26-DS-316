"""Calibrate on development data, then evaluate the locked daily and temporal policy."""
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import hashlib
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

from .evaluate_temporal_explain import controlled, GROUPS, json_value, prepare
from .src.behavioral_explain import SCORE_ATOL, explain
from .src.behavioral_explanation_reliability import (
    DEFAULT_POLICY, Policy, assess_daily, assess_temporal, daily_evidence, leader, temporal_evidence,
)
from .src.behavioral_profile import BUNDLE
from .src.behavioral_temporal_explain import explain_temporal

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "xai_reliability_step02"
SCENARIOS = ("stable", "gradual", "sustained", "spike", "missing")
PROBE = 0.05


def save(name, data):
    (OUT / name).write_text(json.dumps(json_value(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def model_hash():
    return hashlib.sha256(BUNDLE.read_bytes()).hexdigest()


def cases(data, phase, bundle):
    split = data.loc[data["split"].eq(phase)]
    first, last = split["date"].min(), split["date"].max()
    anchors = data.loc[data["date"].lt(first) & data["is_anomaly"].eq(0)]
    anchors = anchors.sort_values(["child_id", "date"]).groupby("child_id").tail(1).head(4)
    for offset, (_, base) in enumerate(anchors.iterrows()):
        for scenario in SCENARIOS:
            sources = GROUPS if scenario in {"gradual", "sustained", "spike"} else ("screen",)
            for source in sources:
                injected = GROUPS[source] if len(sources) == 2 else None
                yield scenario, controlled(base, last - pd.Timedelta(days=offset), scenario, source), injected, True
    for _, child in data.sort_values("date").groupby("child_id"):
        through = child.loc[child["date"].le(last)]
        yield "cohort_endpoint", through, None, True
        history, _ = prepare(through, bundle)
        eligible = history.loc[history["date"].between(first, last) & history["score"].notna()]
        if not eligible.empty:
            closest = (eligible["score"] - bundle["threshold"]).abs().idxmin()
            staged = through.loc[through["date"].le(history.at[closest, "date"])]
            yield "near_boundary", staged, None, False


def collect(data, phase, bundle):
    observations = []
    for scenario, daily, injected, include_temporal in cases(data, phase, bundle):
        history, matrix = prepare(daily, bundle)
        i = history.index[-1]
        child, date, score, flag = history.loc[i, ["child_id", "date", "score", "flag"]]
        row = matrix.loc[[i], bundle["features"]]
        flag = bool(flag) if pd.notna(flag) else False
        start = perf_counter()
        raw = explain(bundle["pipeline"], row, bundle["background"]) if np.isfinite(score) else None
        baseline_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        result, variants = daily_evidence(row, score, bundle)
        assessment = assess_daily(result, score, flag, bundle, variants)
        proposed_ms = (perf_counter() - start) * 1000
        probe_result, probe_variants = daily_evidence(row, score, bundle, PROBE)
        probe = assess_daily(probe_result, score, flag, bundle, probe_variants, Policy(1, 0))
        sign = 1 if flag else -1
        eligible = {key: value for key, value in raw["shap"].items() if value * sign > SCORE_ATOL} if raw else {}
        baseline_group = max(eligible, key=lambda key: abs(eligible[key])) if eligible else None
        alternative = explain(bundle["pipeline"], row, bundle["background"], seed=69) if raw else None
        lime_group = assessment.diagnostics.get("lime_group")
        alt_group = leader(alternative["lime"], sign) if alternative else None
        alt_valid = bool(alternative and alternative["lime_variance"] > SCORE_ATOL**2
                         and alternative["lime_r2"] > 0 and lime_group and alt_group == lime_group)
        record = {
            "scenario": scenario, "child": child, "date": str(date.date()), "injected_group": injected,
            "score": score, "margin": score - bundle["threshold"], "flagged": flag,
            "daily_before": baseline_group, "daily_probe_supported": probe.status == "supported",
            "daily_before_ms": baseline_ms, "daily_after_ms": proposed_ms, "lime_alt_supported": alt_valid,
            "lime_alt_r2": alternative["lime_r2"] if alternative else None,
        }
        evidence = (result, score, flag, variants)
        if include_temporal:
            start = perf_counter()
            temporal, temporal_variants = temporal_evidence(history, matrix, bundle, child, date)
            record["temporal_after_ms"] = (perf_counter() - start) * 1000
            start = perf_counter()
            baseline = explain_temporal(history, matrix, bundle, child, date)
            record["temporal_before_ms"] = (perf_counter() - start) * 1000
            probe_temporal, probe_temporal_variants = temporal_evidence(history, matrix, bundle, child, date, PROBE)
            probe = assess_temporal(probe_temporal, probe_temporal_variants, Policy(1, 0))
            record.update(temporal_before=baseline.group, temporal_pattern=baseline.status,
                          temporal_unclear_reason=baseline.reason, temporal_probe_supported=probe.status == "supported")
        else:
            temporal, temporal_variants = None, []
        observations.append((record, evidence, temporal, temporal_variants))
    return observations


def decisions(observations, bundle, policy):
    rows = []
    for original, (result, score, flag, variants), temporal, temporal_variants in observations:
        daily = assess_daily(result, score, flag, bundle, variants, policy)
        row = dict(original, daily_status=daily.status, daily_after=daily.group,
                   lime_corroborated=daily.corroborated, daily_reason=" ".join(daily.reasons))
        row.update(daily.diagnostics)
        if temporal:
            review = assess_temporal(temporal, temporal_variants, policy)
            row.update(temporal_status=review.status, temporal_after=review.group,
                       temporal_reason=" ".join(review.reasons),
                       temporal_vote=review.diagnostics.get("stability_vote"),
                       temporal_error=temporal.reconstruction_error)
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_grid(observations, bundle):
    outputs = []
    for stability in (0.5, 1.0):
        for r2 in (0.0, 0.5, 0.7, 0.9):
            table = decisions(observations, bundle, Policy(stability, r2))
            daily = table["daily_before"].notna()
            temporal = table["temporal_before"].notna()
            target = pd.concat([table.loc[daily, "daily_probe_supported"],
                                table.loc[temporal, "temporal_probe_supported"]]).astype(bool)
            predicted = pd.concat([table.loc[daily, "daily_after"], table.loc[temporal, "temporal_after"]]).notna()
            lime = table["lime_valid"].eq(True) & table["lime_group"].notna()
            lime_ok = table.loc[lime, "lime_alt_supported"].astype(bool)
            lime_pred = table.loc[lime, "lime_r2"].ge(r2)
            outputs.append({
                "stability": stability, "lime_r2": r2,
                "unstable_named": int((predicted & ~target).sum()), "stable_named": int((predicted & target).sum()),
                "lime_false_positive": int((lime_pred & ~lime_ok).sum()),
                "lime_true_positive": int((lime_pred & lime_ok).sum()),
                "lime_accuracy": float(lime_pred.eq(lime_ok).mean()),
            })
    grid = pd.DataFrame(outputs)
    ordering = ["unstable_named", "stable_named", "lime_accuracy", "lime_true_positive", "stability", "lime_r2"]
    best = grid.sort_values(ordering, ascending=[True, False, False, False, False, False]).iloc[0]
    return grid, Policy(float(best["stability"]), float(best["lime_r2"]))


def summarize(data):
    controlled_changes = data["injected_group"].notna()
    summary = {"cases": len(data), "controlled_changes": int(controlled_changes.sum())}
    for mode in ("daily", "temporal"):
        eligible = data.loc[data[mode + "_status"].notna()]
        for stage in ("before", "after"):
            names = eligible[mode + "_" + stage]
            named = names.notna()
            summary[mode + "_" + stage] = {
                "cases": len(eligible), "named": int(named.sum()), "neutral": int((~named).sum()),
                "named_but_probe_failed": int((named & ~eligible[mode + "_probe_supported"].eq(True)).sum()),
                "controlled_injected_hits": int((names.eq(eligible["injected_group"])
                                                 & eligible["injected_group"].notna()).sum()),
                "median_ms": eligible[mode + "_" + stage + "_ms"].median(),
            }
        summary[mode + "_statuses"] = eligible[mode + "_status"].value_counts().to_dict()
    summary.update({
        "max_daily_error": data["shap_error"].max(), "max_temporal_error": data["temporal_error"].max(),
        "median_lime_r2": data["lime_r2"].median(), "median_lime_rmse": data["lime_rmse"].median(),
        "median_lime_mae": data["lime_mae"].median(),
        "lime_group_agreement_n": int(data["lime_group_agrees"].eq(True).sum()),
        "lime_direction_agreement_n": int(data["lime_direction_agrees"].eq(True).sum()),
        "lime_comparable_n": int((data["lime_valid"].eq(True) & data["shap_group"].notna()).sum()),
        "lime_below_gate_n": int(data["daily_reason"].eq("lime_below_fidelity_gate").sum()),
        "corroborated_n": int(data["lime_corroborated"].sum()),
        "mean_full_rank_stability": data["full_rank_stability"].mean(),
        "mean_direction_stability": data["direction_stability"].mean(),
        "decision_sensitive_n": int(data["decision_flip"].eq(True).sum()),
        "historical_unclear": dict(Counter(data.loc[data["scenario"].eq("cohort_endpoint"),
                                                   "temporal_unclear_reason"].dropna())),
    })
    return summary


def stress_cases(observations, bundle, policy):
    """Inject diagnostic faults over valid holdout SHAP to exercise policy branches."""
    for record, (output, score, flag, variants), _, _ in observations:
        original = assess_daily(output, score, flag, bundle, variants, policy)
        if flag and original.status == "supported":
            break
    else:
        raise RuntimeError("The diagnostic stress cases need one supported flagged day")
    group = original.group
    competitor = next(name for name in output["shap"] if name != group)
    corroborating = dict.fromkeys(output["shap"], 0.0)
    corroborating[group] = 1.0
    opposing = dict.fromkeys(output["shap"], 0.0)
    opposing[competitor] = 1.0
    opposing[group] = -1.0
    low_error = np.sqrt(output["lime_variance"] * 1.25)
    low = {**output, "lime": corroborating, "lime_r2": -0.25, "lime_rmse": low_error, "lime_mae": low_error}
    high_error = np.sqrt(output["lime_variance"] * 0.01)
    conflict = {**output, "lime": opposing, "lime_r2": 0.99, "lime_rmse": high_error, "lime_mae": high_error}
    shuffled = dict(output["shap"])
    shuffled[group], shuffled[competitor] = shuffled[competitor], shuffled[group]
    unstable = ({**output, "shap": shuffled}, score)
    rows = []
    for name, result, replays in [
        ("original", output, variants),
        ("low_lime_fit", low, variants),
        ("lime_disagrees_stable_shap", conflict, variants),
        ("lime_disagrees_unstable_shap", conflict, [unstable, unstable]),
        ("unstable_shap_rank", output, [unstable, unstable]),
    ]:
        reviewed = assess_daily(result, score, flag, bundle, replays, policy)
        rows.append({"case": name, "base_child": record["child"], "base_date": record["date"],
                     "status": reviewed.status, "group": reviewed.group,
                     "lime_corroborated": reviewed.corroborated, "reasons": " ".join(reviewed.reasons)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "holdout"))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    bundle = joblib.load(BUNDLE)
    if args.stage == "validate":
        data = pd.read_csv(ROOT / "data" / "behavioral_daily.csv", parse_dates=["date"])
        observed = collect(data, "validation", bundle)
        grid, policy = threshold_grid(observed, bundle)
        grid.to_csv(OUT / "threshold_sensitivity.csv", index=False)
        decisions(observed, bundle, policy).to_csv(OUT / "validation.csv", index=False)
        save("policy_lock.json", {"policy": asdict(policy), "model_sha256": model_hash(), "development": "validation"})
        print(policy, flush=True)
        return
    lock = json.loads((OUT / "policy_lock.json").read_text(encoding="utf-8"))
    policy = Policy(**lock["policy"])
    if policy != DEFAULT_POLICY or lock["model_sha256"] != model_hash():
        raise RuntimeError("Holdout requires the validated policy and deployed model to match the lock")
    data = pd.read_csv(ROOT / "data" / "step4a_heldout_seed20261020" / "behavioral_daily.csv", parse_dates=["date"])
    observed = collect(data, "test", bundle)
    table = decisions(observed, bundle, policy)
    stress_cases(observed, bundle, policy).to_csv(OUT / "diagnostic_stress.csv", index=False)
    table.to_csv(OUT / "holdout.csv", index=False)
    summaries = {"all": summarize(table)}
    for label, subset in table.groupby("scenario"):
        if label != "near_boundary":
            summaries[label] = summarize(subset)
    save("metrics.json", summaries)
    rates = pd.DataFrame({
        f"{mode} {stage}": [summaries["all"][f"{mode}_{stage}"][key] for key in ("named", "named_but_probe_failed")]
        for mode in ("daily", "temporal") for stage in ("before", "after")
    }, index=["Named group", "Named group that failed the larger probe"])
    axis = rates.T.plot.barh(color=["#2369a2", "#cc692e"], figsize=(9, 4))
    axis.set(xlabel="Cases", title="Holdout coverage and probe failures")
    axis.figure.tight_layout()
    axis.figure.savefig(OUT / "coverage.png", dpi=160)
    plt.close(axis.figure)
    print(json.dumps(json_value(summaries["all"]), indent=2), flush=True)


if __name__ == "__main__":
    main()

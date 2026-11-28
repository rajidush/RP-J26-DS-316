"""Run the preregistered controlled robustness replication without changing the frozen model."""
from argparse import ArgumentParser
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import behavioral_shap_robustness_synthetic as design
from .evaluate_behavioral_shap_independent import DATA, HOLDOUT, MANIFEST, ROOT, digest, verify_freeze, write_json
from .src.behavioral_anomaly import FEATURE_SET_A, feature_matrix, predict_anomalies
from .src.behavioral_shap import BehavioralShap, TOLERANCE, mask_violations, normal_training_rows
from .src.behavioral_shap_reliability import ReliabilityGate

OUT = DATA.parent.parent / "outputs"
LOCK = OUT / "behavioral_robustness_preregistration_lock_v1.json"
REPLICATES = 10_000
PRIMARY_ARMS = design.ARMS[1:-1]
MARGIN = 0.05
ALPHA = 0.05
OUTCOMES = ("TP", "FP", "FN", "TN", "insufficient_data", "reliable",
            "uncertain", "support_failure", "stable_leader", "attributed")
SUPPORT_REASONS = ("target_context_sparse", "explanation_donors_sparse", "deletion_donors_sparse")


def verify_plan():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for path, fingerprint in lock["sha256"].items():
        if digest(ROOT / path) != fingerprint:
            raise ValueError(f"preregistered file has changed: {path}")
    if version("matplotlib") != lock["plotting_dependency"]:
        raise ValueError("Matplotlib version differs from preregistration")
    verify_freeze(json.loads(MANIFEST.read_text(encoding="utf-8")))


def validate_design(daily, train, previous):
    if daily.duplicated(["arm", "pair_id", "day_index"]).any():
        raise ValueError("duplicate paired day")
    if len(daily) != len(design.ARMS) * design.PAIRS * design.DAYS:
        raise ValueError("cohort size differs from preregistration")
    indexed = {arm: daily.loc[daily.arm.eq(arm)].set_index(["pair_id", "day_index"]).sort_index()
               for arm in design.ARMS}
    base = indexed["reference"]
    if not set(base.child_id) <= set(train.child_id):
        raise ValueError("a seen profile is absent from normal training")
    if set(indexed["unseen_profile"].child_id) & (set(train.child_id) | set(previous.child_id)):
        raise ValueError("unseen profile leaked from an earlier dataset")
    for arm, data in indexed.items():
        if not data[["anomaly_type", "is_anomaly", "profile_age"]].equals(base[["anomaly_type", "is_anomaly", "profile_age"]]):
            raise ValueError(f"label or age pairing differs: {arm}")
    for arm, add_screen, add_language in (("screen_plus_two", 2, 0), ("language_plus_two", 0, 2)):
        actual = indexed[arm][["screen_incidents", "language_flags"]] - base[["screen_incidents", "language_flags"]]
        if not actual.eq([add_screen, add_language]).all(axis=None):
            raise ValueError(f"unpaired volume shift: {arm}")
        untouched = "language_" if add_screen else "screen_"
        cols = [f for f in FEATURE_SET_A if f.startswith(untouched)]
        if not indexed[arm][cols].equals(base[cols]):
            raise ValueError(f"volume arm changed the other source: {arm}")
    if not indexed["balance_to_screen"].total_risk_events.equals(base.total_risk_events):
        raise ValueError("balance arm changed total volume")
    non_night = [f for f in FEATURE_SET_A if "night" not in f]
    if not indexed["night_40pct"][non_night].equals(base[non_night]):
        raise ValueError("night arm changed a non-night input")
    non_calendar = [f for f in FEATURE_SET_A if f != "is_weekend"]
    if not indexed["weekend"][non_calendar].equals(base[non_calendar]):
        raise ValueError("calendar arm changed a non-calendar input")
    if base.is_weekend.any() or not indexed["weekend"].is_weekend.all():
        raise ValueError("calendar condition is mislabeled")
    missing = indexed["coverage_loss"].outage.ne("none")
    if missing.sum() != 3 * design.PAIRS or not indexed["coverage_loss"].loc[~missing, FEATURE_SET_A].equals(base.loc[~missing, FEATURE_SET_A]):
        raise ValueError("coverage arm changed observed records")
    if mask_violations(feature_matrix(daily.loc[daily.outage.eq("none")], FEATURE_SET_A)).any(axis=None):
        raise ValueError("generated a structurally invalid assessed day")


def run_gate(daily, original):
    scores = predict_anomalies(daily)
    gate = ReliabilityGate(BehavioralShap(), original)
    features = feature_matrix(daily, FEATURE_SET_A)
    feature_rows = features.to_numpy(dtype="<f8")
    cache, results = {}, []
    for i, row in enumerate(daily.itertuples(index=False)):
        # Policy inputs are exactly the 27 features and two observed indicators.
        key = feature_rows[i].tobytes(), row.screen_monitor_observed, row.language_monitor_observed
        if key not in cache:
            cache[key] = gate.review(daily.iloc[[i]])
        decision = cache[key]
        expected = scores.iloc[i]
        absent = pd.isna(expected.is_flagged)
        if absent:
            if decision.status.value != "insufficient_data" or decision.text is not None or decision.score is not None:
                raise ValueError("coverage decision changed")
            outcome = "insufficient_data"
        else:
            if decision.is_flagged != bool(expected.is_flagged) or abs(decision.score - expected.anomaly_score) > TOLERANCE:
                raise ValueError("frozen score or prediction changed")
            outcome = ("TP" if row.is_anomaly else "FP") if decision.is_flagged else ("FN" if row.is_anomaly else "TN")
        diag = decision.diagnostics
        orders = diag.get("stability", {}).get("rank_orders", [])
        full = orders[:len(diag.get("full_groups", []))]
        results.append({
            "pair_id": row.pair_id, "day_index": row.day_index, "arm": row.arm,
            "child_id": row.child_id, "is_anomaly": row.is_anomaly, "anomaly_type": row.anomaly_type,
            "outage": row.outage, "outcome": outcome, "score": decision.score,
            "status": decision.status.value, "reasons": decision.reasons, "text": decision.text,
            "support_keys_failed": [k for k, passed in diag.get("target_support", {}).items() if not passed],
            "support_failure": any(reason in decision.reasons for reason in SUPPORT_REASONS),
            "full_leaders": [order[0] for order in full],
            "restricted_leaders": [order[0] for order in orders[len(full):]],
            "full_sign_agreement": {g: len({float(np.sign(values[g])) for values in diag["full_groups"]}) == 1
                                    for g in diag["full_groups"][0]} if full else {},
            "full_checks": diag.get("full_score_checks", []),
            "restricted_checks": diag.get("supported_score_checks", []),
        })
    return results, len(cache)


def count_vectors(results, pairs):
    counts = {arm: {label: np.zeros((len(pairs), len(OUTCOMES)), dtype=np.int64)
                    for label in ("all", *design.CLUSTERS)} for arm in design.ARMS}
    positions = {pair: i for i, pair in enumerate(pairs)}
    for row in results:
        labels = ("all", row["anomaly_type"]) if row["is_anomaly"] else ("all",)
        for label in labels:
            vec = counts[row["arm"]][label][positions[row["pair_id"]]]
            vec[OUTCOMES.index(row["outcome"])] += 1
            if row["status"] in ("reliable", "uncertain"):
                vec[OUTCOMES.index(row["status"])] += 1
            if row["support_failure"]:
                vec[OUTCOMES.index("support_failure")] += 1
            if row["full_leaders"]:
                vec[OUTCOMES.index("attributed")] += 1
                vec[OUTCOMES.index("stable_leader")] += len(set(row["full_leaders"])) == 1
    return counts


def calculate(counts):
    tp, fp, fn, tn, miss, specific, neutral, sparse, stable, examined = np.moveaxis(counts, -1, 0)
    assessed = tp + fp + fn + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn),
            "fpr": fp / (fp + tn), "fnr": fn / (fn + tp), "coverage": assessed / (assessed + miss),
            "monitoring_abstention": miss / (assessed + miss), "positive_detection_all_days": tp / (tp + fn + miss),
            "specific_among_flagged": specific / (specific + neutral),
            "neutral_among_flagged": neutral / (specific + neutral),
            "support_failure_among_flagged": sparse / (specific + neutral),
            "unanimous_leader_among_attributed": stable / examined,
        }


def scalar(value):
    return float(value) if np.isfinite(value) else None


def interval(values, alpha=ALPHA):
    finite = values[np.isfinite(values)]
    return [float(q) for q in np.quantile(finite, [alpha / 2, 1 - alpha / 2])] if finite.size else None


def summarize(results, metadata):
    pairs = sorted({r["pair_id"] for r in results})
    rng = np.random.default_rng(design.BOOTSTRAP_SEED)
    weights = rng.multinomial(len(pairs), [1 / len(pairs)] * len(pairs), size=REPLICATES)
    counts = count_vectors(results, pairs)
    arms, bootstrap = {}, {}
    for arm, labels in counts.items():
        axes = {}
        for label, rows in labels.items():
            point = rows.sum(axis=0)
            replicates = calculate(weights @ rows)
            axes[label] = {
                "counts": dict(zip(OUTCOMES, map(int, point), strict=True)),
                "metrics": {metric: {"value": scalar(value), "ci95": interval(replicates[metric]),
                                     "valid_bootstrap_replicates": int(np.isfinite(replicates[metric]).sum())}
                            for metric, value in calculate(point).items()
                            if label == "all" and metric != "positive_detection_all_days"
                            or label != "all" and metric in ("recall", "positive_detection_all_days")},
            }
            if label == "all":
                bootstrap[arm] = replicates
        arm_rows = [row for row in results if row["arm"] == arm]
        checks = [x for r in arm_rows for field in ("full_checks", "restricted_checks") for x in r[field]]
        reasons = Counter(reason for r in arm_rows for reason in r["reasons"])
        axes["evidence"] = {
            "reason_counts": dict(reasons),
            "target_key_failures": dict(Counter(key for r in arm_rows for key in r["support_keys_failed"])),
            "primary_leaders": dict(Counter(r["full_leaders"][0] for r in arm_rows if r["full_leaders"])),
            "unstable_sign_by_group": dict(Counter(
                key for r in arm_rows for key, stable in r["full_sign_agreement"].items() if not stable)),
            "reconstruction_attempts": len(checks),
            "reconstruction_failures": sum(
                x is None or not x["threshold_match"] or not x["score_match"]
                or x["invalid_mask_rows"] or x["reconstruction_error"] > TOLERANCE for x in checks),
            "maximum_reconstruction_error": max((x["reconstruction_error"] for x in checks if x), default=None),
        }
        arms[arm] = axes

    reference = arms["reference"]["all"]["metrics"]
    contrasts = {}
    alpha_primary = ALPHA / (len(PRIMARY_ARMS) * 2)
    for arm in design.ARMS[1:]:
        metrics = arms[arm]["all"]["metrics"]
        contrasts[arm] = {}
        for name in ("fpr", "fnr", "f1", "coverage", "specific_among_flagged"):
            delta = bootstrap[arm][name] - bootstrap["reference"][name]
            primary = arm in PRIMARY_ARMS and name in ("fpr", "fnr")
            limits = interval(delta, alpha_primary) if primary else None
            difference = metrics[name]["value"] - reference[name]["value"] if (
                metrics[name]["value"] is not None and reference[name]["value"] is not None
            ) else None
            contrasts[arm][name] = {
                "difference": difference, "paired_ci95": interval(delta),
                "simultaneous_interval": limits,
                "passes_noninferiority_screen": bool(limits and limits[1] <= MARGIN) if primary else None,
            }
    historical = metadata["test"]
    anchor = {"fpr": historical["fp"] / (historical["fp"] + historical["tn"]),
              "fnr": historical["fn"] / (historical["fn"] + historical["tp"])}
    baseline_pass = {name: reference[name]["value"] <= ceiling for name, ceiling in anchor.items()}
    coverage_pass = (arms["coverage_loss"]["all"]["counts"]["insufficient_data"] == 60
                     and all(r["status"] == "insufficient_data" if r["outage"] != "none" else
                             r["status"] != "insufficient_data" for r in results if r["arm"] == "coverage_loss"))
    failure_reasons = {"invalid_feature_relationships", "explanation_failed",
                       "score_or_structure_check_failed", "supported_score_or_structure_check_failed"}
    fidelity_pass = all(a["evidence"]["reconstruction_failures"] == 0 for a in arms.values())
    fidelity_pass &= not any(failure_reasons.intersection(r["reasons"]) for r in results)
    return {
        "synthetic": True, "matched_profile_pairs": len(pairs), "bootstrap_replicates": REPLICATES,
        "threshold": metadata["threshold"], "score_decision_agreement": len(results),
        "arms": arms, "contrasts_vs_reference": contrasts,
        "acceptance": {"historical_reference_screen": anchor, "reference_pass_by_metric": baseline_pass,
                       "primary_maximum_upper_delta": MARGIN, "primary_interval_confidence": 1 - alpha_primary,
                       "primary_endpoints_passing": sum(
                           bool(c["passes_noninferiority_screen"]) for arm in contrasts.values() for c in arm.values()),
                       "primary_endpoints_total": 12,
                       "expected_coverage_suppression_pass": coverage_pass,
                       "frozen_attribution_fidelity_pass": fidelity_pass},
    }


def plots(summary, output):
    axes = summary["contrasts_vs_reference"]
    fig, panels = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, layout="constrained")
    labels = [arm.replace("_", " ") for arm in PRIMARY_ARMS]
    for panel, metric, title in zip(panels, ("fpr", "fnr"), ("False positives", "False negatives"), strict=True):
        for i, arm in enumerate(PRIMARY_ARMS):
            result = axes[arm][metric]
            point = result["difference"] * 100
            lo, hi = (value * 100 for value in result["simultaneous_interval"])
            panel.hlines(i, lo, hi, color="#206486", linewidth=2)
            panel.scatter(point, i, color="#123c51", s=30)
        panel.axvline(0, color="#444444", linewidth=1)
        panel.axvline(MARGIN * 100, color="#bd5319", linestyle="--")
        panel.set_yticks(range(len(labels)), labels)
        panel.set_title(title)
        panel.set_xlabel("Difference from reference (percentage points)")
    panels[0].invert_yaxis()
    fig.suptitle("Paired synthetic shifts with simultaneous interval estimates")
    fig.savefig(output / "paired_error_differences.png", dpi=160)
    plt.close(fig)

    fig, panel = plt.subplots(figsize=(9, 4.3), layout="constrained")
    arm_data = summary["arms"]
    x = range(len(arm_data))
    specific = np.array([a["all"]["counts"]["reliable"] for a in arm_data.values()])
    neutral = [a["all"]["counts"]["uncertain"] for a in arm_data.values()]
    panel.bar(x, specific, label="Specific", color="#256c69")
    panel.bar(x, neutral, bottom=specific, label="Neutral", color="#d7a14b")
    panel.set_xticks(list(x), [a.replace("_", " ") for a in arm_data], rotation=25, ha="right")
    panel.set_ylabel("Flagged synthetic days")
    panel.set_title("Parent wording among frozen-model flags")
    panel.legend(frameon=False)
    fig.savefig(output / "flagged_wording.png", dpi=160)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output directory")
    verify_plan()
    lock_sha = digest(LOCK)
    original = pd.read_csv(DATA)
    previous = pd.concat([pd.read_csv(HOLDOUT, usecols=["child_id"]),
                          pd.read_csv(OUT / "shap_independent_v1/synthetic_daily.csv", usecols=["child_id"])])
    daily = design.generate()
    validate_design(daily, normal_training_rows(original), previous)
    print(f"Verified {len(daily)} paired synthetic records", flush=True)
    results, unique = run_gate(daily, original)
    summary = summarize(results, BehavioralShap().metadata)
    summary["unique_policy_inputs"] = unique
    verify_plan()
    if digest(LOCK) != lock_sha:
        raise ValueError("preregistration lock changed")
    args.output.mkdir()
    data = args.output / "synthetic_daily.csv"
    daily.to_csv(data, index=False, mode="x", lineterminator="\n")
    cases = args.output / "decisions.json"
    write_json(cases, results)
    summary["provenance"] = {"lock_sha256": lock_sha, "daily_sha256": digest(data), "decisions_sha256": digest(cases)}
    write_json(args.output / "metrics.json", summary)
    plots(summary, args.output)
    print(f"Scored {len(results)} cases with {unique} unique policy inputs", flush=True)


if __name__ == "__main__":
    main()

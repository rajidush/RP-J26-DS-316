"""Run the locked out-of-scenario synthetic evaluation of the frozen SHAP policy."""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from itertools import combinations
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd

from . import behavioral_shap_independent_synthetic as simulator
from .evaluate_behavioral_shap import DATA, benchmark
from .evaluate_behavioral_shap_expanded import HOLDOUT
from .src.behavioral_anomaly import anomaly_scores, predict_anomalies
from .src.behavioral_shap import GROUPS, TOLERANCE, BehavioralShap
from .src.behavioral_shap_reliability import ReliabilityGate

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = DATA.parent.parent / "outputs" / "behavioral_shap_policy_freeze_manifest_v1.json"
NEAREST_PER_SIDE = 8


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify_freeze(manifest):
    for filename, expected in manifest["sha256"].items():
        if digest(ROOT / filename) != expected:
            raise ValueError(f"frozen input changed: {filename}")
    if platform.python_version() != manifest["environment"]["python"]:
        raise ValueError("Python version differs from the frozen environment")
    for package, expected in manifest["environment"]["packages"].items():
        if version(package) != expected:
            raise ValueError(f"dependency version differs from freeze: {package}")
    for variable, expected in manifest["environment"]["thread_environment"].items():
        if os.getenv(variable) != expected:
            raise ValueError(f"set {variable}={expected} before running")


def rate(num, den):
    return num / den if den else None


def status_metrics(rows):
    statuses = Counter(row["status"] for row in rows)
    flagged = statuses["reliable"] + statuses["uncertain"]
    return {
        "n": len(rows), "status_counts": dict(statuses),
        "assessment_coverage": rate(len(rows) - statuses["insufficient_data"], len(rows)),
        "insufficient_data_rate": rate(statuses["insufficient_data"], len(rows)),
        "flagged": flagged,
        "specific_rate_among_flagged": rate(statuses["reliable"], flagged),
        "neutral_rate_among_flagged": rate(statuses["uncertain"], flagged),
        "specific_rate_over_all_days": rate(statuses["reliable"], len(rows)),
        "neutral_rate_over_all_days": rate(statuses["uncertain"], len(rows)),
        "withhold_reasons_among_flagged": dict(Counter(
            reason for row in rows if row["is_flagged"] for reason in row["reasons"]
        )),
    }


def model_metrics(rows):
    confusion = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
    for row in rows:
        if row["is_flagged"] is not None:
            key = ("t" if row["is_flagged"] == bool(row["is_anomaly"]) else "f")
            confusion[key + ("p" if row["is_flagged"] else "n")] += 1
    tp, fp, fn, tn = (confusion[name] for name in ("tp", "fp", "fn", "tn"))
    positives = sum(row["is_anomaly"] for row in rows)
    return {
        **confusion, "assessed": tp + fp + fn + tn, "total_positive_labels": positives,
        "missing_positive_labels": sum(row["is_anomaly"] for row in rows if row["is_flagged"] is None),
        "precision_assessed": rate(tp, tp + fp), "recall_assessed": rate(tp, tp + fn),
        "f1_assessed": rate(2 * tp, 2 * tp + fp + fn),
        "positive_detection_all_days": rate(tp, positives),
    }


def reconstruction_metrics(checks):
    complete = [check for check in checks if check is not None]
    errors = [check["reconstruction_error"] for check in complete]
    return {
        "attempts": len(checks), "returned": len(complete),
        "score_matches": sum(check["score_match"] for check in complete),
        "threshold_matches": sum(check["threshold_match"] for check in complete),
        "within_tolerance": sum(error <= TOLERANCE for error in errors),
        "invalid_mask_rows": sum(check["invalid_mask_rows"] for check in complete),
        "maximum_absolute_error": max(errors, default=None),
    }


def timing_metrics(rows):
    def elapsed(subset):
        if not subset:
            return None
        seconds = [row["timing"]["seconds"] for row in subset]
        return {
            "n": len(seconds), "median_seconds": float(np.median(seconds)),
            "p95_seconds": float(np.quantile(seconds, .95, method="higher")),
            "max_seconds": max(seconds),
            "peak_rss_bytes": max(row["timing"]["rss_peak_bytes"] for row in subset),
            "max_rss_increase_bytes": max(row["timing"]["rss_peak_increase_bytes"] for row in subset),
        }

    return {"all_days": elapsed(rows), "flagged_days": elapsed([row for row in rows if row["is_flagged"]])}


def evidence_metrics(rows):
    comparisons, directions, unanimous, supported_cases, support_swaps = [], [], 0, 0, 0
    for row in rows:
        diagnostics = row["diagnostics"]
        vectors = diagnostics.get("full_groups", [])
        orders = diagnostics.get("stability", {}).get("rank_orders", [])
        full = orders[:len(vectors)]
        if len(full) < 2:
            continue
        leaders = {order[0] for order in full}
        unanimous += len(leaders) == 1
        comparisons.append(float(np.mean([a[0] == b[0] for a, b in combinations(full, 2)])))
        directions.append(float(np.mean([
            np.sign(a[group]) == np.sign(b[group])
            for a, b in combinations(vectors, 2) for group in GROUPS
        ])))
        restricted = orders[len(vectors):]
        if restricted:
            supported_cases += 1
            support_swaps += len(leaders) == 1 and any(order[0] not in leaders for order in restricted)
    return {
        "flagged_with_attribution_vectors": len(comparisons), "unanimous_full_reference_leader": unanimous,
        "mean_pairwise_full_leader_agreement": float(np.mean(comparisons)) if comparisons else None,
        "mean_pairwise_full_group_direction_agreement": float(np.mean(directions)) if directions else None,
        "cases_with_supported_references": supported_cases,
        "unanimous_full_leader_changed_on_restriction": support_swaps,
        "full_reconstruction": reconstruction_metrics([
            run for row in rows for run in row["diagnostics"].get("full_score_checks", [])
        ]),
        "supported_reconstruction": reconstruction_metrics([
            run for row in rows for run in row["diagnostics"].get("supported_score_checks", [])
        ]),
    }


def cutoff_panel(rows, span):
    assessed = [row for row in rows if row["is_flagged"] is not None]
    panel = {}
    for flagged, name in ((True, "flagged"), (False, "unflagged")):
        side = [row for row in assessed if row["is_flagged"] == flagged]
        ordered = sorted(side, key=lambda row: (abs(row["margin"]), row["child_id"], row["date"]))
        nearest = ordered[:NEAREST_PER_SIDE]
        panel[f"nearest_{name}"] = {
            **status_metrics(nearest), "requested": NEAREST_PER_SIDE,
            "absolute_margin_range": [abs(nearest[0]["margin"]), abs(nearest[-1]["margin"])] if nearest else None,
            "case_keys": [[row["child_id"], row["date"]] for row in nearest],
        }
        panel[f"within_frozen_background_span_{name}"] = status_metrics([
            row for row in side if abs(row["margin"]) <= span + TOLERANCE
        ])
    return panel


def evaluate(cases, original, independent_prior):
    if set(cases.child_id) & (set(original.child_id) | set(independent_prior.child_id)):
        raise ValueError("evaluation child leaked from development datasets")
    if cases.duplicated(["child_id", "date"]).any() or len(cases) != 672:
        raise ValueError("unexpected cohort size or duplicate case")
    started = perf_counter()
    explainer = BehavioralShap()
    gate = ReliabilityGate(explainer, original)
    setup_seconds = perf_counter() - started
    threshold = explainer.metadata["threshold"]
    expected_span = float(np.ptp([anomaly_scores(explainer.pipeline, b.values).mean() for b in gate.references]))
    predictions = predict_anomalies(cases)
    results = []
    for index, row in enumerate(cases.itertuples(index=False)):
        decision, timing = benchmark(lambda: gate.review(cases.iloc[[index]]))
        expected = predictions.iloc[index]
        if pd.isna(expected.is_flagged):
            if decision.is_flagged is not None or decision.score is not None or decision.text is not None:
                raise ValueError("coverage suppression or score disagreement")
        elif (decision.is_flagged != bool(expected.is_flagged)
              or abs(decision.score - expected.anomaly_score) > TOLERANCE):
            raise ValueError("gate changed a frozen score or flag")
        results.append({
            **{name: getattr(row, name) for name in (
                "child_id", "date", "scenario", "profile", "is_anomaly", "anomaly_type",
                "screen_outage", "language_outage",
            )},
            **asdict(decision), "timing": timing,
            "margin": None if decision.score is None else decision.score - threshold,
        })
    summaries = {}
    for scenario in ("overall", *simulator.SCENARIOS):
        rows = [row for row in results if scenario == "overall" or row["scenario"] == scenario]
        summaries[scenario] = {
            "gate": status_metrics(rows), "model": model_metrics(rows),
            "evidence": evidence_metrics(rows), "runtime": timing_metrics(rows),
            "cutoff_panel": cutoff_panel(rows, expected_span),
            "label_counts": dict(Counter(row["anomaly_type"] for row in rows)),
            "outages": dict(Counter(
                "both" if row["screen_outage"] and row["language_outage"] else
                "screen_only" if row["screen_outage"] else "language_only" if row["language_outage"] else "none"
                for row in rows
            )),
            "specific_by_label": dict(Counter(row["is_anomaly"] for row in rows if row["status"] == "reliable")),
        }
    return results, {
        "synthetic": True, "prediction_agreement_cases": len(results), "setup_seconds": setup_seconds,
        "platform": platform.platform(), "processor": platform.processor(),
        "threshold": threshold, "background_expected_score_span": expected_span, "summaries": summaries,
    }


def write_json(path, contents):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(contents, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="new directory for synthetic data and evaluation outputs")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("use a new output directory")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    verify_freeze(manifest)
    manifest_sha = digest(MANIFEST)
    dataset = simulator.generate()
    results, metrics = evaluate(dataset, pd.read_csv(DATA), pd.read_csv(HOLDOUT))
    verify_freeze(manifest)
    if digest(MANIFEST) != manifest_sha:
        raise ValueError("manifest changed during evaluation")
    args.output_dir.mkdir()
    data_file = args.output_dir / "synthetic_daily.csv"
    dataset.to_csv(data_file, index=False, mode="x", lineterminator="\n")
    case_file = args.output_dir / "cases.json"
    write_json(case_file, results)
    metrics["provenance"] = {
        "freeze_manifest_sha256": manifest_sha, "synthetic_daily_sha256": digest(data_file),
        "cases_sha256": digest(case_file),
    }
    write_json(args.output_dir / "metrics.json", metrics)
    print(f"Synthetic validation complete: {len(results)} cases")


if __name__ == "__main__":
    main()

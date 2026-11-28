"""Replay the fixed expanded SHAP panel against the v3 parent reliability policy."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from .evaluate_behavioral_shap_expanded import DATA, HOLDOUT
from .src.behavioral_anomaly import ARTIFACT_DIR, METADATA_FILE, PIPELINE_FILE, predict_anomalies
from .src.behavioral_shap import BehavioralShap, TOLERANCE
from .src import behavioral_shap_reliability as reliability

PANEL = DATA.parent.parent / "outputs" / "behavioral_shap_expanded_v2.json"


def evaluate():
    files = {"original_daily": DATA, "independent_daily": HOLDOUT, "v2_panel": PANEL,
             "pipeline": ARTIFACT_DIR / PIPELINE_FILE, "metadata": ARTIFACT_DIR / METADATA_FILE}
    before = {name: sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    for name in ("pipeline", "metadata", "original_daily"):
        if before[name] != panel["hashes_before"][name]:
            raise ValueError(f"{name} differs from the v2 evaluation")
    lf_hash = sha256(HOLDOUT.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if lf_hash != panel["independent_daily_lf_sha256"]:
        raise ValueError("independent synthetic dataset differs from v2")
    explainer = BehavioralShap()
    original = pd.read_csv(DATA)
    gate = reliability.ReliabilityGate(explainer, original)
    datasets = {}
    for name, daily in (("original_test", original), ("independent_test", pd.read_csv(HOLDOUT))):
        test = daily.loc[daily["split"].eq("test")].set_index(["child_id", "date"], drop=False)
        entries = panel["datasets"][name]["cases"]
        selected = test.loc[[(item["child_id"], item["date"]) for item in entries]]
        predictions = predict_anomalies(selected)
        results = []
        for index, entry in enumerate(entries):
            row = selected.iloc[[index]]
            score = float(predictions.iloc[index]["anomaly_score"])
            flagged = bool(predictions.iloc[index]["is_flagged"])
            decision = gate.review(row)
            if decision.is_flagged != flagged or abs(decision.score - score) > TOLERANCE:
                raise ValueError("gate disagrees with the frozen model prediction")
            if abs(score - entry["score"]) > TOLERANCE or flagged != (entry["margin"] > 0):
                raise ValueError("frozen model prediction differs from v2")
            if flagged:
                full_checks = decision.diagnostics.get("full_score_checks", [])
            else:
                # Reconstruction is a research check. The parent path skips SHAP for these days.
                runs = [explainer.explain(row, b, method="group_aware") for b in gate.references]
                full_checks = gate._score_checks(runs, score, explainer.metadata["threshold"])
            control = gate.review(row.assign(language_monitor_observed=False))
            results.append({
                **{key: entry[key] for key in ("child_id", "date", "stratum", "outcome", "anomaly_type", "margin")},
                "prior_named_groups": list(entry["verified_groups"]),
                "missing_coverage_control": asdict(control),
                "full_score_checks": full_checks,
                **asdict(decision),
            })
        datasets[name] = results
        print(f"{name}: {len(results)} cases", flush=True)

    after = {name: sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    if after != before:
        raise ValueError("a protected file changed during the replay")
    return {
        "input_hashes_before": before, "input_hashes_after": after,
        "policy_sha256": sha256(Path(reliability.__file__).read_bytes()).hexdigest(),
        "backgrounds": [background.metadata() for background in gate.references],
        "deletion_background": gate.deletion.metadata(),
        "supported_training_key_counts": {name: len(keys) for name, keys in gate.support.supported.items()},
        "datasets": datasets,
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

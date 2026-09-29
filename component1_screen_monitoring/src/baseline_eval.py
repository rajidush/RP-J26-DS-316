"""
baseline_eval.py — Baseline evaluation harness for Component 1 (Screen Monitoring).

Responsibilities
----------------
* Load a labelled evaluation set (JSONL with ground-truth `flagged` + `category`).
* Run the classifier (stub: random predictions) over each record.
* Report precision, recall, F1, and per-category breakdown.

Status: scaffold — replace `classify()` with the real model call and point
        `EVAL_MANIFEST` at a labelled dataset produced by build_dataset.py.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

EVAL_MANIFEST = (
    Path(__file__).resolve().parents[1] / "data" / "eval" / "labelled_manifest.jsonl"
)

CATEGORIES = ["explicit_visual", "violence", "hate_speech", "safe"]


# ---------------------------------------------------------------------------
# Stub classifier — replace with real model call
# ---------------------------------------------------------------------------

def classify(frame_reference: str, metadata: dict | None = None) -> dict:
    """
    Classify a frame and return an output-schema–compliant record.

    Returns
    -------
    dict with keys: flagged, category, confidence, timestamp, frame_reference
    """
    from datetime import datetime, timezone

    # --- stub: random prediction ---
    category = random.choice(CATEGORIES)
    confidence = round(random.uniform(0.5, 1.0), 4)
    flagged = category != "safe" and confidence >= 0.7

    return {
        "flagged": flagged,
        "category": category,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_reference": frame_reference,
    }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_metrics(
    records: Iterable[dict],
) -> dict:
    tp = fp = fn = 0
    per_cat: dict[str, dict] = {}

    for rec in records:
        gt_flagged = rec["ground_truth"]["flagged"]
        gt_cat = rec["ground_truth"]["category"]
        pred = classify(rec["frame_reference"], rec.get("metadata"))

        pred_flagged = pred["flagged"]
        pred_cat = pred["category"]

        # binary flag metrics
        if gt_flagged and pred_flagged:
            tp += 1
        elif not gt_flagged and pred_flagged:
            fp += 1
        elif gt_flagged and not pred_flagged:
            fn += 1

        # per-category tally
        per_cat.setdefault(gt_cat, {"correct": 0, "total": 0})
        per_cat[gt_cat]["total"] += 1
        if pred_cat == gt_cat:
            per_cat[gt_cat]["correct"] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "per_category": {
            cat: {
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
                **v,
            }
            for cat, v in per_cat.items()
        },
    }


def run_eval(manifest_path: Path = EVAL_MANIFEST) -> None:
    """Load the labelled manifest and print evaluation metrics."""
    if not manifest_path.exists():
        print(f"[baseline_eval] Manifest not found: {manifest_path}")
        print("  → Run build_dataset.py first and add ground-truth labels.")
        return

    with manifest_path.open() as fh:
        records = [json.loads(l) for l in fh if l.strip()]

    metrics = _compute_metrics(records)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run_eval()

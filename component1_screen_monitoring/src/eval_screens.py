"""
eval_screens.py — Evaluate the violence detector on the screen-style dataset.

Compares whole-screen inference ("full") with region inference ("tiles") on
every image under data/screens/:

    source      folder                              label
    scraped     scraped_pages/*.png                 safe (0)
    composite   composite/violence/*.png            violence (1)
                composite/non_violence/*.png        safe (0)
    recorded    recorded/violence/*.png             violence (1)
                recorded/safe/*.png                 safe (0)

"tiles+motion" needs a previous frame; for single images it is identical to
"tiles", so "tiles" is what is evaluated (reported as tiles / tiles+motion).

Writes results/screen_eval.json, results/screen_eval.md and
results/screen_eval_predictions.csv (one row per image per strategy).

Usage
-----
    python component1_screen_monitoring/src/eval_screens.py
    python component1_screen_monitoring/src/eval_screens.py --limit 50     # quick check
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.config import COMPONENT_ROOT, MODEL_DIR, SCREENS_DIR, THRESHOLD

STRATEGIES = ("full", "tiles")
SOURCES = ("scraped", "composite", "recorded")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
RESULTS_DIR = COMPONENT_ROOT / "results"

#: (source, sub-folder relative to data/screens, label)
LAYOUT = [
    ("scraped", "scraped_pages", 0),
    ("composite", "composite/violence", 1),
    ("composite", "composite/non_violence", 0),
    ("recorded", "recorded/violence", 1),
    ("recorded", "recorded/safe", 0),
]


def collect_dataset(root: Path = SCREENS_DIR) -> list[dict]:
    """Every labelled image under *root*: dicts with path, source, label."""
    items = []
    for source, sub, label in LAYOUT:
        folder = root / sub
        if folder.is_dir():
            items += [{"path": p, "source": source, "label": label}
                      for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return items


def compute_metrics(labels: list[int], preds: list[int]) -> dict:
    """Binary metrics with violence = positive. Undefined ratios are None."""
    tp = sum(1 for y, p in zip(labels, preds) if y and p)
    tn = sum(1 for y, p in zip(labels, preds) if not y and not p)
    fp = sum(1 for y, p in zip(labels, preds) if not y and p)
    fn = sum(1 for y, p in zip(labels, preds) if y and not p)
    n = len(labels)

    def ratio(a, b):
        return round(a / b, 4) if b else None

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn) if tp + fn else None   # = 2PR/(P+R); 0 if no TP
    return {
        "n": n, "positives": tp + fn, "negatives": tn + fp,
        "accuracy": ratio(tp + tn, n), "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": ratio(fp, fp + tn),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def evaluate(detector, items: list[dict], strategies=STRATEGIES) -> tuple[dict, list[dict]]:
    """Run every strategy on every item. Returns (metrics, prediction rows)."""
    rows = []
    for i, item in enumerate(items, 1):
        with Image.open(item["path"]) as img:
            img.load()
            for strategy in strategies:
                d = detector.predict_regions(img, None, strategy)
                rows.append({"file": str(item["path"]), "source": item["source"],
                             "label": item["label"], "strategy": strategy,
                             "p_violent": d.confidence, "pred": int(d.flagged),
                             "region": d.region, "n_crops": d.n_crops,
                             "inference_ms": d.inference_ms})
        if i % 25 == 0 or i == len(items):
            print(f"  {i}/{len(items)} images", flush=True)

    metrics: dict = {}
    for strategy in strategies:
        sr = [r for r in rows if r["strategy"] == strategy]
        per = {}
        for source in SOURCES + ("overall",):
            sel = sr if source == "overall" else [r for r in sr if r["source"] == source]
            if sel:
                m = compute_metrics([r["label"] for r in sel], [r["pred"] for r in sel])
                m["mean_inference_ms"] = round(sum(r["inference_ms"] for r in sel) / len(sel), 1)
                per[source] = m
        metrics[strategy] = per
    return metrics, rows


def _fmt(x) -> str:
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def to_markdown(result: dict) -> str:
    lines = [
        "# Screen-style evaluation — whole screen vs. region inference",
        "",
        f"Model: `{result['model_dir']}` · threshold {result['threshold']} · "
        f"device {result['device']} · {result['evaluated_at']}",
        "",
        "Violence = positive class. `tiles` = whole frame + 2×2 overlapping tiles + centre "
        "crop (identical to `tiles+motion` on single images). Precision / recall / F1 are "
        "— when undefined (e.g. the scraped pages contain no violent images).",
        "",
        "| Source | Strategy | n (viol/safe) | Accuracy | Precision | Recall | F1 | FPR | ms/img |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for source in SOURCES + ("overall",):
        for strategy, per in result["metrics"].items():
            m = per.get(source)
            if not m:
                continue
            lines.append(
                f"| {source} | {strategy} | {m['n']} ({m['positives']}/{m['negatives']}) | "
                f"{_fmt(m['accuracy'])} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | "
                f"{_fmt(m['f1'])} | {_fmt(m['false_positive_rate'])} | {m['mean_inference_ms']} |")
    lines += ["", "Counts per source: " + ", ".join(
        f"{k}: {v}" for k, v in result["counts"].items()), ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", type=Path, default=SCREENS_DIR)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate a random (seeded) subset of N images — quick check only")
    args = ap.parse_args()

    items = collect_dataset(args.root)
    if not items:
        sys.exit(f"No images under {args.root} — scrape / composite / record first.")
    if args.limit:
        items = random.Random(0).sample(items, min(args.limit, len(items)))

    from component1_screen_monitoring.src.detector import ViolenceDetector

    detector = ViolenceDetector(threshold=args.threshold)
    counts = {s: sum(1 for i in items if i["source"] == s) for s in SOURCES}
    print(f"Evaluating {len(items)} images {counts} on {detector.device} …")
    metrics, rows = evaluate(detector, items)

    result = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_dir": str(MODEL_DIR.relative_to(COMPONENT_ROOT.parent)),
        "threshold": args.threshold, "device": detector.device,
        "limit": args.limit, "counts": counts, "metrics": metrics,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "screen_eval.json").write_text(json.dumps(result, indent=2))
    (RESULTS_DIR / "screen_eval.md").write_text(to_markdown(result))
    with open(RESULTS_DIR / "screen_eval_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            r = {**r, "file": str(Path(r["file"]).relative_to(args.root))
                 if Path(r["file"]).is_relative_to(args.root) else r["file"]}
            w.writerow(r)
    print(to_markdown(result))
    print(f"Saved → results/screen_eval.json, screen_eval.md, screen_eval_predictions.csv")


if __name__ == "__main__":
    main()

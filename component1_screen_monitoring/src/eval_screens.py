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

Evaluates config.MODEL_DIR — the deployed model (V2, violence-vit-screen-ft-v2) —
unless --model-dir is given (repeatable, e.g. to compare P / V1 / V2). The model(s)
are printed before anything else runs.

Writes <stem>.json, <stem>.md (slide table), <stem>_predictions.csv (one row per
image × model × strategy) and <stem>_examples.png (8 misclassified examples;
violent ones blurred). Where, by default (never the results/ root):

    results/05_model_comparison/runs/<model>_t<thr>_<full|clean>.*
    (<model> = folder name(s) joined by "+"; with or without --model-dir)

(results/03_screen_eval_pretrained/ holds the 14 Oct V1 run and is not written to
by default.)

--out-dir / --run-name override this. Existing files are never overwritten
unless --overwrite is given (the experiment folders hold reported results).

--subset clean evaluates only the composites whose RLVS frame is from the val or
test split of results/01_rlvs_finetune/splits.csv (frames the checkpoint never
trained on), plus all plain scraped pages; recorded screens are left out (no
RLVS provenance).

Usage
-----
    python component1_screen_monitoring/src/eval_screens.py --limit 50 --out-dir /tmp/quick   # quick check
    python component1_screen_monitoring/src/eval_screens.py --subset clean \
        --model-dir component1_screen_monitoring/models/violence-vit-screen-ft-v2   # → 05_model_comparison/runs/
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.config import (COMPONENT_ROOT, MODEL_DIR, RESULTS_DIR,
                                                    RESULTS_MODEL_COMPARISON, SCREENS_DIR, SPLITS_CSV,
                                                    THRESHOLD)

STRATEGIES = ("full", "tiles")
SOURCES = ("scraped", "composite", "recorded")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
COMPARISON_RUNS_DIR = RESULTS_MODEL_COMPARISON / "runs"
OUTPUT_SUFFIXES = (".json", ".md", "_predictions.csv", "_examples.png")
SUBSETS = ("all", "clean")
CLEAN_RLVS_SPLITS = {"val", "test"}

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


def composite_rlvs_splits(composites_csv: Path, splits_csv: Path = SPLITS_CSV) -> dict[str, str]:
    """composites.csv filename ('violence/x_0.png') → RLVS split of its source frame."""
    split_of = {r["path"]: r["split"] for r in csv.DictReader(open(splits_csv))}
    out = {}
    for r in csv.DictReader(open(composites_csv)):
        _, cls, name = r["source_frame"].split("/")          # 'raw/violence/1045.jpg'
        out[r["filename"]] = split_of[f"{cls} /{name}"]       # splits.csv key 'violence /1045.jpg'
    return out


def filter_subset(items: list[dict], subset: str, root: Path = SCREENS_DIR,
                  splits_csv: Path = SPLITS_CSV) -> list[dict]:
    """
    'all' → unchanged. 'clean' → scraped pages + composites whose RLVS frame is in
    CLEAN_RLVS_SPLITS (not seen when the checkpoint was trained); recorded dropped.
    """
    if subset == "all":
        return items
    if subset != "clean":
        raise ValueError(f"unknown subset {subset!r} (choose from {SUBSETS})")
    split_of = composite_rlvs_splits(root / "composites.csv", splits_csv)
    out = []
    for item in items:
        if item["source"] == "scraped":
            out.append(item)
        elif item["source"] == "composite":
            key = Path(item["path"]).relative_to(root / "composite").as_posix()
            if key not in split_of:
                raise ValueError(f"{key} is not in composites.csv — cannot tell its RLVS split")
            if split_of[key] in CLEAN_RLVS_SPLITS:
                out.append(item)
    return out


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
        "| Source | Strategy | n (viol/safe) | Accuracy | Precision | Recall | F1 | Missed | False alarms | ms/img |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    if result.get("subset") == "clean":
        lines[2:2] = ["**Subset: clean** — only composites whose RLVS frame is from the RLVS "
                      "val/test split (never trained on), plus the plain scraped pages.", ""]
    for source in SOURCES + ("overall",):
        for strategy, per in result["metrics"].items():
            m = per.get(source)
            if not m:
                continue
            lines.append(
                f"| {source} | {strategy} | {m['n']} ({m['positives']}/{m['negatives']}) | "
                f"{_fmt(m['accuracy'])} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | "
                f"{_fmt(m['f1'])} | {m['confusion']['fn']} | {m['confusion']['fp']} | "
                f"{m['mean_inference_ms']} |")
    lines += ["", "Counts per source: " + ", ".join(
        f"{k}: {v}" for k, v in result["counts"].items()), ""]
    return "\n".join(lines)


def report_markdown(report: dict) -> str:
    """Full report: one table per model, then a timing summary."""
    parts = [to_markdown({**report, **m}) for m in report["models"]]
    lines = ["## Mean inference time per image", "",
             "| Model | full | tiles |", "|---|---|---|"]
    for m in report["models"]:
        ms = {s: m["metrics"][s]["overall"]["mean_inference_ms"] for s in m["metrics"]}
        lines.append(f"| `{m['model_dir']}` | {ms.get('full', '—')} ms | {ms.get('tiles', '—')} ms |")
    lines += ["", "Missed = violent images not flagged (false negatives). "
              "False alarms = safe images flagged (false positives).", ""]
    return "\n\n".join(parts) + "\n" + "\n".join(lines)


def pick_examples(rows: list[dict], n: int = 8, seed: int = 0) -> list[dict]:
    """
    Up to *n* misclassified predictions, about half missed violent and half false
    alarms, spread over sources/strategies, one per image.
    """
    rng = random.Random(seed)
    wrong = [r for r in rows if r["pred"] != r["label"]]
    fn = [r for r in wrong if r["label"] == 1]
    fp = [r for r in wrong if r["label"] == 0]
    for group in (fn, fp):
        rng.shuffle(group)
        group.sort(key=lambda r: (r["source"], r["strategy"]))   # stable: shuffled within groups
    def spread(group):
        buckets: dict = {}
        for r in group:
            buckets.setdefault((r["source"], r["strategy"]), []).append(r)
        out = []
        while any(buckets.values()):
            for k in list(buckets):
                if buckets[k]:
                    out.append(buckets[k].pop())
        return out
    fn, fp = spread(fn), spread(fp)
    picked, seen = [], set()
    for r in [x for pair in zip(fn, fp) for x in pair] + fn + fp:
        if r["file"] not in seen and len(picked) < n:
            picked.append(r)
            seen.add(r["file"])
    return picked


def save_examples(examples: list[dict], path: Path, cols: int = 4, tile=(400, 250)) -> None:
    """Grid of examples; every violent (label 1) image is heavily blurred."""
    rows_n = max(1, -(-len(examples) // cols))
    caption_h = 34
    sheet = Image.new("RGB", (cols * tile[0], rows_n * (tile[1] + caption_h)), "white")
    for i, ex in enumerate(examples):
        with Image.open(ex["file"]) as im:
            thumb = ImageOps.fit(im.convert("RGB"), tile)
        if ex["label"] == 1:
            thumb = thumb.filter(ImageFilter.GaussianBlur(14))
        x, y = (i % cols) * tile[0], (i // cols) * (tile[1] + caption_h)
        sheet.paste(thumb, (x, y))
        kind = "MISSED violent (blurred)" if ex["label"] == 1 else "FALSE ALARM on safe"
        d = ImageDraw.Draw(sheet)
        d.rectangle([x, y + tile[1], x + tile[0] - 2, y + tile[1] + caption_h - 2],
                    fill="#b00020" if ex["label"] == 1 else "#8a5a00")
        d.text((x + 6, y + tile[1] + 3), f"{kind} · {ex['source']} · {ex['strategy']}", fill="white")
        d.text((x + 6, y + tile[1] + 17), f"p={ex['p_violent']:.3f} · {Path(ex['file']).name}"[:60],
               fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def resolve_model_dirs(model_dirs: list[Path] | None) -> list[Path]:
    """--model-dir values, or the deployed model (config.MODEL_DIR) when none are given."""
    return [Path(d) for d in model_dirs] if model_dirs else [MODEL_DIR]


def default_output(model_dirs: list[Path], threshold: float, subset: str) -> tuple[Path, str]:
    """(folder, file stem) for a run when --out-dir / --run-name are not given."""
    tag = "full" if subset == "all" else subset
    names = "+".join(Path(d).name for d in model_dirs)
    return COMPARISON_RUNS_DIR, f"{names}_t{threshold:.2f}_{tag}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", type=Path, default=SCREENS_DIR)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate a random (seeded) subset of N images — quick check only")
    ap.add_argument("--model-dir", type=Path, action="append", default=None,
                    help="model directory to evaluate (repeatable; default: config.MODEL_DIR, "
                         f"the deployed model — {MODEL_DIR.name})")
    ap.add_argument("--subset", choices=SUBSETS, default="all",
                    help="'clean' = only composites with RLVS val/test frames + scraped pages")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="folder for the output files (default: results/05_model_comparison/runs/)")
    ap.add_argument("--run-name", default=None,
                    help="output file stem (default: <model>_t<thr>_<full|clean>)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = ap.parse_args()
    model_dirs = resolve_model_dirs(args.model_dir)
    for d in model_dirs:
        role = " (config.MODEL_DIR — the deployed model)" if d.resolve() == MODEL_DIR.resolve() else ""
        print(f"Evaluating model: {d.name}{role}\n  {d}", flush=True)
    out_dir, stem = default_output(model_dirs, args.threshold, args.subset)
    out_dir, stem = args.out_dir or out_dir, args.run_name or stem
    if out_dir.resolve() == RESULTS_DIR.resolve():
        sys.exit("Write runs into an experiment folder, not the results/ root (see results/README.md).")
    existing = [out_dir / f"{stem}{s}" for s in OUTPUT_SUFFIXES if (out_dir / f"{stem}{s}").exists()]
    if existing and not args.overwrite:
        sys.exit(f"{existing[0]} already exists — pass --overwrite, or choose --out-dir / --run-name.")
    print(f"  threshold {args.threshold} · subset {args.subset} · output {out_dir / stem}.*", flush=True)
    missing = [d for d in model_dirs if not (Path(d) / "config.json").exists()]
    if missing:
        sys.exit(f"Not a Hugging Face model directory (no config.json): {missing}")

    items = filter_subset(collect_dataset(args.root), args.subset, args.root)
    if not items:
        sys.exit(f"No images under {args.root} — scrape / composite / record first.")
    if args.limit:
        items = random.Random(0).sample(items, min(args.limit, len(items)))

    from component1_screen_monitoring.src.detector import ViolenceDetector

    counts = {s: sum(1 for i in items if i["source"] == s) for s in SOURCES}
    models, all_rows = [], []
    for model_dir in model_dirs:
        detector = ViolenceDetector(model_dir, threshold=args.threshold)
        name = (str(model_dir.resolve().relative_to(COMPONENT_ROOT.parent))
                if model_dir.resolve().is_relative_to(COMPONENT_ROOT.parent) else str(model_dir))
        print(f"[{name}] evaluating {len(items)} images {counts} on {detector.device} …")
        detector.predict_regions(Image.new("RGB", (1280, 800)), None, "tiles")   # warm-up
        metrics, rows = evaluate(detector, items)
        models.append({"model_dir": name, "device": detector.device, "metrics": metrics})
        all_rows += [{"model": name, **r} for r in rows]
        del detector

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": args.threshold, "limit": args.limit, "subset": args.subset,
        "counts": counts, "models": models,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2))
    md = report_markdown(report)
    (out_dir / f"{stem}.md").write_text(md)
    with open(out_dir / f"{stem}_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        for r in all_rows:
            r = {**r, "file": str(Path(r["file"]).relative_to(args.root))
                 if Path(r["file"]).is_relative_to(args.root) else r["file"]}
            w.writerow(r)
    examples = pick_examples([r for r in all_rows if r["model"] == models[-1]["model_dir"]])
    save_examples(examples, out_dir / f"{stem}_examples.png")
    print(md)
    print(f"Saved → {out_dir}/{stem}.json, {stem}.md, {stem}_predictions.csv, "
          f"{stem}_examples.png ({len(examples)} examples)")


if __name__ == "__main__":
    main()

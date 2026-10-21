"""
compare_models.py — Model A (config.MODEL_DIR) vs model B (screen fine-tune v2).

Reads the eval_screens.py runs in results/05_model_comparison/runs/ (one per model × threshold ×
subset, both strategies), then adds:

  * recall by pasted-video size (under30 / 30-45 / 45plus, as in make_composites.py)
  * the RLVS TEST split (results/01_rlvs_finetune/splits.csv, held out for both models): accuracy,
    precision, recall, F1 on the full frames
  * ms/img on this machine, full frame and tiles (same pages for both models)
  * the decision rule (see DECISION below)

Writes results/05_model_comparison/model_comparison.md and model_comparison.json.

Run the screen evaluations first (from the repo root):
    python component1_screen_monitoring/src/eval_screens.py --model-dir <A> --threshold 0.5 \
        --run-name A_t0.50_full        # --model-dir → 05_model_comparison/runs/; … and _clean, B_t0.50_*, B_t0.25_*
    python component1_screen_monitoring/src/compare_models.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.config import (COMPONENT_ROOT, MODEL_DIR, RESULTS_MODEL_COMPARISON,
                                                    SCREENS_DIR, SPLITS_CSV)
from component1_screen_monitoring.src.eval_screens import COMPARISON_RUNS_DIR, compute_metrics
from component1_screen_monitoring.src.make_composites import size_bucket

COMPARE_DIR = COMPARISON_RUNS_DIR
RLVS_ROOT = Path.home() / "Desktop" / "new_violence"
MODELS = {"A": MODEL_DIR, "B": COMPONENT_ROOT / "models" / "violence-vit-screen-ft-v2"}
RUNS = [("A", 0.5), ("B", 0.5), ("B", 0.25)]            # (model, threshold)
SUBSETS = [("full", "full test set (520)"), ("clean", "clean subset")]
BUCKETS = ("under30", "30-45", "45plus")
STRATEGIES = ("full", "tiles")
#: DECISION: B only if, with tiles, recall is higher than A's on BOTH sets, its false-alarm
#: rate is at most 2x A's (on both sets) and its RLVS test accuracy is >= 98 %.
MAX_FPR_RATIO, MIN_RLVS_ACC = 2.0, 0.98


def run_name(model: str, thr: float, subset: str) -> str:
    return f"{model}_t{thr:.2f}_{subset}"


def load_run(name: str) -> tuple[dict, list[dict]]:
    report = json.loads((COMPARE_DIR / f"{name}.json").read_text())
    rows = list(csv.DictReader(open(COMPARE_DIR / f"{name}_predictions.csv")))
    return report, rows


def composite_buckets(root: Path = SCREENS_DIR) -> dict[str, str]:
    """'composite/violence/x_0.png' → size bucket of its pasted frame."""
    out = {}
    for r in csv.DictReader(open(root / "composites.csv")):
        box = tuple(int(r[k]) for k in ("box_x0", "box_y0", "box_x1", "box_y1"))
        with Image.open(root / "composite" / r["filename"]) as im:
            width = im.width
        out[f"composite/{r['filename']}"] = size_bucket(box, width)
    return out


def recall_by_size(rows: list[dict], buckets: dict[str, str], strategy: str) -> dict[str, dict]:
    """Recall on the violent composites per size bucket: {bucket: {n, hit, recall}}."""
    out = {}
    for b in BUCKETS:
        sel = [r for r in rows if r["strategy"] == strategy and r["source"] == "composite"
               and r["label"] == "1" and buckets.get(r["file"]) == b]
        hit = sum(r["pred"] == "1" for r in sel)
        out[b] = {"n": len(sel), "hit": hit, "recall": round(hit / len(sel), 4) if sel else None}
    return out


def rlvs_test_probs(model_dir: Path, batch: int = 32) -> tuple[list[int], list[float]]:
    """P(violent) on every RLVS test frame, full frame, with the detector's own preprocessing."""
    from component1_screen_monitoring.src.detector import ViolenceDetector
    det = ViolenceDetector(model_dir)
    rows = [r for r in csv.DictReader(open(SPLITS_CSV)) if r["split"] == "test"]
    labels, probs = [int(r["label"]) for r in rows], []
    torch = det._torch
    for i in range(0, len(rows), batch):
        imgs = []
        for r in rows[i:i + batch]:
            cls, name = r["path"].split(" /")
            with Image.open(RLVS_ROOT / cls / name) as im:
                imgs.append(im.convert("RGB"))
        x = det.processor(images=imgs, return_tensors="pt").to(det.device)
        with torch.inference_mode():
            p = det.model(**x).logits.float().softmax(-1)[:, det.violent_idx].cpu()
        probs += p.tolist()
    return labels, probs


def benchmark(model_dir: Path, pages: list[Path], warmup: int = 5, reps: int = 3) -> dict:
    """Per-image inference ms (preprocess + forward) on *pages*, both strategies."""
    from component1_screen_monitoring.src.detector import ViolenceDetector
    det = ViolenceDetector(model_dir)
    imgs = [Image.open(p).convert("RGB") for p in pages]
    out = {"device": det.device, "images": len(imgs) * reps}
    for strategy in STRATEGIES:
        for im in imgs[:warmup]:
            det.predict_regions(im, None, strategy)
        ms = [det.predict_regions(im, None, strategy).inference_ms for _ in range(reps) for im in imgs]
        out[strategy] = {"mean": round(statistics.mean(ms), 1), "median": round(statistics.median(ms), 1),
                         "p90": round(sorted(ms)[int(0.9 * (len(ms) - 1))], 1)}
    return out


def decide(table: dict, rlvs: dict) -> dict:
    """Apply DECISION to each B threshold; recommend one (or keep A)."""
    a = {s: table[("A", 0.5, s, "tiles")] for s, _ in SUBSETS}
    verdicts = {}
    for thr in sorted({t for m, t in RUNS if m == "B"}, reverse=True):
        b = {s: table[("B", thr, s, "tiles")] for s, _ in SUBSETS}
        checks = {}
        for s, _ in SUBSETS:
            checks[f"recall_higher_{s}"] = b[s]["recall"] > a[s]["recall"]
            limit = MAX_FPR_RATIO * a[s]["false_positive_rate"]
            checks[f"fpr_within_2x_{s}"] = b[s]["false_positive_rate"] <= limit + 1e-12
        checks["rlvs_test_acc_ge_98"] = rlvs[("B", thr)]["accuracy"] >= MIN_RLVS_ACC
        verdicts[thr] = {"passes": all(checks.values()), "checks": checks,
                         "tiles_f1_mean": round(statistics.mean(b[s]["f1"] or 0 for s, _ in SUBSETS), 4)}
    passing = [t for t, v in verdicts.items() if v["passes"]]
    # among passing thresholds: best mean tiles F1 (full + clean); tie → the higher threshold (fewer alarms)
    choice = max(passing, key=lambda t: (verdicts[t]["tiles_f1_mean"], t)) if passing else None
    return {"verdicts": verdicts, "choose": "B" if choice is not None else "A", "threshold": choice}


def fmt(x, nd=3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> int:
    for m, d in MODELS.items():
        if not (d / "config.json").exists():
            sys.exit(f"model {m}: no config.json in {d}")
    missing = [run_name(m, t, s) for m, t in RUNS for s, _ in SUBSETS
               if not (COMPARE_DIR / f"{run_name(m, t, s)}.json").exists()]
    if missing:
        sys.exit(f"missing eval_screens runs in {COMPARE_DIR}: {missing}")

    buckets = composite_buckets()
    table, sizes, counts = {}, {}, {}
    for m, t in RUNS:
        for s, _ in SUBSETS:
            report, rows = load_run(run_name(m, t, s))
            assert abs(report["threshold"] - t) < 1e-9 and len(report["models"]) == 1
            counts[s] = report["counts"]
            for strat in STRATEGIES:
                table[(m, t, s, strat)] = report["models"][0]["metrics"][strat]["overall"]
                sizes[(m, t, s, strat)] = recall_by_size(rows, buckets, strat)

    print("RLVS test split …")
    rlvs, rlvs_n = {}, None
    for m in MODELS:
        labels, probs = rlvs_test_probs(MODELS[m])
        rlvs_n = len(labels)
        for mm, t in RUNS:
            if mm == m:
                rlvs[(m, t)] = compute_metrics(labels, [int(p >= t) for p in probs])

    print("Latency …")
    pages = sorted((SCREENS_DIR / "scraped_pages").glob("*.png"))[::10][:10]
    lat = {m: benchmark(MODELS[m], pages) for m in MODELS}

    decision = decide(table, rlvs)
    name_of = {m: str(d.relative_to(COMPONENT_ROOT)) for m, d in MODELS.items()}

    L = ["# Model comparison — A (current) vs B (screen fine-tune v2)", "",
         f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `src/compare_models.py`.", "",
         f"- **A** = `{name_of['A']}` (`config.MODEL_DIR`): `jaranohaal/vit-base-violence-detection` "
         "fine-tuned on full-frame RLVS images (Week 2).",
         f"- **B** = `{name_of['B']}`: the same pretrained checkpoint fine-tuned on screen composites + tiles "
         "with 40 % RLVS frames (notebook section 15; run u4_lr3e-5). Threshold 0.25 = B's max-F1 threshold "
         "on screen **val**.", "",
         "## How to read this", "",
         "- The **screen test set was never used for training or model selection** — B's run and its 0.25 "
         "threshold were chosen on screen val; these are the first and only test numbers.",
         f"- The **full test set** ({sum(counts['full'].values())} images: "
         f"{counts['full']['composite']} composites + {counts['full']['scraped']} plain pages) reuses RLVS "
         "frames that were in the RLVS **training** split (203 of its 300 frames), i.e. frames A was trained on. "
         "B never saw them (its training excluded every frame near a test frame), so the full set slightly "
         "favours A.",
         f"- The **clean subset** ({sum(counts['clean'].values())} images: {counts['clean']['composite']} composites "
         f"with RLVS val/test frames + {counts['clean']['scraped']} plain pages) uses no training frames of either "
         "model — the fairer comparison, but small: one violent image ≈ 1.6 recall points.",
         "- `tiles` = whole frame + 2×2 overlapping tiles + centre crop, flagged if any crop ≥ threshold "
         "(the live strategy); `full` = whole frame only. Missed = violent images not flagged; "
         "false alarms = safe images flagged. FPR = false alarms / safe images.",
         f"- ms/img = preprocess + forward pass on this laptop ({lat['A']['device']}), median over "
         f"{lat['A']['images']} runs on 1280×800 page screenshots; it does not depend on the threshold.", "",
         "## Screen test set", "",
         "| model | threshold | test set | strategy | accuracy | precision | recall | F1 | missed | false alarms | FPR | ms/img |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, label in SUBSETS:
        for strat in STRATEGIES:
            for m, t in RUNS:
                r = table[(m, t, s, strat)]
                L.append(f"| {m} | {t:.2f} | {label} | {strat} | {fmt(r['accuracy'])} | {fmt(r['precision'])} | "
                         f"{fmt(r['recall'])} | {fmt(r['f1'])} | {r['confusion']['fn']} / {r['positives']} | "
                         f"{r['confusion']['fp']} / {r['negatives']} | {fmt(r['false_positive_rate'])} | "
                         f"{lat[m][strat]['median']} |")
    L += ["", "## Recall by video size (violent composites)", "",
          "Size = pasted video width / page width.", "",
          "| model | threshold | test set | strategy | " + " | ".join(
              {"under30": "under 30%", "30-45": "30–45%", "45plus": "45%+"}[b] for b in BUCKETS) + " |",
          "|---|---|---|---|" + "---|" * len(BUCKETS)]
    for s, label in SUBSETS:
        for strat in STRATEGIES:
            for m, t in RUNS:
                z = sizes[(m, t, s, strat)]
                L.append(f"| {m} | {t:.2f} | {label} | {strat} | " + " | ".join(
                    f"{fmt(z[b]['recall'])} ({z[b]['hit']}/{z[b]['n']})" for b in BUCKETS) + " |")
    L += ["", f"## RLVS test split (held-out full frames, n = {rlvs_n})", "",
          "| model | threshold | accuracy | precision | recall | F1 | missed | false alarms |",
          "|---|---|---|---|---|---|---|---|"]
    for m, t in RUNS:
        r = rlvs[(m, t)]
        L.append(f"| {m} | {t:.2f} | {fmt(r['accuracy'], 4)} | {fmt(r['precision'], 4)} | {fmt(r['recall'], 4)} | "
                 f"{fmt(r['f1'], 4)} | {r['confusion']['fn']} | {r['confusion']['fp']} |")
    L += ["", "## Speed on this laptop", "", "| model | device | full: median / mean / p90 ms | tiles: median / mean / p90 ms |",
          "|---|---|---|---|"]
    for m in MODELS:
        f_, t_ = lat[m]["full"], lat[m]["tiles"]
        L.append(f"| {m} | {lat[m]['device']} | {f_['median']} / {f_['mean']} / {f_['p90']} | "
                 f"{t_['median']} / {t_['mean']} / {t_['p90']} |")
    L += ["", "## Decision", "",
          "Rule: choose B only if, **with tiles**, its recall is higher than A's on **both** the full set and the "
          f"clean subset, its false-alarm rate is at most {MAX_FPR_RATIO:g}× A's (on both), and its RLVS test "
          f"accuracy is ≥ {MIN_RLVS_ACC:.0%}. Otherwise keep A.", "",
          "| B threshold | recall > A (full) | recall > A (clean) | FPR ≤ 2×A (full) | FPR ≤ 2×A (clean) | RLVS acc ≥ 98% | passes |",
          "|---|---|---|---|---|---|---|"]
    yn = lambda v: "yes" if v else "**no**"
    for t, v in decision["verdicts"].items():
        c = v["checks"]
        L.append(f"| {t:.2f} | {yn(c['recall_higher_full'])} | {yn(c['recall_higher_clean'])} | "
                 f"{yn(c['fpr_within_2x_full'])} | {yn(c['fpr_within_2x_clean'])} | {yn(c['rlvs_test_acc_ge_98'])} | "
                 f"{yn(v['passes'])} |")
    L += ["", ("**Result: choose B at threshold "
               f"{decision['threshold']:.2f}.**" if decision["choose"] == "B" else "**Result: keep A.**"), ""]
    (RESULTS_MODEL_COMPARISON / "model_comparison.md").write_text("\n".join(L))

    key = lambda k: "|".join(str(x) for x in k)
    (RESULTS_MODEL_COMPARISON / "model_comparison.json").write_text(json.dumps({
        "models": name_of, "runs": RUNS, "counts": counts,
        "screen": {key(k): v for k, v in table.items()}, "recall_by_size": {key(k): v for k, v in sizes.items()},
        "rlvs_test": {key(k): v for k, v in rlvs.items()}, "latency": lat,
        "decision": {**decision, "verdicts": {str(t): v for t, v in decision["verdicts"].items()}},
    }, indent=2))
    print("\n".join(L))
    print("\nSaved → results/05_model_comparison/model_comparison.md, model_comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

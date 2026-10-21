"""
compare_models.py — P (true pretrained) vs V1 (RLVS fine-tune) vs V2 (screen fine-tune).

    P  = models/violence-vit-pretrained-hf   jaranohaal/vit-base-violence-detection, no training of
                                             ours (config.PRETRAINED_MODEL_DIR; src/convert_pretrained.py)
    V1 = models/violence-vit-screen-ft-v1    P fine-tuned on full-frame RLVS images (notebook §7;
                                             config.RLVS_FT_V1_MODEL_DIR; labelled "A" in the 21 Oct runs)
    V2 = models/violence-vit-screen-ft-v2    P fine-tuned on screen composites + tiles (notebook §15;
                                             config.MODEL_DIR; labelled "B" in the 21 Oct runs)

Reads the eval_screens.py runs in results/05_model_comparison/runs/ (one per model ×
threshold × subset, both strategies; file prefixes P_, A_ (=V1) and B_ (=V2)), then adds:

  * recall by pasted-video size (under30 / 30-45 / 45plus, as in make_composites.py)
  * the RLVS TEST split (results/01_rlvs_finetune/splits.csv): accuracy, precision, recall, F1
  * ms/img on this machine, full frame and tiles (same pages for every model)
  * the pre-set rule (V2 + tiles vs V1 + tiles) and the deployment check
    (V2 whole screen vs the previously deployed V1 + tiles)

RLVS-test and ms/img measurements already in model_comparison.json are reused (matched by
model folder), so re-running never changes existing numbers; only models without a
measurement are measured. --remeasure measures every model again.

Writes results/05_model_comparison/model_comparison.md and model_comparison.json.

Run the screen evaluations first (from the repo root), e.g.
    python component1_screen_monitoring/src/eval_screens.py \
        --model-dir component1_screen_monitoring/models/violence-vit-pretrained-hf --run-name P_t0.50_full
    python component1_screen_monitoring/src/compare_models.py
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.config import (COMPONENT_ROOT, PRETRAINED_MODEL_DIR,
                                                    RESULTS_MODEL_COMPARISON, RLVS_FT_V1_MODEL_DIR,
                                                    SCREENS_DIR, SPLITS_CSV)
from component1_screen_monitoring.src.eval_screens import COMPARISON_RUNS_DIR, compute_metrics
from component1_screen_monitoring.src.make_composites import size_bucket

COMPARE_DIR = COMPARISON_RUNS_DIR
REPORT_JSON = RESULTS_MODEL_COMPARISON / "model_comparison.json"
RLVS_ROOT = Path.home() / "Desktop" / "new_violence"
MODELS = {"P": PRETRAINED_MODEL_DIR, "V1": RLVS_FT_V1_MODEL_DIR,
          "V2": COMPONENT_ROOT / "models" / "violence-vit-screen-ft-v2"}
#: Run-file prefix per model: the 21 Oct runs were saved as A_ (= V1) and B_ (= V2).
RUN_PREFIX = {"P": "P", "V1": "A", "V2": "B"}
#: (model, threshold). V2 @ 0.25 (its max-F1 threshold on screen val) is kept for reference.
RUNS = [("P", 0.5), ("V1", 0.5), ("V2", 0.5), ("V2", 0.25)]
SUBSETS = [("full", "full test set (520)"), ("clean", "clean subset")]
BUCKETS = ("under30", "30-45", "45plus")
STRATEGIES = ("full", "tiles")
#: Pre-set rule: V2 + tiles only if, vs V1 + tiles, recall is higher on BOTH sets, the
#: false-alarm rate is at most 2x V1's (on both) and RLVS test accuracy is >= 98 %.
MAX_FPR_RATIO, MIN_RLVS_ACC = 2.0, 0.98
BASELINE = ("V1", 0.5, "tiles")      # previously deployed: V1, tiles(+motion — same as tiles on single images)
DEPLOY = ("V2", 0.5, "full")         # new default mode
STRICT = ("V2", 0.5, "tiles")        # new strict mode


def run_name(model: str, thr: float, subset: str) -> str:
    return f"{RUN_PREFIX[model]}_t{thr:.2f}_{subset}"


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


def previous_measurements(report: dict) -> dict[str, dict]:
    """
    {model folder name: {"rlvs": {threshold: metrics}, "latency": {...}}} from an earlier
    model_comparison.json — the current format ("measurements") or the 21 Oct A/B format.
    """
    if "measurements" in report:
        return {name: {"rlvs": {float(t): m for t, m in v["rlvs"].items()}, "latency": v["latency"]}
                for name, v in report["measurements"].items()}
    out = {}
    for label, path in report.get("models", {}).items():          # 21 Oct: {"A": "models/…-v1", "B": …}
        rlvs = {float(k.split("|")[1]): m for k, m in report.get("rlvs_test", {}).items()
                if k.split("|")[0] == label}
        out[Path(path).name] = {"rlvs": rlvs, "latency": report["latency"][label]}
    return out


def decide(table: dict, rlvs: dict) -> dict:
    """Pre-set rule: V2 + tiles vs V1 + tiles, for each V2 threshold."""
    base = {s: table[("V1", 0.5, s, "tiles")] for s, _ in SUBSETS}
    verdicts = {}
    for thr in sorted({t for m, t in RUNS if m == "V2"}, reverse=True):
        cand = {s: table[("V2", thr, s, "tiles")] for s, _ in SUBSETS}
        checks = {}
        for s, _ in SUBSETS:
            checks[f"recall_higher_{s}"] = cand[s]["recall"] > base[s]["recall"]
            checks[f"fpr_within_2x_{s}"] = (cand[s]["false_positive_rate"]
                                           <= MAX_FPR_RATIO * base[s]["false_positive_rate"] + 1e-12)
        checks["rlvs_test_acc_ge_98"] = rlvs[("V2", thr)]["accuracy"] >= MIN_RLVS_ACC
        verdicts[thr] = {"passes": all(checks.values()), "checks": checks,
                         "tiles_f1_mean": round(statistics.mean(cand[s]["f1"] or 0 for s, _ in SUBSETS), 4)}
    passing = [t for t, v in verdicts.items() if v["passes"]]
    choice = max(passing, key=lambda t: (verdicts[t]["tiles_f1_mean"], t)) if passing else None
    return {"verdicts": verdicts, "choose": "V2" if choice is not None else "V1", "threshold": choice}


def deployment_check(table: dict, rlvs: dict, lat: dict) -> dict:
    """V2 whole screen (DEPLOY) vs the previously deployed V1 + tiles (BASELINE)."""
    (bm, bt, bs), (dm, dt, ds) = BASELINE, DEPLOY
    checks, detail = {}, {}
    for s, _ in SUBSETS:
        b, d = table[(bm, bt, s, bs)], table[(dm, dt, s, ds)]
        checks[f"recall_higher_{s}"] = d["recall"] > b["recall"]
        ratio = d["false_positive_rate"] / b["false_positive_rate"] if b["false_positive_rate"] else float("inf")
        checks[f"fpr_within_2x_{s}"] = ratio <= MAX_FPR_RATIO + 1e-12
        detail[s] = {"recall": (b["recall"], d["recall"]), "fpr": (b["false_positive_rate"], d["false_positive_rate"]),
                     "fpr_ratio": round(ratio, 2)}
    checks["rlvs_test_acc_ge_98"] = rlvs[(dm, dt)]["accuracy"] >= MIN_RLVS_ACC
    speedup = lat[bm][bs]["median"] / lat[dm][ds]["median"]
    checks["faster"] = speedup > 1
    return {"passes": all(checks.values()), "checks": checks, "detail": detail,
            "rlvs_acc": rlvs[(dm, dt)]["accuracy"], "speedup": round(speedup, 1),
            "ms": (lat[bm][bs]["median"], lat[dm][ds]["median"])}


def fmt(x, nd=3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--remeasure", action="store_true",
                    help="measure RLVS test + ms/img for every model again (default: reuse saved values)")
    args = ap.parse_args()
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
            used = Path(report["models"][0]["model_dir"]).name
            assert abs(report["threshold"] - t) < 1e-9 and len(report["models"]) == 1
            assert used == MODELS[m].name, f"{run_name(m, t, s)} was made with {used}, not {MODELS[m].name}"
            counts[s] = report["counts"]
            for strat in STRATEGIES:
                table[(m, t, s, strat)] = report["models"][0]["metrics"][strat]["overall"]
                sizes[(m, t, s, strat)] = recall_by_size(rows, buckets, strat)

    prev = {} if args.remeasure or not REPORT_JSON.exists() else previous_measurements(json.loads(REPORT_JSON.read_text()))
    pages = sorted((SCREENS_DIR / "scraped_pages").glob("*.png"))[::10][:10]
    rlvs, lat, measured, rlvs_n = {}, {}, {}, 1661
    for m, d in MODELS.items():
        thrs = [t for mm, t in RUNS if mm == m]
        old = prev.get(d.name)
        if old and all(t in old["rlvs"] for t in thrs):
            lat[m] = old["latency"]
            for t in thrs:
                rlvs[(m, t)] = old["rlvs"][t]
            measured[m] = "reused"
        else:
            print(f"{m}: RLVS test split + latency …", flush=True)
            labels, probs = rlvs_test_probs(d)
            rlvs_n = len(labels)
            for t in thrs:
                rlvs[(m, t)] = compute_metrics(labels, [int(p >= t) for p in probs])
            lat[m] = benchmark(d, pages)
            measured[m] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    decision = decide(table, rlvs)
    deploy = deployment_check(table, rlvs, lat)
    name_of = {m: str(d.relative_to(COMPONENT_ROOT)) for m, d in MODELS.items()}
    when = {m: ("21 Oct" if v == "reused" else v[:10]) for m, v in measured.items()}

    L = ["# Model comparison — P (pretrained) vs V1 (RLVS fine-tune) vs V2 (screen fine-tune)", "",
         f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `src/compare_models.py`.", "",
         f"- **P** = `{name_of['P']}`: the pretrained `jaranohaal/vit-base-violence-detection` (ViT-base, "
         "trained by its author on RLVS frames), converted to Hugging Face format — no training of ours.",
         f"- **V1** = `{name_of['V1']}`: P fine-tuned on full-frame RLVS images (notebook §7). Deployed until "
         "the switch to V2. *Labelled “A” in the first version of this report (21 Oct) and in the `runs/A_*` files.*",
         f"- **V2** = `{name_of['V2']}`: P fine-tuned on screen composites + tiles with 40 % RLVS frames "
         "(notebook §15; run u4_lr3e-5). Now `config.MODEL_DIR`. *Labelled “B” in the 21 Oct version and in "
         "`runs/B_*`.* Threshold 0.25 = V2's max-F1 threshold on screen **val**, kept for reference.", "",
         "## How to read this", "",
         "- The **screen test set was never used for training or model selection** — V2's run and its 0.25 "
         "threshold were chosen on screen val.",
         f"- The **full test set** ({sum(counts['full'].values())} images: "
         f"{counts['full']['composite']} composites + {counts['full']['scraped']} plain pages) reuses RLVS "
         "frames from the RLVS **training** split (203 of its 300 frames), which V1 was trained on; V2's "
         "training excluded every frame near a test frame. So the full set slightly favours V1.",
         f"- The **clean subset** ({sum(counts['clean'].values())} images: {counts['clean']['composite']} composites "
         f"with RLVS val/test frames + {counts['clean']['scraped']} plain pages) uses no training frames of V1 or "
         "V2 — the fairer comparison, but small: one violent image ≈ 1.6 recall points.",
         "- **P's own training data is only documented as “RLVS frames”**, so P may have seen any of these "
         "frames (screen test, clean subset and RLVS test alike); its numbers are an upper bound for “no "
         "screen training”.",
         "- `tiles` = whole frame + 2×2 overlapping tiles + centre crop, flagged if any crop ≥ threshold "
         "(strict mode; the previously deployed `tiles+motion` is identical on single images); `full` = whole "
         "frame only (default mode). Missed = violent images not flagged; false alarms = safe images flagged; "
         "FPR = false alarms / safe images.",
         f"- ms/img = preprocess + forward pass on this laptop ({lat['V2']['device']}), median over "
         f"{lat['V2']['images']} runs on 1280×800 page screenshots; independent of the threshold. Measured "
         + ", ".join(f"{m} {w}" for m, w in when.items()) + ".", "",
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

    yn = lambda v: "yes" if v else "**no**"
    L += ["", "## Pre-set rule (tiles vs tiles)", "",
          "Set before the test run: choose V2 + tiles only if, vs V1 + tiles, its recall is higher on **both** "
          f"the full set and the clean subset, its false-alarm rate is at most {MAX_FPR_RATIO:g}× V1's (on both), "
          f"and its RLVS test accuracy is ≥ {MIN_RLVS_ACC:.0%}.", "",
          "| V2 threshold | recall > V1 (full) | recall > V1 (clean) | FPR ≤ 2×V1 (full) | FPR ≤ 2×V1 (clean) | RLVS acc ≥ 98% | passes |",
          "|---|---|---|---|---|---|---|"]
    for t, v in decision["verdicts"].items():
        c = v["checks"]
        L.append(f"| {t:.2f} | {yn(c['recall_higher_full'])} | {yn(c['recall_higher_clean'])} | "
                 f"{yn(c['fpr_within_2x_full'])} | {yn(c['fpr_within_2x_clean'])} | {yn(c['rlvs_test_acc_ge_98'])} | "
                 f"{yn(v['passes'])} |")
    L += ["", ("**Rule result: V2 + tiles is rejected** — its false-alarm rate is more than 2× V1 + tiles."
               if decision["choose"] == "V1" else
               f"**Rule result: V2 + tiles passes at threshold {decision['threshold']:.2f}.**")]

    dd, dc = deploy["detail"], deploy["checks"]
    L += ["", "## Deployment decision", "",
          "Compared with the **previously deployed setup — V1 + tiles at 0.5** — **V2 on the whole screen "
          "(`full`, 0.5)**:", "",
          "| check | V1 + tiles | V2 whole screen | passes |", "|---|---|---|---|"]
    for s, label in SUBSETS:
        b, d = dd[s]["recall"]
        L.append(f"| recall, {label} | {b:.3f} | {d:.3f} | {yn(dc[f'recall_higher_{s}'])} |")
    for s, label in SUBSETS:
        b, d = dd[s]["fpr"]
        L.append(f"| false-alarm rate, {label} (≤ 2×) | {b:.3f} | {d:.3f} ({dd[s]['fpr_ratio']:.2f}×) | "
                 f"{yn(dc[f'fpr_within_2x_{s}'])} |")
    L.append(f"| RLVS test accuracy (≥ 98 %) | {rlvs[(BASELINE[0], BASELINE[1])]['accuracy']:.4f} | "
             f"{deploy['rlvs_acc']:.4f} | {yn(dc['rlvs_test_acc_ge_98'])} |")
    L.append(f"| ms/img (median) | {deploy['ms'][0]} | {deploy['ms'][1]} ({deploy['speedup']:.1f}× faster) | "
             f"{yn(dc['faster'])} |")
    if deploy["passes"]:
        L += ["", "The pre-set rule compared tiles with tiles and rejected V2 + tiles on false alarms. "
              "**V2 on the whole screen is the default mode**: against the previously deployed V1 + tiles it has "
              "higher recall on both test sets, a false-alarm rate within 2×, RLVS test accuracy ≥ 98 % and "
              f"~{deploy['speedup']:.0f}× lower latency. **V2 + tiles is the strict mode** (recall "
              f"{table[STRICT[:2] + ('full', 'tiles')]['recall']:.3f} on the full set, false-alarm rate "
              f"{table[STRICT[:2] + ('full', 'tiles')]['false_positive_rate']:.3f}). Both at threshold 0.5 "
              "(`config.MODES`)."]
    else:
        L += ["", "**V2 whole screen does NOT pass every check above against V1 + tiles** — revisit the default."]
    L.append("")
    (RESULTS_MODEL_COMPARISON / "model_comparison.md").write_text("\n".join(L))

    key = lambda k: "|".join(str(x) for x in k)
    REPORT_JSON.write_text(json.dumps({
        "models": name_of, "run_files": {m: f"runs/{RUN_PREFIX[m]}_t<thr>_<full|clean>.*" for m in MODELS},
        "runs": RUNS, "counts": counts,
        "screen": {key(k): v for k, v in table.items()}, "recall_by_size": {key(k): v for k, v in sizes.items()},
        "rlvs_test": {key(k): v for k, v in rlvs.items()}, "latency": lat,
        "measurements": {MODELS[m].name: {"rlvs": {str(t): rlvs[(m, t)] for mm, t in RUNS if mm == m},
                                          "latency": lat[m], "measured": measured[m]} for m in MODELS},
        "pre_set_rule": {**decision, "verdicts": {str(t): v for t, v in decision["verdicts"].items()}},
        "deployment_decision": {"baseline": BASELINE, "default": DEPLOY, "strict": STRICT, **deploy},
    }, indent=2))
    print("\n".join(L))
    print("\nSaved → results/05_model_comparison/model_comparison.md, model_comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Meme head v2: train on the three views, and compare it with the v1 recipe on every view.

    python -m component2_hate_speech_detection.meme_model.train_views

Fixed before any result was seen (protocol written 19 Oct 2026, run 22 Oct):
- v1 recipe: the shipped head's settings, trained on clean captions only (as v1 was).
- v2: the same settings, trained on clean + ocr + screen (3 x 8,000 rows). No new search.
- Settings, from v1's chosen run (evaluation/results/meme_clip-b32_align.json): align fusion,
  projection 1024, 3 pre-output layers, hidden 1024, dropout 0.4, AdamW lr 1e-4, weight
  decay 1e-4, gradient clip 0.1, batch 64, up to 20 epochs, patience 5, seeds 316-320.
- Validation (early stopping, temperature, threshold): v1 recipe on clean val, as before;
  v2 on all three val views pooled.
- Test (dev_seen, 500 memes) is reported once per view. On the screen view a meme the
  picture finder missed scores 0, as evaluation/screen_memes.py does.
- Success rule: v2 beats the v1 recipe on ocr and screen (ensemble AUROC) and loses no
  more than 0.01 on clean.
The shipped v1 head (models/meme/meme_clip-b32_align.pt) is scored on the same features too.

Writes evaluation/results/meme_v2/meme_v2_views.json (aggregate numbers only, safe to commit),
models/meme/meme_clip-b32_align_v2views.pt and data/predictions/meme_v2_views.csv (git-ignored).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .embed import EMBED_DIR
from .heads import FusionHead, count_parameters
from .train import (MODEL_DIR, PREDICTION_DIR, RESULTS_DIR, SEEDS, best_threshold, bootstrap_auroc, fit_temperature,
                    mean_sd, predict_logits, resolve_device, sigmoid, summarise, train_one)

VIEWS = ("clean", "ocr", "screen")
V1_SETTINGS = dict(fusion="align", proj_dim=1024, pre_layers=3, hidden=1024, dropout=0.4, lr=1e-4, weight_decay=1e-4,
                   grad_clip=0.1, batch_size=64, epochs=20, patience=5, class_weight=False, rgcl=0.0, knn=0)
RECIPES = {"v1_recipe": ("clean",), "v2_views": VIEWS}
OUT_DIR = RESULTS_DIR / "meme_v2"
SHIPPED = MODEL_DIR / "meme_clip-b32_align.pt"


def load_views(path: Path = EMBED_DIR / "clip-b32-views.npz") -> dict:
    z = np.load(path)
    data = {}
    for split in ("train", "val", "test"):
        m = z["split"] == split
        data[split] = {"ids": z["ids"][m], "y": z["label"][m].astype(np.float32), "found": z["found"][m].astype(bool),
                       "clean": (z["image_orig"][m], z["text_clean"][m]), "ocr": (z["image_orig"][m], z["text_ocr"][m]),
                       "screen": (z["image_screen"][m], z["text_screen"][m])}
    return data


def arrays(split: dict, views) -> dict:
    return {"image": np.concatenate([split[v][0] for v in views]).astype(np.float32),
            "text": np.concatenate([split[v][1] for v in views]).astype(np.float32),
            "y": np.concatenate([split["y"]] * len(views)).astype(np.float32)}


def test_probabilities(model: FusionHead, data: dict, temperature: float) -> dict:
    out = {}
    for v in VIEWS:
        p = sigmoid(predict_logits(model, arrays(data["test"], (v,))) / temperature)
        out[v] = np.where(data["test"]["found"], p, 0.0) if v == "screen" else p
    return out


def view_report(y: np.ndarray, found: np.ndarray, per_seed: list[dict], thresholds: list[float]) -> dict:
    rep = {}
    for v in VIEWS:
        ens = np.mean([s[v] for s in per_seed], axis=0)
        rep[v] = {"auroc_mean_sd": mean_sd([float(roc_auc_score(y, s[v])) for s in per_seed]),
                  "ensemble": {**summarise(y, ens, float(np.mean(thresholds))), "auroc_ci95": bootstrap_auroc(y, ens)}}
        if v == "screen":
            rep[v]["ensemble_found_only_auroc"] = round(float(roc_auc_score(y[found], ens[found])), 4)
    return rep


def run_recipe(name: str, data: dict, args: argparse.Namespace) -> dict:
    views = RECIPES[name]
    splits = {"train": arrays(data["train"], views), "val": arrays(data["val"], views)}
    print(f"\n{name}: train rows {len(splits['train']['y'])}, val rows {len(splits['val']['y'])}", flush=True)
    runs, per_seed_p, thresholds = [], [], []
    for seed in SEEDS[: args.seeds]:
        started = time.perf_counter()
        model, info = train_one(splits, args, seed)
        val_logits = predict_logits(model, splits["val"])
        temperature = fit_temperature(val_logits, splits["val"]["y"])
        val_p = sigmoid(val_logits / temperature)
        threshold = best_threshold(splits["val"]["y"], val_p)
        probs = test_probabilities(model, data, temperature)
        val_auroc = float(roc_auc_score(splits["val"]["y"], val_p))
        runs.append({"seed": seed, "model": model, "temperature": round(temperature, 4), "val_threshold": round(threshold, 4),
                     "val_auroc": round(val_auroc, 4), "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                     "history": info["history"], "seconds": round(time.perf_counter() - started, 1),
                     "test_auroc": {v: round(float(roc_auc_score(data["test"]["y"], probs[v])), 4) for v in VIEWS}})
        per_seed_p.append(probs)
        thresholds.append(threshold)
        r = runs[-1]
        print(f"  seed {seed}: val {r['val_auroc']:.4f}  test clean {r['test_auroc']['clean']:.4f}  ocr {r['test_auroc']['ocr']:.4f}"
              f"  screen {r['test_auroc']['screen']:.4f}  epoch {r['best_epoch']}  {r['seconds']}s", flush=True)
    return {"runs": runs, "probs": per_seed_p, "thresholds": thresholds,
            "report": view_report(data["test"]["y"], data["test"]["found"], per_seed_p, thresholds)}


def shipped_v1(data: dict) -> Optional[dict]:
    if not SHIPPED.exists():
        return None
    ckpt = torch.load(SHIPPED, map_location="cpu", weights_only=False)
    model = FusionHead(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    probs = test_probabilities(model, data, float(ckpt["temperature"]))
    y = data["test"]["y"]
    return {"seed": ckpt.get("seed"), "temperature": ckpt["temperature"],
            "views": {v: summarise(y, probs[v], float(ckpt.get("val_threshold", 0.5))) for v in VIEWS}, "probs": probs}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    for k, v in V1_SETTINGS.items():
        setattr(args, k, v)

    started = time.perf_counter()
    data = load_views()
    device = resolve_device(args.device)
    sizes = {s: int(len(data[s]["y"])) for s in data}
    print(f"memes per split {sizes}  device {device}", flush=True)

    results = {name: run_recipe(name, data, args) for name in RECIPES}
    shipped = shipped_v1(data)

    v1, v2 = results["v1_recipe"]["report"], results["v2_views"]["report"]
    auc = lambda rep, v: rep[v]["ensemble"]["auroc"]
    rule = {"ocr_gain": round(auc(v2, "ocr") - auc(v1, "ocr"), 4), "screen_gain": round(auc(v2, "screen") - auc(v1, "screen"), 4),
            "clean_change": round(auc(v2, "clean") - auc(v1, "clean"), 4)}
    rule["pass"] = rule["ocr_gain"] > 0 and rule["screen_gain"] > 0 and rule["clean_change"] >= -0.01

    best = max(results["v2_views"]["runs"], key=lambda r: r["val_auroc"])  # chosen on validation, never on test
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"config": best["model"].config, "encoder": "clip-b32", "trained_on": list(VIEWS), "seed": best["seed"],
                "temperature": best["temperature"], "val_threshold": best["val_threshold"],
                "state_dict": best["model"].state_dict()}, MODEL_DIR / "meme_clip-b32_align_v2views.pt")

    report = {
        "run": "meme_v2_views", "date": time.strftime("%Y-%m-%d"), "encoder": "openai/clip-vit-base-patch32 (frozen)",
        "settings": V1_SETTINGS, "seeds": list(SEEDS[: args.seeds]),
        "trainable_parameters": count_parameters(best["model"]),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "memes_per_split": sizes, "train_rows": {k: sizes["train"] * len(v) for k, v in RECIPES.items()},
        "test_found_by_finder": round(float(data["test"]["found"].mean()), 4),
        "recipes": {name: {"test": r["report"], "val_auroc_mean_sd": mean_sd([x["val_auroc"] for x in r["runs"]]),
                           "per_seed": [{k: v for k, v in x.items() if k != "model"} for x in r["runs"]]}
                    for name, r in results.items()},
        "shipped_v1_head": {k: v for k, v in shipped.items() if k != "probs"} if shipped else None,
        "success_rule": rule, "chosen_v2_seed": best["seed"], "seconds": round(time.perf_counter() - started, 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "meme_v2_views.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREDICTION_DIR / "meme_v2_views.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "found", "view", "p_v1_recipe", "p_v2_views", "p_shipped_v1"])
        for v in VIEWS:
            p1 = np.mean([s[v] for s in results["v1_recipe"]["probs"]], axis=0)
            p2 = np.mean([s[v] for s in results["v2_views"]["probs"]], axis=0)
            p0 = shipped["probs"][v] if shipped else np.full(len(p1), np.nan)
            for i, mid in enumerate(data["test"]["ids"]):
                w.writerow([int(mid), int(data["test"]["y"][i]), int(data["test"]["found"][i]), v,
                            round(float(p1[i]), 5), round(float(p2[i]), 5), round(float(p0[i]), 5)])

    print("\nTest AUROC (dev_seen, 500 memes), ensemble of seeds [95% CI]:")
    print(f"  {'view':8} {'shipped v1':>11} {'v1 recipe':>22} {'v2 views':>22}")
    for v in VIEWS:
        s = f"{shipped['views'][v]['auroc']:.4f}" if shipped else "-"
        c = lambda rep: f"{rep[v]['ensemble']['auroc']:.4f} {rep[v]['ensemble']['auroc_ci95']}"
        print(f"  {v:8} {s:>11} {c(v1):>22} {c(v2):>22}")
    print(f"\nSuccess rule: {rule}")
    print(f"wrote {OUT_DIR / 'meme_v2_views.json'}")


if __name__ == "__main__":
    main()

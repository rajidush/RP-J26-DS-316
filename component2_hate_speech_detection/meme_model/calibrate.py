"""Calibration gate for a trained meme head (invariant I-1 in `engine/fusion.py`).

    python -m component2_hate_speech_detection.meme_model.calibrate --run meme_clip-b32_align

A meme score may move a decision only if it is calibrated. This script decides
that with a rule set in advance, not by eye on one ECE number:

- ECE on 500 memes is noisy: a perfectly calibrated score still shows ~0.05.
  So the gate uses consistency resampling (Broecker & Smith, 2007): draw labels
  from the scores themselves and ask how often a calibrated score would look as
  bad as ours. A score passes when p >= 0.05, on val (5-fold, in distribution)
  and on test (held out).
- Calibration holds for one base rate. Train and val are 35 % hateful, dev_seen
  49 %. Test is reweighted to the training base rate (label shift; Saerens et
  al., 2002), so the check is about the score, not about the base rate.
- Reliability is also reported per decision band: 0.35 is the stage-1 floor
  and the bottom of the fusion agreement band, 0.55 / 0.65 / 0.75 are the
  persona thresholds.

Temperature scaling, fitted on val in train.py, is the shipped calibrator
(Guo et al., 2017). Platt and isotonic are fitted on val for comparison only.
Test chooses nothing: it gives the gate verdict, which is read, never tuned.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .heads import FusionHead
from .train import (MODEL_DIR, RESULTS_DIR, expected_calibration_error, fit_temperature, load_splits,
                    predict_logits, sigmoid)

GATE_P = 0.05
CONSISTENCY_ROUNDS = 2000
CV_FOLDS = 5
# 0.35 = stage-1 floor and agreement-band floor; 0.55 / 0.65 / 0.75 = persona thresholds.
BANDS = ((0.0, 0.35), (0.35, 0.55), (0.55, 0.75), (0.75, 1.0))


# ---------------------------------------------------------------- base rate

def log_odds(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def shift_prior(p: np.ndarray, from_rate: float, to_rate: float) -> np.ndarray:
    """Move a calibrated probability to another base rate (Saerens et al., 2002)."""
    return sigmoid(log_odds(p) + log_odds(to_rate) - log_odds(from_rate))


def class_weights(y: np.ndarray, from_rate: float, to_rate: float) -> np.ndarray:
    """Per-meme weights that make a sample with base rate `from_rate` count as `to_rate`."""
    return np.where(np.asarray(y) == 1, to_rate / from_rate, (1 - to_rate) / (1 - from_rate))


# ---------------------------------------------------------------- tests of calibration

def consistency_test(y: np.ndarray, p: np.ndarray, rate: Optional[float] = None,
                     reference: Optional[float] = None, rounds: int = CONSISTENCY_ROUNDS,
                     seed: int = 316) -> dict:
    """H0: `p` is calibrated at the `reference` base rate.

    The sample has base rate `rate`, so under H0 the chance a meme is hateful is
    p moved to `rate`. Every ECE, observed and simulated, is computed with the
    weights that move the sample back to `reference`.
    """
    y = np.asarray(y, dtype=float)
    rate = float(y.mean()) if rate is None else rate
    reference = rate if reference is None else reference
    observed = expected_calibration_error(y, p, weights=class_weights(y, rate, reference))
    truth = shift_prior(p, reference, rate)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(rounds):
        draw = rng.binomial(1, truth).astype(float)
        null.append(expected_calibration_error(draw, p, weights=class_weights(draw, rate, reference)))
    null = np.array(null)
    return {"ece": round(observed, 4), "null_ece_95": round(float(np.percentile(null, 95)), 4),
            "p_value": round(float((null >= observed).mean()), 4)}


def band_reliability(y: np.ndarray, p: np.ndarray, weights: Optional[np.ndarray] = None) -> list[dict]:
    """Mean score vs observed hateful rate inside each decision band."""
    w = np.ones(len(p)) if weights is None else weights
    rows = []
    for lo, hi in BANDS:
        mask = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if mask.any():
            rows.append({"band": [lo, hi], "n": int(mask.sum()),
                         "mean_p": round(float(np.average(p[mask], weights=w[mask])), 3),
                         "observed": round(float(np.average(y[mask], weights=w[mask])), 3)})
    return rows


# ---------------------------------------------------------------- calibrators (fitted on val only)

Calibrator = Callable[[np.ndarray], np.ndarray]


def fit_calibrator(kind: str, logits: np.ndarray, y: np.ndarray) -> Calibrator:
    if kind == "temperature":
        t = fit_temperature(logits, y)
        return lambda z: sigmoid(z / t)
    if kind == "platt":
        model = LogisticRegression(C=1e6).fit(logits[:, None], y)
        return lambda z: model.predict_proba(z[:, None])[:, 1]
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3).fit(logits, y)
        return lambda z: model.predict(z)
    raise ValueError(f"unknown calibrator {kind!r}")


def cross_validated(kind: str, logits: np.ndarray, y: np.ndarray, folds: int = CV_FOLDS) -> np.ndarray:
    """Out-of-fold probabilities on val, so val can judge its own calibrator fairly."""
    out = np.zeros(len(y))
    for fit_idx, held_idx in StratifiedKFold(folds, shuffle=True, random_state=316).split(logits, y):
        out[held_idx] = fit_calibrator(kind, logits[fit_idx], y[fit_idx])(logits[held_idx])
    return out


# ---------------------------------------------------------------- report

def load_head(run: str) -> tuple[FusionHead, dict]:
    checkpoint = torch.load(MODEL_DIR / f"{run}.pt", map_location="cpu", weights_only=False)
    model = FusionHead(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def evaluate(run: str) -> dict:
    model, checkpoint = load_head(run)
    splits = load_splits(checkpoint["encoder"])
    val_z, val_y = predict_logits(model, splits["val"]).astype(float), splits["val"]["y"]
    test_z, test_y = predict_logits(model, splits["test"]).astype(float), splits["test"]["y"]
    reference = float(splits["train"]["y"].mean())
    test_rate = float(test_y.mean())
    to_reference = class_weights(test_y, test_rate, reference)

    calibrators = {}
    for kind in ("temperature", "platt", "isotonic"):
        val_cv = cross_validated(kind, val_z, val_y)
        if kind == "temperature":  # the shipped head's own temperature, fitted on all of val
            test_p = sigmoid(test_z / checkpoint["temperature"])
        else:
            test_p = fit_calibrator(kind, val_z, val_y)(test_z)
        calibrators[kind] = {
            "val_cv": consistency_test(val_y, val_cv),
            "test_raw_ece": round(expected_calibration_error(test_y, test_p), 4),
            "test_at_reference": consistency_test(test_y, test_p, test_rate, reference),
            "test_mean_p": round(float(test_p.mean()), 3),
        }

    shipped = calibrators["temperature"]
    test_p = sigmoid(test_z / checkpoint["temperature"])
    val_cv_p = cross_validated("temperature", val_z, val_y)
    passes = shipped["val_cv"]["p_value"] >= GATE_P and shipped["test_at_reference"]["p_value"] >= GATE_P
    return {
        "run": run,
        "date": dt.date.today().isoformat(),
        "encoder": checkpoint["encoder"],
        "seed": checkpoint["seed"],
        "temperature": round(float(checkpoint["temperature"]), 4),
        "base_rates": {"train": round(reference, 3), "val": round(float(val_y.mean()), 3),
                       "test": round(test_rate, 3), "reference": round(reference, 3)},
        "calibrators": calibrators,
        "bands_val_cv": band_reliability(val_y, val_cv_p),
        "bands_test_at_reference": band_reliability(test_y, test_p, to_reference),
        "gate": {
            "rule": f"consistency-resampling p >= {GATE_P} on val (5-fold) and on test at the reference "
                    "base rate, for the shipped temperature-scaled score",
            "val_cv_p": shipped["val_cv"]["p_value"],
            "test_p": shipped["test_at_reference"]["p_value"],
            "passes": bool(passes),
        },
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="meme_clip-b32_align", help="checkpoint name in models/meme/")
    args = parser.parse_args(argv)
    report = evaluate(args.run)
    out = RESULTS_DIR / f"{args.run}_calibration.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    gate = report["gate"]
    print(f"I-1 gate: val p={gate['val_cv_p']}, test p={gate['test_p']} -> {'PASS' if gate['passes'] else 'FAIL'}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

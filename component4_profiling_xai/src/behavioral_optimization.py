"""Training-only scorer construction and validation selection for the PP2 study."""
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from .behavioral_anomaly import FEATURE_SET_A, anomaly_scores, build_pipeline, assessable_days
from .behavioral_profile import FUSED, PERSONAL, TEMPORAL, ProfileConfig, profiles

WINDOWS = (7, 14, 21, 30)
HOLDOUT_SEED = 2026112607
GRIDS = {
    "one_class_svm": [{"nu": nu, "gamma": gamma}
                      for nu, gamma in product((0.01, 0.05), (0.0003, 0.001, 0.01))],
    "isolation_forest": [{"n_estimators": 64, "n_jobs": 1, "max_samples": rows, "max_features": columns}
                         for rows, columns in product((128, 256, 512), (0.65, 1.0))],
}
DEFAULTS = {"one_class_svm": {"nu": 0.01, "gamma": 0.001},
            "isolation_forest": {"n_estimators": 64, "max_samples": 256, "n_jobs": 1}}


@dataclass
class Hybrid:
    """Blend normal-training-calibrated global and personal rarity."""
    global_pipeline: object
    global_columns: list
    personal_columns: list
    global_sorted: np.ndarray
    personal_sorted: np.ndarray
    weight: float

    @staticmethod
    def personal_scores(frame):
        # Positive deviations represent activity above the child's recorded baseline.
        return np.maximum(frame.fillna(0).to_numpy(dtype=float), 0).max(axis=1)

    @staticmethod
    def rarity(reference, scores):
        tail = (len(reference) - np.searchsorted(reference, scores, side="left") + 1) / (len(reference) + 1)
        return -np.log(tail)

    def score_samples(self, frame):
        raw = anomaly_scores(self.global_pipeline, frame[self.global_columns])
        personal = self.personal_scores(frame[self.personal_columns])
        return -((1 - self.weight) * self.rarity(self.global_sorted, raw)
                 + self.weight * self.rarity(self.personal_sorted, personal))


def calibration(labels, scores):
    """Translate sklearn's inclusive PR thresholds to the strict deployed rule."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or set(labels) != {0, 1}:
        raise ValueError("Calibration needs finite scores and both label classes")
    precision, recall, inclusive = precision_recall_curve(labels, scores)
    thresholds = np.nextafter(inclusive, -np.inf)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=p + r > 0)
    positives = labels.sum()
    fp = np.round(np.divide(r * positives, p, out=np.zeros_like(p), where=p > 0) - r * positives)
    index = np.lexsort((-thresholds, fp, -f1))[0]
    return {"threshold": float(thresholds[index]), "validation_f1": float(f1[index]),
            "validation_fpr": float(fp[index] / (len(labels) - positives)),
            "validation_ap": float(average_precision_score(labels, scores))}


def selection_key(result):
    return (-result["validation_f1"], result["validation_fpr"], -result["validation_ap"], result["id"])


def development(daily):
    """Discard consumed test dates before profiles, fitting or model selection."""
    rows = daily[daily["split"].isin(["train", "val", "validation"])].copy().reset_index(drop=True)
    rows["split"] = rows["split"].replace("val", "validation")
    if rows.loc[rows["split"].eq("train"), "date"].max() >= rows.loc[rows["split"].eq("validation"), "date"].min():
        raise ValueError("Training and validation dates overlap")
    return rows


def matrices(daily):
    return {window: profiles(daily, ProfileConfig(days=window))[0] for window in WINDOWS}


def candidates(daily, frames):
    covered = assessable_days(daily, frames[14])
    fit = covered & daily["split"].eq("train") & daily["is_anomaly"].eq(0)
    valid = covered & daily["split"].eq("validation")
    normal, labels = frames[14].loc[fit], daily.loc[valid, "is_anomaly"]
    correlations = normal[FUSED].corr(method="spearman").abs()
    pruned = []
    for column in FUSED:
        if normal[column].nunique(dropna=True) > 1 and not correlations.loc[column, pruned].ge(0.95).any():
            pruned.append(column)
    all_results, winners, temporal_comparator = [], {}, {}

    def register(design, arm, window, features, model, params):
        frame = frames[window]
        metric = calibration(labels, anomaly_scores(model, frame.loc[valid, features]))
        result = {"id": f"{design}-{arm}-{window}-{len(all_results):03d}", "design": design,
                  "algorithm": arm, "window": window, "features": features, "params": params, **metric}
        all_results.append(result)
        key = (design, arm, window)
        if key not in winners or selection_key(result) < selection_key(winners[key][0]):
            winners[key] = (result, model)

    def run(design, window, features, grid):
        normal = frames[window].loc[fit, features]
        for arm, options in grid.items():
            for params in options:
                register(design, arm, window, features, build_pipeline(arm, params).fit(normal), params)

    run("global", 14, FEATURE_SET_A, GRIDS)
    run("global_pruned", 14, pruned, GRIDS)
    for window in WINDOWS:
        run("full_personalized", window, FUSED + PERSONAL + TEMPORAL, GRIDS)
        run("personal_only_context", window, FUSED + PERSONAL, DEFAULT_GRID)
    picked = sorted((pair for (design, _, _), pair in winners.items() if design == "personal_only_context"),
                    key=lambda pair: selection_key(pair[0]))
    short_windows = sorted({14, picked[0][0]["window"]})
    for window in short_windows:
        for feature in PERSONAL:
            run(f"add_{feature}", window, FUSED + [feature], DEFAULT_GRID)
        for prefix in ("screen_", "language_"):
            run(f"drop_{prefix[:-1]}_history", window,
                FUSED + [f for f in PERSONAL + TEMPORAL if not f.startswith(prefix)], DEFAULT_GRID)
        ranked = sorted((pair[0] for (design, _, days), pair in winners.items()
                         if days == window and design.startswith("add_")), key=selection_key)
        best_three = list(dict.fromkeys(row["features"][-1] for row in ranked))[:3]
        for n in (1, 3):
            run("selected_personalized", window, FUSED + best_three[:n], GRIDS)
    run("frequency_only_context", 14, FUSED + TEMPORAL, DEFAULT_GRID)
    run("fused_raw", 14, FUSED, DEFAULT_GRID)
    run("no_direct_urgent", 14, [f for f in FUSED if f not in {
        "language_urgent_flags", "language_self_harm_flags", "language_grooming_flags"}], GRIDS)
    for source in ("screen", "language"):
        run(f"{source}_only", 14, [f for f in FUSED if f.startswith(source + "_")], GRIDS)
    global_result, global_model = min((pair for (d, _, _), pair in winners.items() if d == "global"),
                                      key=lambda pair: selection_key(pair[0]))
    raw = global_result["features"]
    reference = np.sort(anomaly_scores(global_model, normal[raw]))
    for window, weight in product(WINDOWS, (0.15, 0.35, 0.55, 0.75)):
        personal = np.sort(Hybrid.personal_scores(frames[window].loc[fit, PERSONAL]))
        model = Hybrid(global_model, raw, PERSONAL, reference, personal, weight)
        register("hybrid", global_result["algorithm"], window, raw + PERSONAL, model, {"weight": weight})

    for pair in winners.values():
        result = pair[0]
        design = result["design"]
        if design not in temporal_comparator or selection_key(result) < selection_key(temporal_comparator[design][0]):
            temporal_comparator[design] = pair
    return all_results, winners, temporal_comparator, fit, valid, correlations


DEFAULT_GRID = {name: [params] for name, params in DEFAULTS.items()}

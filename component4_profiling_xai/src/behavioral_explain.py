"""Exact interventional group SHAP and regression LIME for the local detector."""
from itertools import product
from math import comb
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .behavioral_anomaly import anomaly_scores
from .behavioral_shap import TOTALS


def source_group(feature):
    if feature.startswith("screen_"):
        return "Screen activity"
    if feature.startswith("language_"):
        return "Language activity"
    return "Calendar context"


def explain(pipeline, row, background, seed=42):
    """Keep source columns and their derived histories together in perturbations."""
    start = perf_counter()
    groups = {source_group(col) for col in row.columns if col not in TOTALS}
    groups = {name: [col for col in row if source_group(col) == name] for name in sorted(groups)}
    n = len(groups)
    masks = np.array(list(product((0, 1), repeat=n)))
    batches = []
    reference = background.to_numpy(dtype=float)
    for mask in masks:
        batch = reference.copy()
        for include, columns in zip(mask, groups.values(), strict=True):
            if include:
                positions = row.columns.get_indexer(columns)
                batch[:, positions] = row.iloc[0, positions].to_numpy(dtype=float)
        for total, constituents in TOTALS.items():
            if total in row:
                batch[:, row.columns.get_loc(total)] = batch[:, row.columns.get_indexer(constituents)].sum(axis=1)
        batches.append(batch)
    scores = anomaly_scores(pipeline, pd.DataFrame(np.concatenate(batches), columns=row.columns))
    scores = scores.reshape(len(masks), len(background))
    shared_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    values = dict(zip(map(tuple, masks), scores.mean(axis=1), strict=True))
    shap = {}
    deletion = {}
    actual = values[(1,) * n]
    for i, name in enumerate(groups):
        shap[name] = sum(
            (values[(*mask[:i], 1, *mask[i + 1:])] - value) / (n * comb(n - 1, sum(mask)))
            for mask, value in values.items() if not mask[i]
        )
        deletion[name] = actual - values[tuple(int(j != i) for j in range(n))]

    shap_ms = shared_ms + (perf_counter() - start) * 1000
    start = perf_counter()
    # LIME with binary interpretable group presence, empirical replacements,
    # exponential locality kernel and a separate set of replacement samples.
    train, test = np.split(np.random.default_rng(seed).permutation(len(background)),
                           [len(background) // 2])
    distance = np.sqrt(np.sum(1 - masks, axis=1) / n)
    weights = np.exp(-distance**2 / 0.75**2)

    def neighborhood(indices):
        return (np.repeat(masks, len(indices), axis=0), scores[:, indices].ravel(),
                np.repeat(weights, len(indices)))

    x, y, w = neighborhood(train)
    surrogate = Ridge(alpha=0.01).fit(x, y, sample_weight=w)
    x_test, y_test, w_test = neighborhood(test)
    lime = dict(zip(groups, surrogate.coef_.tolist(), strict=True))
    return {"shap_ms": shap_ms, "lime_ms": shared_ms + (perf_counter() - start) * 1000,
            "shap": shap, "lime": lime, "deletion": deletion, "score": actual,
            "reference": values[(0,) * n], "lime_intercept": float(surrogate.intercept_),
            "lime_r2": float(surrogate.score(x_test, y_test, sample_weight=w_test))}



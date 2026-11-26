"""Exact coalition batching scoped to one dashboard explanation request."""
from collections import OrderedDict
from itertools import product
from time import perf_counter

import numpy as np
import pandas as pd

from .behavioral_anomaly import anomaly_scores
from .behavioral_explain import (
    fit_surrogate, grouped_features, reference_score, source_group, summarize_coalitions, verify_shap,
)
from .behavioral_explanation_reliability import (
    EXPLANATION_ERRORS, PERTURBATION, assess_daily, daily_evidence, perturb, temporal_evidence,
)
from .behavioral_shap import TOTALS
from .behavioral_temporal_explain import DEFAULT_WINDOW, compare_contributions, explain_temporal


class ExplanationRequest:
    """Owns one bundle and a bounded set of exact per-row coalition results.

    Created per render. Only actual feature values key the memo within the
    request, and raw rows and outcomes never persist across dashboard renders.
    """

    MAX_ROWS = 44
    BATCH_ROWS = 2

    def __init__(self, bundle):
        self.bundle = bundle
        self.pipeline = bundle["pipeline"]
        self.columns = pd.Index(bundle["features"])
        self.background = bundle["background"].loc[:, self.columns].to_numpy(dtype=float)
        self.groups = sorted(grouped_features(self.columns))
        self.masks = np.array(list(product((0, 1), repeat=len(self.groups))))
        group_positions = [self.groups.index(source_group(column)) if column not in TOTALS else 0
                           for column in self.columns]
        self.membership = self.masks[:, group_positions].astype(bool)
        self.totals = [(self.columns.get_loc(column), self.columns.get_indexer(parts))
                       for column, parts in TOTALS.items() if column in self.columns]
        self._reference = None
        self._empty_scores = None
        self._memo = OrderedDict()

    def reference(self):
        if self._reference is None:
            self._reference = reference_score(self.pipeline, self.bundle["background"])
        return self._reference

    def _batch(self, rows):
        donors, width = self.background.shape
        mixed_count = len(self.masks) - 2
        mixed_rows = len(rows) * mixed_count * donors
        full_end = mixed_rows + len(rows)
        include_empty = self._empty_scores is None
        values = np.empty((full_end + (donors if include_empty else 0), width))
        mixed = values[:mixed_rows].reshape(len(rows), mixed_count, donors, width)
        mixed[...] = self.background
        np.copyto(mixed, rows[:, None, None, :], where=self.membership[None, 1:-1, None, :])
        values[mixed_rows:full_end] = rows
        if include_empty:
            values[full_end:] = self.background
        for position, parts in self.totals:
            values[:, position] = values[:, parts].sum(axis=1)
        scores = anomaly_scores(self.pipeline, pd.DataFrame(values, columns=self.columns))
        if not np.isfinite(scores).all() or len(self.background) < 2:
            raise ValueError("Coalition scores cannot support a finite local explanation")
        if include_empty:
            self._empty_scores = scores[full_end:].copy()
        # Keep the identical score grid and reduction order. The reference
        # verification still uses an independent call to the original scorer.
        grid = np.empty((len(rows), len(self.masks), donors))
        grid[:, 0] = self._empty_scores
        grid[:, 1:-1] = scores[:mixed_rows].reshape(len(rows), mixed_count, donors)
        grid[:, -1] = scores[mixed_rows:full_end, None]
        return grid

    def _coalitions(self, frame, batch_rows):
        rows = frame.loc[:, self.columns].to_numpy(dtype=float)
        keys = [row.tobytes() for row in rows]
        found = {}
        missing = {}
        for key, row in zip(keys, rows, strict=True):
            if key in self._memo:
                found[key] = self._memo[key]
                self._memo.move_to_end(key)
            else:
                missing[key] = row
        keys_to_score = list(missing)
        for start in range(0, len(keys_to_score), batch_rows):
            part = keys_to_score[start:start + batch_rows]
            scores = self._batch(np.stack([missing[key] for key in part]))
            for key, grid in zip(part, scores, strict=True):
                item = (grid, summarize_coalitions(grid, self.masks, self.groups))
                found[key] = self._memo[key] = item
                if len(self._memo) > self.MAX_ROWS:
                    self._memo.popitem(last=False)
        return [found[key] for key in keys]

    def explain_rows(self, frame, seed=42):
        start = perf_counter()
        evidence = self._coalitions(frame, 1)
        shap_ms = (perf_counter() - start) * 1000 / max(len(evidence), 1)
        results = []
        for scores, shap in evidence:
            start = perf_counter()
            lime = fit_surrogate(scores, self.masks, self.groups, seed)
            results.append({**shap, **lime, "shap_ms": shap_ms,
                            "lime_ms": shap_ms + (perf_counter() - start) * 1000})
        return results

    def daily(self, row, score, amount=PERTURBATION):
        if not np.isfinite(score):
            return None, []
        try:
            candidates = [perturb(row, delta) for delta in (-amount, amount)]
            changed = [candidate for candidate in candidates if not candidate.equals(row)]
            results = self.explain_rows(pd.concat([row, *changed]))
            variants = []
            if changed:
                scores = anomaly_scores(self.pipeline, pd.concat(changed))
                variants = list(zip(results[1:], scores.tolist(), strict=True))
            changing = iter(variants)
            return results[0], [(results[0], score) if candidate.equals(row) else next(changing)
                                for candidate in candidates]
        except EXPLANATION_ERRORS:
            return daily_evidence(row, score, self.bundle, amount)

    def review_daily(self, result, score, flagged, variants):
        if result is None or not np.isfinite(score):
            return assess_daily(result, score, flagged, self.bundle, variants)
        return assess_daily(result, score, flagged, self.bundle, variants, reference=self.reference())

    def _temporal_result(self, history, rows, child, day):
        reference = self.reference()
        contributions = pd.DataFrame(np.nan, index=history.index, columns=self.groups)
        verified = self._coalitions(rows, self.BATCH_ROWS)
        groups = set(self.groups)
        for index, (_, attribution) in zip(rows.index, verified, strict=True):
            if verify_shap(attribution, history.at[index, "score"], groups, reference):
                contributions.loc[index] = tuple(attribution["shap"][group] for group in self.groups)
        return compare_contributions(history, contributions, child, day, self.bundle["scale"])

    def temporal(self, history, matrix, child, day, amount=PERTURBATION, *, review=True):
        end = pd.Timestamp(day).normalize()
        dates = pd.to_datetime(history["date"])
        rows = history.loc[history["child_id"].eq(child) & dates.between(
            end - pd.Timedelta(days=2 * DEFAULT_WINDOW - 1), end)]
        observed = rows.index[np.isfinite(rows["score"])]
        features = matrix.loc[observed, self.columns]
        try:
            result = self._temporal_result(rows, features, child, day)
            variants = []
            if review and result.status in {"stable", "isolated", "sustained"}:
                for delta in (-amount, amount):
                    changed = perturb(features, delta)
                    if changed.equals(features):
                        variants.append(result)
                        continue
                    replay = rows.copy()
                    scores = anomaly_scores(self.pipeline, changed)
                    if not np.isfinite(scores).all():
                        raise ValueError("Nonfinite perturbed scores")
                    replay.loc[observed, "score"] = scores
                    variants.append(self._temporal_result(replay, changed, child, day))
            return result, variants
        except EXPLANATION_ERRORS:
            if review:
                return temporal_evidence(history, matrix, self.bundle, child, day, amount)
            return explain_temporal(history, matrix, self.bundle, child, day), []

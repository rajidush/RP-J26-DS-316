"""Offline attributions for the frozen Function 2 anomaly score."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from math import comb
from threading import Lock

import numpy as np
import pandas as pd

from .behavioral_anomaly import (
    ARTIFACT_DIR, FEATURE_SET_A, anomaly_scores, assessable_days, feature_matrix,
    load_artifacts, validate_artifacts,
)
from .behavioral_dataset import daily_numeric

TOTALS = {
    "total_risk_events": ("screen_incidents", "language_flags"),
    "night_risk_events": ("screen_night_incidents", "language_night_flags"),
}
GROUPS = {
    "screen-related recorded activity": tuple(f for f in FEATURE_SET_A if f.startswith("screen_")),
    "language-related recorded signals": tuple(f for f in FEATURE_SET_A if f.startswith("language_")),
    "weekday or weekend context": ("is_weekend",),
}
# Additive display bookkeeping only. The constrained game recomputes both totals.
AGGREGATE_GROUPS = {**GROUPS, "combined recorded event totals": tuple(TOTALS)}
TOLERANCE = 1e-8
FALLBACK = (
    "The system identified a statistical difference in the available records, "
    "but cannot reliably explain which recorded factors contributed most."
)
_KERNEL_LOCK = Lock()


@dataclass(frozen=True)
class Background:
    values: pd.DataFrame
    identifiers: tuple[tuple[str, str], ...]
    seed: int

    def metadata(self):
        return {
            "identifiers": self.identifiers,
            "seed": self.seed,
            "size": len(self.values),
            "strategy": "round robin across children and within-child event-volume terciles",
            "source": "eligible baseline-ready normal training days",
        }


def normal_training_rows(daily):
    """Reproduce the frozen model's training filter plus current coverage gate."""
    features = feature_matrix(daily, FEATURE_SET_A)
    eligible = assessable_days(daily, features)
    ready = daily_numeric(daily["baseline_ready"], indicator=True).eq(1)
    return daily.loc[eligible & ready & daily["split"].eq("train") & daily["is_anomaly"].eq(0)]


def select_background(daily, size, seed):
    pool = normal_training_rows(daily).sort_values(["child_id", "date"]).reset_index(drop=True)
    if not 1 <= size <= len(pool):
        raise ValueError("background size must be within the eligible training pool")
    if pool.duplicated(["child_id", "date"]).any():
        raise ValueError("duplicate background identifiers")
    rng = np.random.default_rng(seed)
    queues = []
    for _, child in pool.groupby("child_id", sort=True):
        ordered = child.sort_values("total_risk_events", kind="stable")
        bins = [list(rng.permutation(indices)) for indices in
                       np.array_split(ordered.index.to_numpy(), 3)]
        start = int(rng.integers(3))
        queues.append(bins[start:] + bins[:start])
    chosen = []
    while len(chosen) < size:
        for child_index in rng.permutation(len(queues)):
            bins = queues[child_index]
            nonempty = [bucket for bucket in bins if bucket]
            if nonempty:
                # Rotate the activity stratum per child, without replacement.
                chosen.append(nonempty[0].pop())
                bins.append(bins.pop(bins.index(nonempty[0])))
            if len(chosen) == size:
                break
    rows = pool.loc[chosen]
    ids = tuple(rows[["child_id", "date"]].astype(str).itertuples(index=False, name=None))
    return Background(feature_matrix(rows, FEATURE_SET_A), ids, seed)


def mask_violations(frame):
    """Audit documented deterministic relationships in raw daily model inputs."""
    failures = {}
    for total, (left, right) in TOTALS.items():
        failures[total] = ~np.isclose(frame[total], frame[left] + frame[right], rtol=0, atol=TOLERANCE)
    for prefix, count, maximum, mean in (
        ("screen_", "screen_incidents", "screen_max_confidence", "screen_mean_confidence"),
        ("language_", "language_flags", "language_max_risk", "language_mean_risk"),
    ):
        for name in frame.columns:
            if name.startswith(prefix) and name.endswith(("_flags", "_incidents", "_apps", "_categories")):
                failures[name + " exceeds source count"] = frame[name] > frame[count]
        failures[mean + " exceeds maximum"] = frame[mean] > frame[maximum] + TOLERANCE
        failures[maximum + " nonzero with zero events"] = frame[count].eq(0) & frame[maximum].fillna(0).ne(0)
        failures[mean + " nonzero with zero events"] = frame[count].eq(0) & frame[mean].fillna(0).ne(0)
    return pd.DataFrame(failures, index=frame.index)


@dataclass(frozen=True)
class Explanation:
    method: str
    input_digest: str
    anomaly_score: float
    expected_score: float
    feature_values: dict[str, float] | None
    group_values: dict[str, float]
    reconstructed_score: float
    reconstruction_error: float
    threshold: float
    decision_margin: float
    is_anomaly: bool
    background: dict
    configuration: dict
    model_calls: int
    model_rows: int
    audited_rows: int
    invalid_rows: int
    violations: dict[str, int]

    @property
    def reconstructs(self):
        return (self.reconstruction_error <= TOLERANCE
                and (self.reconstructed_score > self.threshold) == self.is_anomaly)


class BehavioralShap:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        self.pipeline, self.metadata = load_artifacts(artifact_dir)
        validate_artifacts(self.pipeline, self.metadata, FEATURE_SET_A)
        if self.metadata["algorithm"] != "one_class_svm":
            raise ValueError("expected the frozen One-Class SVM artifact")

    def _input(self, daily):
        if len(daily) != 1 or not daily.columns.is_unique:
            raise ValueError("expected one daily record with unique column names")
        row = feature_matrix(daily, self.metadata["features"])
        return row if assessable_days(daily, row).iloc[0] else None

    def explain(self, daily, background, *, method="kernel", nsamples=256, seed=42):
        """Return None when the existing coverage gate precludes scoring."""
        if method not in ("kernel", "group_aware") or (
            method == "kernel" and nsamples < 2 * len(FEATURE_SET_A)
        ):
            raise ValueError("unsupported explanation method or Kernel SHAP budget")
        row = self._input(daily)
        if row is None:
            return None
        reference = background.values
        if (list(reference.columns) != self.metadata["features"] or reference.empty
                or len(reference) != len(background.identifiers)
                or not assessable_days(reference, reference).all()
                or mask_violations(reference).any(axis=None)):
            raise ValueError("invalid background feature order, eligibility or relationships")
        if method == "group_aware" and mask_violations(row).any(axis=None):
            raise ValueError("group-aware attribution requires consistent daily relationships")

        calls = rows = audited = invalid = 0
        violations = {}

        def score(values, *, audit=True):
            nonlocal calls, rows, audited, invalid
            frame = pd.DataFrame(values, columns=self.metadata["features"])
            calls += 1
            rows += len(frame)
            if audit:
                flags = mask_violations(frame)
                audited += len(frame)
                invalid += int(flags.any(axis=1).sum())
                for name, count in flags.sum().items():
                    violations[name] = violations.get(name, 0) + int(count)
            return anomaly_scores(self.pipeline, frame)

        actual = float(score(row, audit=False)[0])
        features = None
        if method == "kernel":
            import shap

            # KernelExplainer uses NumPy's legacy global RNG for mask sampling.
            with _KERNEL_LOCK:
                state = np.random.get_state()
                try:
                    np.random.seed(seed)
                    explainer = shap.KernelExplainer(score, reference, link="identity")
                    values = explainer.shap_values(row, nsamples=nsamples, l1_reg=0, silent=True)
                finally:
                    np.random.set_state(state)
            expected = float(explainer.expected_value)
            features = dict(zip(row.columns, np.asarray(values)[0].tolist(), strict=True))
            groups = {name: sum(features[f] for f in members)
                      for name, members in AGGREGATE_GROUPS.items()}
            configuration = {"nsamples": nsamples, "seed": seed, "link": "identity", "l1_reg": 0}
        else:
            coalitions = self._coalitions(row, reference, score)
            expected = coalitions[frozenset()]
            groups = {}
            n = len(GROUPS)
            for group in GROUPS:
                groups[group] = sum(
                    (coalitions[subset | {group}] - value) / (n * comb(n - 1, len(subset)))
                    for subset, value in coalitions.items() if group not in subset
                )
            configuration = {"coalitions": len(coalitions), "recompute_totals": True}

        reconstructed = expected + sum(groups.values())
        threshold = self.metadata["threshold"]
        return Explanation(
            method, sha256(row.to_numpy(dtype="<f8").tobytes()).hexdigest(),
            actual, expected, features, groups, reconstructed, abs(actual - reconstructed),
            threshold, actual - threshold, actual > threshold, background.metadata(), configuration,
            calls, rows, audited, invalid, violations,
        )

    @staticmethod
    def _replace_groups(row, reference, groups):
        masked = reference.copy()
        for group in groups:
            for feature in GROUPS[group]:
                masked[feature] = row[feature].iloc[0]
        for total, (left, right) in TOTALS.items():
            masked[total] = masked[left] + masked[right]
        return masked

    def _coalitions(self, row, reference, score):
        return {
            frozenset(subset): float(score(self._replace_groups(row, reference, subset)).mean())
            for size in range(len(GROUPS) + 1)
            for subset in combinations(GROUPS, size)
        }

    def deletion_effects(self, daily, background):
        """Score loss when one group is replaced with independent normal donors."""
        row = self._input(daily)
        if row is None:
            raise ValueError("deletion requires an assessable day")
        actual = anomaly_scores(self.pipeline, row)[0]
        effects = {}
        for group in GROUPS:
            masked = self._replace_groups(row, background.values, set(GROUPS) - {group})
            if mask_violations(masked).any(axis=None):
                raise ValueError("invalid grouped perturbations")
            effects[group] = float(actual - anomaly_scores(self.pipeline, masked).mean())
        return effects


def ranked(values, top_n=3):
    return sorted(values, key=lambda key: (-abs(values[key]), key))[:top_n]


def top_overlap(left, right, top_n=3):
    """Jaccard similarity of the top absolute contributions."""
    a, b = set(ranked(left, top_n)), set(ranked(right, top_n))
    return len(a & b) / len(a | b) if a or b else 1.0


def verified_groups(explanations, deletion, *, minimum=0.01):
    """Require stable, material signs across configurations and donor deletion."""
    if (len(explanations) < 2 or any(e is None for e in explanations)
            or len({tuple(e.background["identifiers"]) for e in explanations}) < 2):
        return {}
    primary = explanations[0]
    if primary.method != "group_aware":
        return {}
    if any(
        not e.reconstructs or e.method != primary.method or e.input_digest != primary.input_digest
        or e.is_anomaly != primary.is_anomaly
        or abs(e.anomaly_score - primary.anomaly_score) > TOLERANCE
        or e.threshold != primary.threshold or e.invalid_rows
        for e in explanations
    ):
        return {}
    verified = {}
    for group in ranked(primary.group_values, 2):
        if group not in GROUPS:
            continue
        delta = primary.group_values[group]
        evidence = [e.group_values.get(group, 0.0) for e in explanations]
        evidence.append(deletion.get(group, 0.0))
        sign = np.sign(delta)
        if all(sign != 0 and np.sign(value) == sign and abs(value) >= minimum for value in evidence):
            if all(group in ranked(e.group_values, 2) for e in explanations):
                verified[group] = delta
    return verified


def format_parent_explanation(explanations, deletion):
    """Generate evidence-gated language only for flagged, assessable records."""
    if not explanations or explanations[0] is None or not explanations[0].is_anomaly:
        return None
    verified = verified_groups(explanations, deletion)
    if not verified:
        return FALLBACK
    sentences = ["The model identified an unusual pattern in the available recorded activity."]
    for sign, direction in ((1, "increased"), (-1, "decreased")):
        names = [name for name, value in verified.items() if np.sign(value) == sign]
        if names:
            sentences.append(f"Compared with typical recorded days, {' and '.join(names)} "
                             f"{direction} the model's score for unusual activity.")
    return " ".join(sentences)

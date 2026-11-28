"""Parent-facing reliability policy for offline, exact group-aware attribution.

The original formatter remains available to reproduce the v1/v2 experiments.
New parent explanations go through ReliabilityGate.review.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import combinations

import numpy as np

from .behavioral_anomaly import FEATURE_SET_A, anomaly_scores, feature_matrix
from .behavioral_shap import (
    GROUPS, TOLERANCE, mask_violations, normal_training_rows, ranked, select_background,
)

REFERENCE_SETTINGS = ((32, 17), (32, 29), (64, 17), (64, 29))
DELETION_SETTING = (64, 101)
MIXED_COALITIONS = tuple(subset for size in (1, 2) for subset in combinations(GROUPS, size))
DELETION_COALITIONS = tuple(combinations(GROUPS, 2))
SUPPORT_KEYS = {
    "joint_event_counts_and_calendar": ("screen_incidents", "language_flags", "is_weekend"),
    "screen_night_counts_and_calendar": ("screen_incidents", "screen_night_incidents", "is_weekend"),
    "language_night_counts_and_calendar": ("language_flags", "language_night_flags", "is_weekend"),
}
FALLBACK = (
    "The system identified an unusual pattern in the available recorded activity, "
    "but cannot reliably identify which recorded factors contributed most."
)


class Reliability(str, Enum):
    RELIABLE = "reliable"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_FLAGGED = "not_flagged"


@dataclass(frozen=True)
class Decision:
    status: Reliability
    reasons: tuple[str, ...]
    score: float | None = None
    is_flagged: bool | None = None
    text: str | None = None
    diagnostics: dict = field(default_factory=dict)


class EmpiricalSupport:
    """Exact count/calendar keys must have witnesses from at least two training children."""

    def __init__(self, train):
        features = feature_matrix(train, FEATURE_SET_A)
        valid = ~mask_violations(features).any(axis=1)
        self.supported = {}
        for name, columns in SUPPORT_KEYS.items():
            witnesses = defaultdict(set)
            keys = features.loc[valid, list(columns)].itertuples(index=False, name=None)
            for key, child in zip(keys, train.loc[valid, "child_id"], strict=True):
                witnesses[key].add(child)
            self.supported[name] = {key for key, children in witnesses.items() if len(children) >= 2}

    def check(self, features):
        return {
            name: np.fromiter(
                (key in self.supported[name]
                 for key in features[list(columns)].itertuples(index=False, name=None)),
                dtype=bool, count=len(features),
            )
            for name, columns in SUPPORT_KEYS.items()
        }

    def restrict(self, explainer, target, background, coalitions):
        """Keep a single paired donor set across all relevant coalitions."""
        keep = np.ones(len(background.values), dtype=bool)
        sparse = dict.fromkeys(SUPPORT_KEYS, 0)
        endpoint_support = self.check(background.values)
        for supported in endpoint_support.values():
            keep &= supported
        for subset in coalitions:
            masked = explainer._replace_groups(target, background.values, subset)
            for name, supported in self.check(masked).items():
                sparse[name] += int((~supported).sum())
                keep &= supported
        identifiers = tuple(key for key, retained in zip(background.identifiers, keep, strict=True)
                            if retained)
        children = len({child for child, _ in identifiers})
        stats = {
            "original_donors": len(keep), "retained_donors": len(identifiers),
            "retained_children": children, "sparse_masks_by_key": sparse,
            "sparse_donor_endpoints_by_key": {name: int((~ok).sum()) for name, ok in endpoint_support.items()},
        }
        if children < 2:
            return None, stats
        return replace(background, values=background.values.iloc[keep], identifiers=identifiers), stats


def attribution_checks(vectors, deletions):
    """Check an empirical rank envelope across full and restricted references."""
    leader = ranked(vectors[0], 1)[0]
    orders = [ranked(values) for values in vectors]
    values = [row[leader] for row in vectors]
    competitor = max(abs(row[group]) for row in vectors for group in GROUPS if group != leader)
    separation = min(map(abs, values)) - competitor
    sign = np.sign(values[0])
    reasons = []
    if any(order[0] != leader for order in orders):
        reasons.append("leading_group_changed")
    if separation <= TOLERANCE:
        reasons.append("attribution_envelopes_overlap")
    if any(np.sign(value) != sign or abs(value) <= TOLERANCE for value in values):
        reasons.append("leading_direction_unstable")
    if sign <= 0:
        reasons.append("leader_not_score_increasing")
    for deletion in deletions:
        if (set(deletion) != set(GROUPS) or not all(np.isfinite(list(deletion.values())))
                or deletion[leader] <= TOLERANCE
                or any(deletion[leader] - abs(deletion[g]) <= TOLERANCE for g in GROUPS if g != leader)):
            reasons.append("donor_deletion_disagrees")
            break
    return leader, reasons, {
        "rank_orders": orders, "leader": leader, "separation": separation,
        "direction_agreement": {
            g: len({float(np.sign(row[g])) for row in vectors}) == 1 for g in GROUPS
        },
    }


class ReliabilityGate:
    def __init__(self, explainer, daily):
        self.explainer = explainer
        train = normal_training_rows(daily)
        self.support = EmpiricalSupport(train)
        self.references = tuple(select_background(train, *setting) for setting in REFERENCE_SETTINGS)
        used = {key for background in self.references for key in background.identifiers}
        keys = train[["child_id", "date"]].itertuples(index=False, name=None)
        self.deletion = select_background(train.loc[[key not in used for key in keys]], *DELETION_SETTING)

    def review(self, daily):
        """Use the frozen decision once and abstain from naming unreliable factors."""
        target = self.explainer._input(daily)
        if target is None:
            return Decision(Reliability.INSUFFICIENT_DATA, ("required_observations_missing",))
        score = float(anomaly_scores(self.explainer.pipeline, target)[0])
        threshold = self.explainer.metadata["threshold"]
        if score <= threshold:
            return Decision(Reliability.NOT_FLAGGED, ("score_at_or_below_threshold",), score, False)

        reasons = []
        diagnostics = {"decision_margin": score - threshold}
        if mask_violations(target).any(axis=None):
            return Decision(Reliability.UNCERTAIN, ("invalid_feature_relationships",),
                            score, True, FALLBACK, diagnostics)
        target_support = {name: bool(values[0]) for name, values in self.support.check(target).items()}
        diagnostics["target_support"] = target_support
        if not all(target_support.values()):
            reasons.append("target_context_sparse")

        try:
            self._check_evidence(daily, target, score, threshold, target_support, diagnostics, reasons)
        except (ArithmeticError, ValueError) as error:
            diagnostics["error"] = str(error)
            reasons.append("explanation_failed")
        if reasons:
            return Decision(Reliability.UNCERTAIN, tuple(dict.fromkeys(reasons)),
                            score, True, FALLBACK, diagnostics)

        group = diagnostics["stability"]["leader"]
        return Decision(
            Reliability.RELIABLE, (), score, True,
            "The model identified an unusual pattern in the available recorded activity. "
            f"Compared with typical recorded days, {group} increased the model's score for unusual activity.",
            diagnostics,
        )

    def _check_evidence(self, daily, target, score, threshold, target_support, diagnostics, reasons):
        full = [self.explainer.explain(daily, background, method="group_aware") for background in self.references]
        diagnostics["full_score_checks"] = self._score_checks(full, score, threshold)
        if not self._valid(full, score, threshold):
            reasons.append("score_or_structure_check_failed")
            return
        expected_span = float(np.ptp([run.expected_score for run in full]))
        diagnostics["expected_score_span"] = expected_span
        if score - threshold <= expected_span + TOLERANCE:
            reasons.append("cutoff_margin_within_background_span")

        restricted = []
        support = []
        for background in self.references:
            filtered, stats = self.support.restrict(self.explainer, target, background, MIXED_COALITIONS)
            restricted.append(filtered)
            support.append(stats)
        diagnostics["reference_support"] = support
        filtered_deletion, stats = self.support.restrict(self.explainer, target, self.deletion, DELETION_COALITIONS)
        diagnostics["deletion_support"] = stats
        if any(background is None for background in restricted):
            reasons.append("explanation_donors_sparse")
        if filtered_deletion is None:
            reasons.append("deletion_donors_sparse")

        vectors = [run.group_values for run in full]
        deletions = [self.explainer.deletion_effects(daily, self.deletion)]
        if all(target_support.values()) and all(background is not None for background in restricted):
            supported = [
                run if background.identifiers == original.identifiers else
                self.explainer.explain(daily, background, method="group_aware")
                for run, background, original in zip(full, restricted, self.references, strict=True)
            ]
            diagnostics["supported_score_checks"] = self._score_checks(supported, score, threshold)
            if not self._valid(supported, score, threshold):
                reasons.append("supported_score_or_structure_check_failed")
            else:
                vectors.extend(run.group_values for run in supported)
        if all(target_support.values()) and filtered_deletion is not None:
            deletions.append(self.explainer.deletion_effects(daily, filtered_deletion))
        _, instability, stability = attribution_checks(vectors, deletions)
        reasons.extend(instability)
        diagnostics.update(stability=stability, full_groups=vectors[:len(full)],
                           support_restricted_groups=vectors[len(full):], deletion_effects=deletions)

    @staticmethod
    def _score_checks(runs, score, threshold):
        return [
            {
                "reconstruction_error": run.reconstruction_error,
                "score_match": abs(run.anomaly_score - score) <= TOLERANCE and run.threshold == threshold,
                "threshold_match": (run.reconstructed_score > threshold) == (score > threshold)
                                   and run.is_anomaly == (score > threshold),
                "invalid_mask_rows": run.invalid_rows,
            } if run is not None else None for run in runs
        ]

    @staticmethod
    def _valid(runs, score, threshold):
        return all(
            run is not None and run.reconstructs and run.is_anomaly and not run.invalid_rows
            and abs(run.anomaly_score - score) <= TOLERANCE and run.threshold == threshold
            and set(run.group_values) == set(GROUPS)
            and np.isfinite(list(run.group_values.values()) + [run.expected_score]).all()
            for run in runs
        )

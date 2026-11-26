"""Local reliability decisions for the deployed daily and temporal explanations."""
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .behavioral_anomaly import anomaly_scores
from .behavioral_explain import SCORE_ATOL, explain, grouped_features, reference_score, verify_shap
from .behavioral_shap import ranked
from .behavioral_temporal_explain import DEFAULT_WINDOW, TemporalExplanation, explain_temporal


@dataclass(frozen=True)
class Policy:
    stability: float
    lime_r2: float


# Selected on development cases by evaluate_explanation_reliability.py.
DEFAULT_POLICY = Policy(stability=1.0, lime_r2=0.5)
PERTURBATION = 0.02
EXPLANATION_ERRORS = (KeyError, ValueError, FloatingPointError, np.linalg.LinAlgError)


@dataclass(frozen=True)
class Review:
    status: str
    group: str | None = None
    corroborated: bool = False
    reasons: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)


def perturb(matrix, amount):
    """Apply a small bounded calibration shift to observed source confidences."""
    changed = matrix.copy()
    for count, maximum, mean in (
        ("screen_incidents", "screen_max_confidence", "screen_mean_confidence"),
        ("language_flags", "language_max_risk", "language_mean_risk"),
    ):
        if not {count, maximum, mean} <= set(matrix.columns):
            continue
        high, average = matrix[maximum], matrix[mean]
        valid = matrix[count].gt(0) & average.between(0, 1) & high.between(0, 1) & average.le(high)
        # Each aggregate follows the same affine map as its underlying probabilities.
        columns = [maximum, mean]
        changed[columns] = changed[columns].astype(float)
        changed.loc[valid, columns] = changed.loc[valid, columns] * (1 - abs(amount)) + max(0, amount)
    return changed


def direction(value):
    return int(value > SCORE_ATOL) - int(value < -SCORE_ATOL)


def leader(values, sign):
    eligible = {group: value for group, value in values.items() if direction(value) == sign}
    order = ranked(eligible)
    if len(order) > 1 and abs(eligible[order[0]]) - abs(eligible[order[1]]) <= SCORE_ATOL:
        return None
    return order[0] if order else None


def valid_lime(lime, groups, metrics):
    if not isinstance(lime, Mapping) or set(lime) != groups:
        return False
    try:
        r2, variance, rmse, mae = np.asarray(metrics, dtype=float)
        return (np.isfinite([r2, variance, rmse, mae, *lime.values()]).all()
                and min(rmse, mae) >= 0 and variance > SCORE_ATOL**2)
    except (TypeError, ValueError):
        return False


def assess_daily(result, score, flagged, bundle, variants, policy=DEFAULT_POLICY):
    """Review signed SHAP separately from held-out LIME fit and agreement."""
    if not np.isfinite(score):
        return Review("insufficient_data")
    if result is None:
        return Review("invalid", reasons=("explanation_unavailable",))
    features = bundle["features"]
    groups = grouped_features(features)
    reference = reference_score(bundle["pipeline"], bundle["background"])
    if not verify_shap(result, score, groups, reference):
        return Review("invalid", reasons=("shap_reconstruction",))
    if any(not verify_shap(candidate, variant_score, groups, reference) for candidate, variant_score in variants):
        return Review("invalid", reasons=("perturbed_shap_reconstruction",))

    sign = 1 if flagged else -1
    shap = result["shap"]
    group = leader(shap, sign)
    lime = result.get("lime") or {}
    r2 = result.get("lime_r2", np.nan)
    variance = result.get("lime_variance", np.nan)
    metrics = [r2, variance, result.get("lime_rmse"), result.get("lime_mae")]
    lime_valid = valid_lime(lime, groups, metrics)
    lime_group = leader(lime, sign) if lime_valid else None
    usable_lime = lime_valid and float(r2) >= policy.lime_r2
    stable_rank = [leader(candidate["shap"], sign) == group for candidate, _ in variants]
    stable_sign = [group is not None and direction(candidate["shap"][group]) == sign
                   for candidate, _ in variants]
    vote = float(np.mean(np.array(stable_rank) & np.array(stable_sign))) if variants else 0.0
    flipped = any((value > bundle["threshold"]) != flagged for _, value in variants)
    diagnostics = {
        "shap_error": abs(result["reference"] + sum(shap.values()) - score),
        "rank_stability": float(np.mean(stable_rank)) if variants else None,
        "full_rank_stability": float(np.mean([ranked(shap) == ranked(candidate["shap"])
                                             for candidate, _ in variants])) if variants else None,
        "direction_stability": float(np.mean(stable_sign)) if variants else None,
        "decision_flip": flipped, "stability_vote": vote, "lime_r2": r2,
        "daily_effective_probes": sum(abs(value - score) > SCORE_ATOL for _, value in variants),
        "lime_rmse": result.get("lime_rmse"), "lime_mae": result.get("lime_mae"),
        "lime_valid": bool(lime_valid), "lime_group": lime_group, "shap_group": group,
        "lime_direction_agrees": bool(group and lime_valid and direction(lime[group]) == sign),
        "lime_group_agrees": bool(group and group == lime_group),
    }
    if group is None or vote < policy.stability or flipped:
        reason = "model_decision_sensitive" if flipped else "rank_or_direction_ambiguous"
        status = "conflicting" if usable_lime and lime_group and lime_group != group else "limited"
        return Review(status, reasons=(reason,), diagnostics=diagnostics)
    agrees = bool(usable_lime and group == lime_group)
    reason = ("lime_agrees" if agrees else "lime_below_fidelity_gate" if not usable_lime
              else "lime_disagrees")
    return Review("supported", group, agrees, (reason,), diagnostics)


def daily_evidence(row, score, bundle, amount=PERTURBATION):
    if not np.isfinite(score):
        return None, []
    try:
        pipeline, background = bundle["pipeline"], bundle["background"]
        result = explain(pipeline, row, background)
        variants = []
        for delta in (-amount, amount):
            changed = perturb(row, delta)
            if changed.equals(row):
                variants.append((result, score))
            else:
                variants.append((explain(pipeline, changed, background), float(anomaly_scores(pipeline, changed)[0])))
        return result, variants
    except EXPLANATION_ERRORS:
        return None, []


def temporal_evidence(history, matrix, bundle, child, day, amount=PERTURBATION):
    """Replay only scored records in the selected child's original Step 01 horizon."""
    end = pd.Timestamp(day).normalize()
    dates = pd.to_datetime(history["date"])
    rows = history.loc[history["child_id"].eq(child) & dates.between(
        end - pd.Timedelta(days=2 * DEFAULT_WINDOW - 1), end)]
    result = explain_temporal(rows, matrix, bundle, child, day)
    variants = []
    if result.status in {"stable", "isolated", "sustained"}:
        valid = rows.index[np.isfinite(rows["score"])]
        features = matrix.loc[valid, bundle["features"]]
        for delta in (-amount, amount):
            changed = perturb(features, delta)
            if changed.equals(features):
                variants.append(result)
                continue
            replay = rows.copy()
            try:
                scores = anomaly_scores(bundle["pipeline"], changed)
            except EXPLANATION_ERRORS:
                variants.append(TemporalExplanation("unsupported"))
                continue
            if not np.isfinite(scores).all():
                variants.append(TemporalExplanation("unsupported"))
                continue
            replay.loc[valid, "score"] = scores
            # Evidence about actual detectors remains attached only to the original result.
            variants.append(explain_temporal(replay, changed, bundle, child, day))
    return result, variants


def assess_temporal(result, variants, policy=DEFAULT_POLICY):
    diagnostics = {"reconstruction_error": result.reconstruction_error, "pattern": result.status,
                   "earlier_count": result.earlier_count, "recent_count": result.recent_count}
    if result.status == "insufficient":
        return Review("insufficient_data", diagnostics=diagnostics)
    if result.status == "unsupported" or result.reconstruction_error is None or (
        not np.isfinite(result.reconstruction_error) or result.reconstruction_error > SCORE_ATOL
    ):
        return Review("invalid", diagnostics=diagnostics)
    if result.status == "unclear":
        return Review("limited", reasons=(result.reason or "unclear_pattern",), diagnostics=diagnostics)
    if any(variant.status == "unsupported" for variant in variants):
        return Review("invalid", reasons=("perturbed_shap_reconstruction",), diagnostics=diagnostics)
    votes = [(candidate.status, candidate.group) == (result.status, result.group) for candidate in variants]
    vote = float(np.mean(votes)) if variants else 0.0
    diagnostics["stability_vote"] = vote
    if vote < policy.stability:
        return Review("limited", reasons=("pattern_or_group_sensitive",), diagnostics=diagnostics)
    return Review("supported", result.group, diagnostics=diagnostics)

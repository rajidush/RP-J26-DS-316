"""Compare deployed grouped SHAP explanations across observed calendar days."""
from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

from .behavioral_explain import SCORE_ATOL as SCORE_ATOL
from .behavioral_explain import explain, grouped_features, reference_score, verify_shap


DEFAULT_WINDOW = 7


@dataclass(frozen=True)
class TemporalExplanation:
    status: str
    group: str | None = None
    deltas: dict[str, float] | None = None
    score_delta: float | None = None
    reconstruction_error: float | None = None
    detectors: tuple[str, ...] = ()
    earlier_count: int = 0
    recent_count: int = 0
    reason: str | None = None


def _windows(day, window):
    end = pd.Timestamp(day).normalize()
    if window not in (7, 14):
        raise ValueError("Expected a 7- or 14-day comparison window")
    return end, end - pd.Timedelta(days=window - 1), end - pd.Timedelta(days=2 * window - 1)


def contribution_history(history, matrix, bundle):
    """Return verified signed group contributions for supplied scored records.

    Indices link to the exact feature matrix used to compute the history scores.
    An unverified scored row retains NaN contributions so it can trigger fallback.
    """
    features = bundle["features"]
    groups = grouped_features(features)
    contributions = pd.DataFrame(np.nan, index=history.index, columns=sorted(groups))
    reference = reference_score(bundle["pipeline"], bundle["background"])
    for index, score in history["score"].items():
        if not np.isfinite(score):
            continue
        result = explain(bundle["pipeline"], matrix.loc[[index], features], bundle["background"])
        if verify_shap(result, score, groups, reference):
            contributions.loc[index] = pd.Series(result["shap"])
    return contributions


def compare_contributions(history, contributions, child, day, scale, window=DEFAULT_WINDOW):
    """Compare two adjacent windows ending on the selected date, for one child."""
    end, boundary, start = _windows(day, window)
    dates = pd.to_datetime(history["date"])
    rows = history.loc[history["child_id"].eq(child) & dates.between(start, end)]
    rows = rows.assign(date=dates.loc[rows.index]).sort_values("date")
    if rows["date"].duplicated().any():
        return TemporalExplanation("unsupported")
    observed = rows.loc[np.isfinite(rows["score"])]
    observed_dates = pd.DatetimeIndex(observed["date"])
    earlier = observed_dates < boundary
    counts = {"earlier_count": int(earlier.sum()), "recent_count": int((~earlier).sum())}
    if min(counts.values()) < ceil(window * 2 / 3) or end not in observed_dates:
        return TemporalExplanation("insufficient", **counts)

    values = contributions.reindex(observed.index).set_axis(observed_dates)
    if values.empty or not np.isfinite(values.to_numpy()).all() or not np.isfinite(scale) or scale <= 0:
        return TemporalExplanation("unsupported", **counts)
    prior, recent = values.loc[earlier], values.loc[~earlier]
    deltas = recent.mean() - prior.mean()
    score_delta = observed.loc[~earlier, "score"].mean() - observed.loc[earlier, "score"].mean()
    error = abs(deltas.sum() - score_delta)
    if error > SCORE_ATOL:
        return TemporalExplanation("unsupported", **counts)
    details = dict(deltas=deltas.to_dict(), score_delta=float(score_delta),
                   reconstruction_error=float(error), **counts)

    median = prior.median()
    band = (3 * 1.4826 * (prior - median).abs().median()).clip(lower=max(4 * SCORE_ATOL, 0.1 * scale))
    higher = recent.sub(median).gt(band) & recent.gt(0)
    consecutive = pd.date_range(end=end, periods=3)
    sustained = higher.reindex(consecutive, fill_value=False).all()
    candidates = deltas[sustained & deltas.gt(band)]
    if not candidates.empty:
        group = candidates.idxmax()
        # Near ties do not support a single-factor summary.
        if (candidates.max() - candidates < band[group]).sum() > 1:
            return TemporalExplanation("unclear", reason="sustained_tie", **details)
        streak = rows.set_index("date").reindex(consecutive)
        detectors = tuple(
            method for method in ("ewma", "cusum")
            if method + "_active" in streak and streak[method + "_active"].eq(True).iloc[-1]
            and (streak[method + "_alert"].eq(True).any()
                 or (streak[method + "_active"].eq(True) & streak["flag"].eq(True)).sum() >= 2)
        )
        return TemporalExplanation("sustained", group, detectors=detectors, **details)

    elevated_dates = higher.any(axis=1)
    if elevated_dates.sum() == 1:
        contributions_on_day = (recent - median).loc[elevated_dates].iloc[0]
        candidates = contributions_on_day[higher.loc[elevated_dates].iloc[0]]
        group = candidates.idxmax()
        if (candidates.max() - candidates < band[group]).sum() == 1:
            return TemporalExplanation("isolated", group, **details)
        return TemporalExplanation("unclear", reason="isolated_tie", **details)
    if recent.sub(median).abs().le(band).to_numpy().all() and deltas.abs().le(band).all():
        return TemporalExplanation("stable", **details)
    reason = "repeated_without_sustained_evidence" if elevated_dates.sum() > 1 else "other_out_of_band_change"
    return TemporalExplanation("unclear", reason=reason, **details)


def explain_temporal(history, matrix, bundle, child, day, window=DEFAULT_WINDOW):
    """Verify only the selected child's records within the comparison horizon."""
    end, _, start = _windows(day, window)
    dates = pd.to_datetime(history["date"])
    rows = history.loc[history["child_id"].eq(child) & dates.between(start, end)]
    try:
        contributions = contribution_history(rows, matrix, bundle)
    except (ValueError, KeyError, FloatingPointError, np.linalg.LinAlgError):
        return TemporalExplanation("unsupported")
    return compare_contributions(rows, contributions, child, day, bundle["scale"], window)

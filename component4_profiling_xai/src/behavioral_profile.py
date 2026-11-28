"""Causal calendar-window profiles and sequential score monitoring."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .behavioral_anomaly import FEATURE_SET_A, anomaly_scores, assessable_days
from .behavioral_shap import TOTALS

BUNDLE = Path(__file__).resolve().parents[1] / "outputs" / "function02_upgrade" / "selected.joblib"

ANCHORS = (
    "screen_incidents", "screen_night_incidents", "screen_escalated_incidents",
    "language_flags", "language_night_flags", "language_bullying_flags",
)
PERSONAL = [f"{name}__robust_z" for name in ANCHORS]
TEMPORAL = [f"{name}__active_days" for name in ANCHORS]
# Cross-source totals overlap with source features and can double-count incidents.
FUSED = [f for f in FEATURE_SET_A if f not in {"total_risk_events", "night_risk_events"}]


@dataclass(frozen=True)
class ProfileConfig:
    days: int = 14
    frequency_days: int = 7
    minimum: int = 5
    count_scale_floor: float = 1.0
    clip_z: float = 12.0

    def __post_init__(self):
        if not 1 <= self.minimum <= self.days or self.frequency_days < 1:
            raise ValueError("Invalid baseline observation windows")
        if self.count_scale_floor <= 0 or self.clip_z <= 0:
            raise ValueError("Invalid robust score settings")


def profiles(daily, config=ProfileConfig()):
    """Return prior-only statistics and model inputs, indexed like daily."""
    blocks = []
    for _, child in daily.groupby("child_id", sort=False):
        dates = pd.to_datetime(child["date"])
        if dates.duplicated().any():
            raise ValueError("Expected one row per child and local date")
        child = child.assign(_day=dates).sort_values("_day")
        observed = child.set_index("_day")[FEATURE_SET_A].astype(float)
        for prefix, coverage in (("screen_", "screen_monitor_observed"),
                                 ("language_", "language_monitor_observed")):
            columns = observed.columns[observed.columns.str.startswith(prefix)]
            if coverage in child:
                covered = child[coverage].astype("string").str.lower().isin(["true", "1", "1.0"])
                observed.loc[~covered.to_numpy(), columns] = np.nan
        for total, (left, right) in TOTALS.items():
            observed[total] = observed[left] + observed[right]
        window = observed.rolling(f"{config.days}D", closed="left", min_periods=config.minimum)
        count = observed.rolling(f"{config.days}D", closed="left", min_periods=0).count()
        median = window.median()
        stats = {"n": count, "median": median, "mean": window.mean(), "std": window.std(ddof=0)}
        anchor_window = observed[list(ANCHORS)].rolling(
            f"{config.days}D", closed="left", min_periods=config.minimum)
        mad = anchor_window.apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True)
        delta = observed - median
        z = (delta[list(ANCHORS)] / (1.4826 * mad).clip(lower=config.count_scale_floor))
        active = observed[list(ANCHORS)].gt(0).where(observed[list(ANCHORS)].notna())
        frequency_window = active.rolling(f"{config.frequency_days}D", closed="left", min_periods=1)
        frequency = frequency_window.sum()
        exposure = frequency_window.count()
        features = pd.concat([observed, z.clip(-config.clip_z, config.clip_z).add_suffix("__robust_z"),
                              frequency.add_suffix("__active_days")], axis=1)
        detail = pd.concat({
            **stats, "mad": mad, "delta": delta, "robust_z": z,
            "recent_active_days": frequency, "recent_recorded_days": exposure,
        }, axis=1)
        features.index = detail.index = child.index
        blocks.append((features, detail))
    return tuple(pd.concat([block[i] for block in blocks]).reindex(daily.index) for i in range(2))


@dataclass(frozen=True)
class TrendConfig:
    threshold: float = 2.0
    cusum_threshold: float | None = None
    alpha: float = 0.3
    drift: float = 0.5
    min_run: int = 2


def temporal(dates, scores, flags, center, scale, config=TrendConfig()):
    """One-sided EWMA and CUSUM. Missing or nonconsecutive dates reset runs."""
    ewma = cusum = 0.0
    run = [0, 0]
    latched = [False, False]
    streak = 0
    previous = None
    output = []
    for day, score, flag in zip(pd.to_datetime(dates), scores, flags, strict=True):
        valid = np.isfinite(score)
        if previous is None or (day - previous).days != 1 or not valid:
            ewma = cusum = 0.0
            run = [0, 0]
            latched = [False, False]
            streak = 0
        prior_streak = streak
        if valid:
            value = float(np.clip((score - center) / max(scale, 1e-12), -10, 10))
            ewma = config.alpha * value + (1 - config.alpha) * ewma
            cusum = max(0.0, cusum + value - config.drift)
            streak = streak + 1 if flag else 0
        alerts = []
        for i, statistic in enumerate((ewma, cusum)):
            threshold = config.cusum_threshold if i == 1 and config.cusum_threshold is not None else config.threshold
            over = valid and statistic > threshold
            run[i] = run[i] + 1 if over else 0
            active = run[i] >= config.min_run
            alerts.extend((active, active and not latched[i]))
            latched[i] = active
        output.append((ewma if valid else np.nan, cusum if valid else np.nan,
                       prior_streak, *alerts, max(run)))
        previous = day if valid else None
    result = pd.DataFrame(output, columns=["ewma", "cusum", "prior_streak", "ewma_active",
                                           "ewma_alert", "cusum_active", "cusum_alert", "change_days"])
    history = pd.Series(flags, index=pd.DatetimeIndex(pd.to_datetime(dates)), dtype="Float64")
    rolling = history.rolling("30D", closed="left", min_periods=1)
    result["prior_flagged"] = rolling.sum().to_numpy()
    result["prior_anomaly_frequency"] = rolling.mean().to_numpy()
    return result


def priority(urgent, score, flagged, prior_flags, sustained, repeated_threshold=2):
    if urgent:
        return "Urgent safety review", "A recorded signal matched a separate safety rule."
    if not np.isfinite(score):
        return "Insufficient data", "Available monitoring records do not support a model assessment."
    if sustained or (flagged and prior_flags >= repeated_threshold):
        return "Priority review", "The score trend stayed elevated or unusual days recurred."
    if flagged:
        return "Needs attention", "The selected day crossed the model’s unusual-activity threshold."
    return "No unusual activity detected", "The available activity stayed below the alert threshold."


def score_history(daily, matrix, bundle):
    """Replay the same detector used for research, respecting per-child boundaries."""
    scores = np.full(len(daily), np.nan)
    observed = assessable_days(daily, matrix)
    if observed.any():
        scores[observed.to_numpy()] = anomaly_scores(
            bundle["pipeline"], matrix.loc[observed, bundle["features"]])
    flags = pd.array(scores > bundle["threshold"], dtype="boolean")
    flags[~observed.to_numpy()] = pd.NA
    scored = daily[["child_id", "date"]].assign(score=scores, flag=flags)
    parts = []
    for _, child in scored.groupby("child_id", sort=False):
        ordered = child.sort_values("date")
        trend = temporal(ordered["date"], ordered["score"], ordered["flag"], bundle["center"],
                         bundle["scale"], TrendConfig(**bundle["trend"]))
        trend.index = ordered.index
        parts.append(trend)
    return scored.join(pd.concat(parts))

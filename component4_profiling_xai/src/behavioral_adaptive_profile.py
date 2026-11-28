"""Prior-only, safety-aware comparisons for recorded activity."""
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .behavioral_anomaly import assessable_days
from .behavioral_profile import ANCHORS
from .urgent_alerts import urgent_alerts

OPTIONAL_CONCERN_COLUMNS = ("screen_violence_incidents", "screen_explicit_visual_incidents",
                            "language_sexual_harassment_flags")
REQUIRED_CONCERN_COLUMNS = ("screen_escalated_incidents", "language_bullying_flags",
                            "language_threat_flags", "language_hate_identity_flags")
CONCERN_COLUMNS = (*REQUIRED_CONCERN_COLUMNS, *OPTIONAL_CONCERN_COLUMNS)


@dataclass(frozen=True)
class AdaptiveConfig:
    horizon: int = 14
    confirmation: int = 2
    minimum: int = 5
    recovery: int = 2
    abrupt_z: float = 3.0
    shift_counts: float = 2.0

    def __post_init__(self):
        if not 2 <= self.minimum <= self.horizon or not 2 <= self.confirmation <= self.horizon:
            raise ValueError("Invalid adaptive observation window")
        if self.recovery < 1 or self.abrupt_z <= 0 or self.shift_counts <= 0:
            raise ValueError("Invalid adaptive update policy")


def admission_evidence(daily, matrix):
    """Determine row-level coverage and risk evidence before sequence rules."""
    urgent = urgent_alerts(daily)["is_urgent"]
    # Optional fields carry evidence when recorded. An unavailable category is
    # never treated as an observed zero or used to infer safety.
    concern_values = daily.reindex(columns=CONCERN_COLUMNS).apply(pd.to_numeric, errors="coerce")
    available = (assessable_days(daily, matrix) & matrix[list(ANCHORS)].notna().all(axis=1)
                 & concern_values[list(REQUIRED_CONCERN_COLUMNS)].notna().all(axis=1))
    return available, urgent, concern_values.gt(0)


def adaptive_profiles(daily, matrix, flags, config=AdaptiveConfig()):
    """Return per-record snapshots and an update audit, aligned to input indices."""
    counts = matrix[list(ANCHORS)]
    available, urgent, concerns = admission_evidence(daily, matrix)
    output = []
    for _, child in daily.groupby("child_id", sort=False):
        ordered = child.assign(_day=pd.to_datetime(child["date"])).sort_values("_day")
        if ordered["_day"].duplicated().any():
            raise ValueError("Expected one row per child and date")
        accepted, pending = deque(), deque()
        accepted_since = clean_run = 0
        last = None
        last_exclusion = None
        frozen = False
        regime = None
        for index, day in ordered["_day"].items():
            cutoff = day - pd.Timedelta(days=config.horizon)
            while accepted and accepted[0][0] < cutoff:
                accepted.popleft()
            if last is not None and (day - last).days != 1:
                pending.clear()
                clean_run = 0
            last = day
            row = counts.loc[index].to_numpy(dtype=float)
            values = np.asarray([item[2] for item in accepted])
            enough = len(accepted) >= config.minimum
            median = np.median(values, axis=0) if enough else np.full(len(ANCHORS), np.nan)
            scale = np.maximum(1, 1.4826 * np.median(abs(values - median), axis=0)) if enough else median
            z = (row - median) / scale
            abrupt = enough and bool(available.at[index]) and np.max(abs(z)) >= config.abrupt_z
            if regime is None and enough:
                regime = median.copy()
            # Detect a change in admitted observations and emit a single onset.
            shifted = enough and np.max(abs(median - regime)) >= config.shift_counts
            if shifted:
                regime = median.copy()
            previous_guard = last_exclusion is not None and last_exclusion >= cutoff
            was_frozen = frozen
            signals = tuple(concerns.columns[concerns.loc[index]])
            status = "accepted"
            admitted = []
            if urgent.at[index] or signals:
                status = "urgent" if urgent.at[index] else "concern"
                last_exclusion = day
                frozen = True
                clean_run = 0
                pending.clear()
            elif not available.at[index]:
                status = "missing"
                pending.clear()
                clean_run = 0
            else:
                clean_run += 1
                if frozen and clean_run < config.recovery:
                    status = "recovery_wait"
                else:
                    frozen = False
                    flag = flags.get(index, pd.NA)
                    if abrupt or (pd.notna(flag) and flag):
                        if pending and np.max(abs(row - pending[0][2])) > config.shift_counts:
                            pending.clear()
                        pending.append((day, index, row))
                        if len(pending) >= config.confirmation:
                            admitted = list(pending)
                            pending.clear()
                            status = "confirmed_update"
                        else:
                            status = "probation"
                    else:
                        pending.clear()
                        admitted = [(day, index, row)]
                        if was_frozen:
                            status = "recovered"
                accepted.extend(admitted)
                accepted_since += len(admitted)
            output.append({
                "index": index, "median": tuple(median), "robust_z": tuple(z),
                "count": len(values), "available": bool(available.at[index]),
                "decision": status, "concern_fields": signals,
                "admitted_indices": tuple(item[1] for item in admitted),
                "prior_total_admitted": accepted_since - len(admitted),
                "frozen_before": was_frozen, "frozen_after": frozen,
                "recent_exclusion": previous_guard or status in {"urgent", "concern"},
                "shift": bool(shifted), "abrupt": bool(abrupt),
            })
    if not output:
        return pd.DataFrame(index=daily.index)
    return pd.DataFrame(output).set_index("index").reindex(daily.index)

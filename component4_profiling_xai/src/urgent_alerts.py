"""Function 2 - urgent safety alerts: a fixed rule, separate from anomaly detection.

A child-day is urgent when any of these daily counts is above zero:

    language_urgent_flags                Component 2 marked a flag urgent
    language_grooming_flags              Component 2 grooming flag
    language_self_harm_flags             Component 2 self-harm flag
    screen_self_harm_imagery_incidents   Component 1 self-harm imagery incident

This is the rule evaluated in the Step 4A research notebook, ported unchanged.
It never reads anomaly scores or synthetic labels, and the anomaly model never
reads its output: the dashboard shows the two side by side, never merged.

    from src.urgent_alerts import urgent_alerts
    alerts = urgent_alerts(daily)       # one row per input row
"""
from __future__ import annotations

import pandas as pd

URGENT_RULE_COLUMNS = {
    "language_urgent_flags": "urgent language flag",
    "language_grooming_flags": "grooming language flag",
    "language_self_harm_flags": "self-harm language flag",
    "screen_self_harm_imagery_incidents": "self-harm imagery on screen",
}


def urgent_alerts(daily: pd.DataFrame) -> pd.DataFrame:
    """Apply the urgent rule to daily rows.

    Every column in URGENT_RULE_COLUMNS must be present. A NaN count means
    "not available" and does not trigger the rule. Returns, in input order,
    child_id / date (when present), is_urgent and reasons (readable, "; "-joined).
    """
    missing = [c for c in URGENT_RULE_COLUMNS if c not in daily.columns]
    if missing:
        raise ValueError(f"missing urgent-rule columns: {missing}")
    counts = daily[list(URGENT_RULE_COLUMNS)].apply(pd.to_numeric, errors="raise")
    hits = counts.fillna(0) > 0
    out = daily[[c for c in ("child_id", "date") if c in daily.columns]].copy()
    out["is_urgent"] = hits.any(axis=1)
    out["reasons"] = [
        "; ".join(f"{URGENT_RULE_COLUMNS[c]} ({int(counts.at[i, c])})"
                  for c in URGENT_RULE_COLUMNS if hits.at[i, c])
        for i in daily.index
    ]
    return out

"""Synthetic chronological development and locked Step 03 holdout."""
import argparse
import hashlib
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
from time import perf_counter
import tracemalloc

import joblib
import matplotlib
import numpy as np
import pandas as pd

from .evaluate_temporal_explain import json_value
from .src.behavioral_adaptive_profile import AdaptiveConfig, CONCERN_COLUMNS, adaptive_profiles
from .src.behavioral_anomaly import assessable_days
from .src.behavioral_profile import ANCHORS, BUNDLE, ProfileConfig, profiles, score_history
from .src.urgent_alerts import URGENT_RULE_COLUMNS

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "adaptive_profiling_step03"
SCENARIOS = ("stable", "gradual", "abrupt", "spike", "screen_concern", "language_concern",
             "repeated_anomaly", "missing", "cold_start", "false_positive", "recovery", "different_children")
ONSET, RECOVERY, LENGTH = 14, 30, 46


def save(name, data):
    (OUT / name).write_text(json.dumps(json_value(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def scenario_rows(anchor, scenario, end, level, magnitude):
    days = pd.date_range(end=end, periods=LENGTH)
    daily = pd.DataFrame([anchor.to_dict()] * LENGTH)
    daily["date"] = days
    daily["is_weekend"] = days.dayofweek >= 5
    daily["screen_monitor_observed"] = True
    daily["language_monitor_observed"] = True
    for column in set(CONCERN_COLUMNS) | set(URGENT_RULE_COLUMNS):
        daily[column] = 0
    volume = np.full(LENGTH, level, dtype=int)
    language = int(anchor["language_flags"])
    affected = np.zeros(LENGTH, dtype=bool)
    if scenario == "gradual":
        volume[ONSET:] += np.minimum(magnitude, 1 + np.arange(LENGTH - ONSET) // 2)
    elif scenario == "abrupt":
        volume[ONSET:] += magnitude
    elif scenario == "spike":
        volume[ONSET] += magnitude * 2
    elif scenario in {"screen_concern", "recovery"}:
        stop = RECOVERY if scenario == "recovery" else LENGTH
        volume[ONSET:stop] += magnitude
        affected[ONSET:stop] = True
        daily.loc[affected, "screen_violence_incidents"] = 1
        if scenario == "recovery":
            daily.loc[ONSET, "screen_self_harm_imagery_incidents"] = 1
    elif scenario == "language_concern":
        affected[ONSET:] = True
        daily.loc[affected, "language_bullying_flags"] = 1
    elif scenario == "repeated_anomaly":
        affected[ONSET:RECOVERY] = True
        volume[affected] += np.resize([magnitude, magnitude * 2], affected.sum())
    daily["screen_incidents"] = volume
    daily["screen_night_incidents"] = np.minimum(volume, max(0, level - 1))
    daily["screen_flagged_frames"] = 5 * volume
    daily["screen_flagged_seconds"] = 2 * volume
    daily["screen_max_confidence"] = np.where(volume > 0, .7, 0)
    daily["screen_mean_confidence"] = np.where(volume > 0, .7, 0)
    for column in ("screen_blocked_incidents", "screen_dismissed_incidents"):
        daily[column] = 0
    daily["language_night_flags"] = min(language, int(anchor["language_night_flags"]))
    daily["language_flags"] = language
    if scenario == "language_concern":
        daily.loc[affected, "language_flags"] += magnitude
    indices = daily.index
    if scenario == "missing":
        daily = daily.drop(index=range(ONSET, ONSET + 6))
        daily.loc[ONSET + 6:ONSET + 8, "language_monitor_observed"] = False
    elif scenario == "cold_start":
        daily = daily.tail(4)
    return daily, set(indices[affected])


def prepare_cases(dataset, split, bundle, magnitude):
    first = dataset.loc[dataset["split"].eq(split), "date"].min()
    last = dataset.loc[dataset["split"].eq(split), "date"].max()
    anchor_rows = dataset.loc[dataset["date"].lt(first) & dataset["is_anomaly"].eq(0)]
    anchors = anchor_rows.sort_values(["child_id", "date"]).groupby("child_id").tail(1).head(4)
    profile = ProfileConfig(**bundle["profile"])
    for offset, (_, anchor) in enumerate(anchors.iterrows()):
        for scenario in SCENARIOS:
            level = (offset * 3 if scenario == "different_children" else offset + 1)
            daily, affected = scenario_rows(anchor, scenario, last - pd.Timedelta(days=offset),
                                            level, magnitude + offset % 2)
            matrix, detail = profiles(daily, profile)
            history = score_history(daily, matrix, bundle)
            flags = history["flag"].copy()
            if scenario == "false_positive":
                flags.loc[ONSET:ONSET + 5] = True
            if scenario == "repeated_anomaly":
                flags.loc[list(affected)] = True
            yield scenario, daily, matrix, detail, flags, affected, level, magnitude + offset % 2


def fixed_comparison(daily, matrix, detail, shift_counts):
    medians = detail["median"][list(ANCHORS)]
    regime = None
    shifts = []
    for row in medians.to_numpy():
        enough = np.isfinite(row).all()
        if enough and regime is None:
            regime = row.copy()
        changed = bool(enough and np.max(abs(row - regime)) >= shift_counts)
        shifts.append(changed)
        if changed:
            regime = row.copy()
    return pd.DataFrame({"median": list(medians.itertuples(index=False, name=None)), "shift": shifts,
                         "admitted_indices": [(index,) if available else () for index, available in
                                              assessable_days(daily, matrix).items()]}, index=daily.index)


def first_delay(condition, start):
    candidates = condition.loc[condition.index >= start]
    return int(candidates.index[candidates][0] - start) if candidates.any() else None


def measures(result, scenario, affected, level, magnitude):
    median = pd.DataFrame(result["median"].tolist(), index=result.index, columns=ANCHORS)
    valid = median.notna().all(axis=1)
    changed = scenario in {"gradual", "abrupt"}
    admitted = {index for indices in result["admitted_indices"] for index in indices}
    if scenario == "language_concern":
        anchor = median["language_flags"].dropna().iloc[0] if valid.any() else np.nan
        concerning_center = median["language_flags"]
    else:
        anchor = level
        concerning_center = median["screen_incidents"]
    exposed = concerning_center.reindex(sorted(affected)).dropna()
    during_concern = exposed.ge(anchor + magnitude / 2).any()
    return {
        "coverage": float(valid.mean()), "labeled_rows": len(affected),
        "labeled_rows_admitted": len(affected & admitted), "normalized": bool(during_concern),
        "drift_expected": changed,
        "drift_delay": first_delay(result["shift"], ONSET) if changed else None,
        "false_drift_alerts": int(result.loc[result.index < ONSET, "shift"].sum() if changed
                                  else result["shift"].sum()),
        "adaptation_delay": first_delay(median["screen_incidents"].sub(level + magnitude).abs().le(1), ONSET)
        if changed else None,
        "recovery_delay": first_delay(median["screen_incidents"].sub(level).abs().le(1) & valid, RECOVERY)
        if scenario == "recovery" else None,
        "update_recovery_delay": first_delay(result["decision"].isin(["recovered", "accepted", "confirmed_update"]),
                                             RECOVERY) if scenario == "recovery" and "decision" in result else
        (0 if scenario == "recovery" else None),
    }


def run(cases, policy, include_fixed=False):
    records, traces = [], []
    for scenario, daily, matrix, detail, flags, affected, level, magnitude in cases:
        start = perf_counter()
        adaptive = adaptive_profiles(daily, matrix, flags, policy)
        elapsed = (perf_counter() - start) * 1000
        methods = [("adaptive", adaptive, elapsed)]
        if include_fixed:
            start = perf_counter()
            fixed = fixed_comparison(daily, matrix, detail, policy.shift_counts)
            methods.insert(0, ("fixed", fixed, (perf_counter() - start) * 1000))
        for method, result, ms in methods:
            records.append({"scenario": scenario, "child": daily["child_id"].iloc[0], "method": method,
                            "runtime_ms": ms, **measures(result, scenario, affected, level, magnitude)})
            if include_fixed:
                trace = result.assign(date=daily["date"], scenario=scenario, method=method,
                                      child=daily["child_id"], labeled=daily.index.isin(affected))
                traces.append(trace.rename_axis("scenario_day"))
    return pd.DataFrame(records), traces


def summary(table):
    drift = table.loc[table["drift_expected"]]
    concern = table.loc[table["labeled_rows"].gt(0)]
    return {
        "cases": len(table), "coverage": table["coverage"].mean(),
        "labeled_rows": int(table["labeled_rows"].sum()),
        "labeled_rows_admitted": int(table["labeled_rows_admitted"].sum()),
        "normalization_cases": int(concern["normalized"].sum()), "concern_cases": len(concern),
        "drift_cases": len(drift), "detected_drifts": int(drift["drift_delay"].count()),
        "false_alerts": int(table["false_drift_alerts"].sum()),
        "median_drift_delay": drift["drift_delay"].median(),
        "median_adaptation_delay": drift["adaptation_delay"].median(),
        "median_recovery_delay": table["recovery_delay"].median(),
        "median_update_recovery_delay": table["update_recovery_delay"].median(),
        "median_comparison_ms": table["runtime_ms"].median(),
    }


def memory(cases, policy):
    _, daily, matrix, details, flags, *_ = max(cases, key=lambda case: len(case[1]))
    results = {}
    for name, operation in (
        ("fixed", lambda: fixed_comparison(daily, matrix, details, policy.shift_counts)),
        ("adaptive", lambda: adaptive_profiles(daily, matrix, flags, policy)),
    ):
        tracemalloc.start()
        operation()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[name] = peak
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "holdout"))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    bundle = joblib.load(BUNDLE)
    model_hash = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    if args.stage == "validate":
        data = pd.read_csv(ROOT / "data" / "behavioral_daily.csv", parse_dates=["date"])
        cases = list(prepare_cases(data, "validation", bundle, magnitude=4))
        outcomes = []
        for horizon, confirmation in product((14, 21, 28), (2, 3)):
            config = AdaptiveConfig(horizon=horizon, confirmation=confirmation)
            table, _ = run(cases, config)
            outcomes.append({"horizon": horizon, "confirmation": confirmation, **summary(table)})
        # Keep safeguards first, then detection and speed; prefer smaller state on ties.
        grid = pd.DataFrame(outcomes)
        ordering = ["labeled_rows_admitted", "false_alerts", "detected_drifts", "median_adaptation_delay",
                    "coverage", "horizon", "confirmation"]
        best = grid.sort_values(ordering, ascending=[True, True, False, True, False, True, False]).iloc[0]
        locked = AdaptiveConfig(horizon=int(best["horizon"]), confirmation=int(best["confirmation"]))
        baseline, _ = run(cases, locked, include_fixed=True)
        grid.to_csv(OUT / "validation_grid.csv", index=False)
        save("validation_summary.json", {key: summary(group) for key, group in baseline.groupby("method")})
        save("policy_lock.json", {"config": asdict(locked), "model_sha256": model_hash})
        print(locked)
        return
    lock = json.loads((OUT / "policy_lock.json").read_text())
    policy = AdaptiveConfig(**lock["config"])
    if policy != AdaptiveConfig() or lock["model_sha256"] != model_hash:
        raise RuntimeError("The validated policy and deployed model must match the lock")
    data = pd.read_csv(ROOT / "data" / "step4a_heldout_seed20261020" / "behavioral_daily.csv", parse_dates=["date"])
    cases = list(prepare_cases(data, "test", bundle, magnitude=5))
    table, traces = run(cases, policy, include_fixed=True)
    trace = pd.concat(traces)
    table.to_csv(OUT / "holdout_comparisons.csv", index=False)
    trace.to_csv(OUT / "holdout_audit.csv")
    metrics = {key: summary(group) for key, group in table.groupby("method")}
    metrics["comparison_peak_python_bytes"] = memory(cases, policy)
    save("metrics.json", metrics)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, scenario in zip(axes, ("abrupt", "screen_concern"), strict=True):
        sample = trace.loc[trace["scenario"].eq(scenario)]
        for label, group in sample.loc[sample["child"].eq(sample["child"].iloc[0])].groupby("method"):
            axis.plot(group.index, group["median"].str[0], label=label)
        axis.axvline(ONSET, color="gray", linestyle=":", label="Injection onset")
        axis.set(title=scenario.replace("_", " ").capitalize(), xlabel="Scenario day")
    axes[0].set_ylabel("Prior median screen incidents")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUT / "holdout_baselines.png", dpi=160)
    print(json.dumps(json_value(metrics), indent=2))


if __name__ == "__main__":
    main()

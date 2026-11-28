"""Offline chronological evaluation for the deployed Function 02 explanations."""
from pathlib import Path
from time import perf_counter
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

from .src.behavioral_explain import explain
from .src.behavioral_profile import BUNDLE, ProfileConfig, profiles, score_history
from .src.behavioral_temporal_explain import (
    contribution_history, explain_temporal,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "temporal_xai_step01"
SCENARIOS = {
    "stable": "stable", "spike": "isolated", "sustained": "sustained",
    "gradual": "sustained", "missing": "insufficient",
}
GROUPS = {"screen": "Screen activity", "language": "Language activity"}


def controlled(base, end, scenario, source):
    """Repeat a fully observed real synthetic row and inject one source at a time."""
    dates = pd.date_range(end=end, periods=28)
    daily = pd.DataFrame([base.to_dict()] * len(dates))
    daily["date"] = dates
    daily["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    counts = np.zeros(len(dates), dtype=int)
    if scenario == "spike":
        counts[-1] = 6
    elif scenario == "sustained":
        counts[-7:] = 6
    elif scenario == "gradual":
        counts[-7:] = [1, 1, 2, 3, 4, 5, 6]
    if source == "screen":
        increments = {"screen_incidents": 1, "screen_night_incidents": 1,
                      "screen_flagged_frames": 12, "screen_flagged_seconds": 4}
    else:
        increments = {"language_flags": 1, "language_bullying_flags": 1,
                      "language_night_flags": 1}
    for column, multiplier in increments.items():
        daily[column] += counts * multiplier
    if scenario == "missing":
        daily = daily.drop(daily.index[-7:-1])
    return daily


def perturb(daily):
    changed = daily.copy()
    for column in ("screen_max_confidence", "language_max_risk"):
        recorded = changed[column].gt(0)
        changed.loc[recorded, column] = (changed.loc[recorded, column] + 0.02).clip(upper=1)
    return changed


def prepare(daily, bundle):
    matrix, _ = profiles(daily, ProfileConfig(**bundle["profile"]))
    return score_history(daily, matrix, bundle), matrix


def daily_result(history, matrix, bundle):
    selected = history.index[-1]
    flag = history.at[selected, "flag"]
    if pd.isna(flag):
        return {}, None, None, np.nan, np.nan
    start = perf_counter()
    output = explain(bundle["pipeline"], matrix.loc[[selected], bundle["features"]], bundle["background"])
    elapsed = (perf_counter() - start) * 1000
    signed = pd.Series(output["shap"])
    sign = 1 if flag else -1
    group = signed.abs().where(signed * sign > 1e-8).idxmax() if (signed * sign > 1e-8).any() else None
    lime_agrees = bool(output["lime"][group] * sign > 1e-8) if group else None
    error = abs(output["reference"] + signed.sum() - history.at[selected, "score"])
    return signed.to_dict(), group, lime_agrees, elapsed, error


def benchmark(daily, bundle, window, phase, scenario, source=None, control=None):
    history, matrix = prepare(daily, bundle)
    altered, altered_matrix = prepare(perturb(daily), bundle)
    child, day = history.iloc[-1][["child_id", "date"]]
    start = perf_counter()
    result = explain_temporal(history, matrix, bundle, child, day, window)
    temporal_ms = (perf_counter() - start) * 1000
    repeat = explain_temporal(altered, altered_matrix, bundle, child, day, window)
    daily_shap, daily_group, lime_agrees, daily_ms, daily_error = daily_result(history, matrix, bundle)
    _, daily_repeat, _, _, _ = daily_result(altered, altered_matrix, bundle)
    dates = pd.to_datetime(history["date"])
    comparison = dates.gt(day - pd.Timedelta(days=window * 2))
    score_changes = (history["score"] - altered["score"]).abs().gt(1e-8)
    record = {
        "phase": phase, "child": child, "date": day, "scenario": scenario,
        "window": window, "expected": SCENARIOS.get(scenario), "temporal_status": result.status,
        "injected_group": GROUPS.get(source) if control is not None else None,
        "temporal_group": result.group, "daily_group": daily_group,
        "daily_lime_agrees": lime_agrees, "daily_ms": daily_ms, "temporal_ms": temporal_ms,
        "daily_error": daily_error, "daily_score_perturbed": bool(score_changes.iloc[-1]),
        "window_scores_perturbed": bool(score_changes.loc[comparison].any()),
        "temporal_stable": (result.status, result.group) == (repeat.status, repeat.group),
        "daily_stable": daily_group == daily_repeat,
        "score_delta": result.score_delta, "reconstruction_error": result.reconstruction_error,
        "detectors": " ".join(result.detectors), "earlier_count": result.earlier_count,
        "recent_count": result.recent_count,
    }
    record.update({f"delta_{key}": value for key, value in (result.deltas or {}).items()})
    if control is not None:
        counter, counter_matrix = prepare(control, bundle)
        observed = history.tail(7)
        matched = counter.loc[observed.index]
        difference = (contribution_history(observed, matrix, bundle)
                      - contribution_history(matched, counter_matrix, bundle)).mean()
        uplift = (observed["score"] - matched["score"]).mean()
        target = difference.idxmax() if uplift > 1e-8 and difference.max() > 1e-8 else None
        record["verified_group"] = target
        record["paired_score_delta"] = uplift
        record["paired_error"] = abs(difference.sum() - uplift)
        if target:
            record["daily_direction_agrees"] = daily_shap.get(target, 0) > 1e-8
            record["temporal_direction_agrees"] = ((result.deltas or {}).get(target, 0) > 1e-8
                                                   if result.deltas else None)
    return record


def controlled_benchmark(dataset, bundle, window, phase, first, last):
    examples = dataset.loc[dataset["date"].lt(first) & dataset["is_anomaly"].eq(0)]
    anchors = examples.groupby("child_id").tail(1).iloc[:4]
    records = []
    for offset, (_, base) in enumerate(anchors.iterrows()):
        end = last - pd.Timedelta(days=offset)
        for scenario in SCENARIOS:
            sources = GROUPS if scenario in {"spike", "gradual", "sustained"} else ("screen",)
            for source in sources:
                daily = controlled(base, end, scenario, source)
                control = controlled(base, end, "stable", source) if len(sources) == 2 else None
                records.append(benchmark(daily, bundle, window, phase, scenario, source, control))
    return pd.DataFrame(records)


def summary(table):
    verified = table.loc[table["verified_group"].notna()] if "verified_group" in table else table.iloc[:0]
    both = table.loc[table["temporal_group"].notna() & table["daily_group"].notna()]
    result = {
        "cases": len(table),
        "status_counts": table["temporal_status"].value_counts().to_dict(),
        "temporal_fidelity_n": int(table["reconstruction_error"].count()),
        "temporal_fidelity_max_error": table["reconstruction_error"].max(),
        "named_group_agreement_n": len(both),
        "named_group_agreement": both["temporal_group"].eq(both["daily_group"]).mean(),
        "daily_lime_agreement": table["daily_lime_agrees"].mean(),
        "daily_fidelity_max_error": table["daily_error"].max(),
        "verified_target_n": len(verified),
        "detector_overlap_n": int(table["detectors"].ne("").sum()),
    }
    for method in ("daily", "temporal"):
        named = table[method + "_group"].notna()
        direction = method + "_direction_agrees"
        perturbed = table["daily_score_perturbed" if method == "daily" else "window_scores_perturbed"]
        result[method] = {
            "named_group_n": int(named.sum()), "stable": table[method + "_stable"].mean(),
            "stable_when_named": table.loc[named, method + "_stable"].mean(),
            "score_perturbed_n": int(perturbed.sum()),
            "stable_when_score_perturbed": table.loc[perturbed, method + "_stable"].mean(),
            "median_ms": table[method + "_ms"].median(), "p95_ms": table[method + "_ms"].quantile(.95),
            "verified_target_hits": (int(verified[method + "_group"].eq(verified["verified_group"]).sum())
                                     if len(verified) else 0),
            "direction_n": int(table[direction].count()) if direction in table else 0,
            "direction_agreement": table[direction].dropna().mean() if direction in table else np.nan,
        }
    if table["expected"].notna().any():
        injected = table.loc[table["injected_group"].notna()]
        result["classification_accuracy"] = table["temporal_status"].eq(table["expected"]).mean()
        result["injected_n"] = len(injected)
        result["daily_injected_hits"] = int(injected["daily_group"].eq(injected["injected_group"]).sum())
        result["temporal_injected_hits"] = int(injected["temporal_group"].eq(injected["injected_group"]).sum())
        result["paired_max_error"] = table["paired_error"].max()
    return result


def json_value(value):
    """Preserve small fidelity errors and emit null for undefined rates."""
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    return None if pd.isna(value) else value


def plot_results(controlled_test, natural, selected):
    cross = pd.crosstab(controlled_test["scenario"], controlled_test["temporal_status"])
    axis = cross.reindex(list(SCENARIOS)).plot.bar(stacked=True, rot=0, figsize=(9, 4))
    axis.set(xlabel="Injected scenario", ylabel="Cases", title=f"Held-out temporal summaries ({selected}-day windows)")
    axis.legend(title="Temporal result")
    axis.figure.tight_layout()
    axis.figure.savefig(OUT / "heldout_scenarios.png", dpi=160)
    plt.close(axis.figure)

    coverage = pd.DataFrame({
        "Group named": [natural["daily_group"].notna().sum(), natural["temporal_group"].notna().sum()],
        "Group withheld": [natural["daily_group"].isna().sum(), natural["temporal_group"].isna().sum()],
    }, index=["Selected-day SHAP", "Temporal SHAP"])
    axis = coverage.plot.barh(stacked=True, color=["#2166ac", "#d9e5f1"], figsize=(8, 3.5))
    for bars in axis.containers:
        axis.bar_label(bars, label_type="center", color="black" if bars is axis.containers[1] else "white")
    axis.set(xlabel="Children", title="Named-group coverage at unmodified holdout endpoints")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.47), ncol=2, frameon=False)
    axis.figure.tight_layout()
    axis.figure.savefig(OUT / "heldout_coverage.png", dpi=160)
    plt.close(axis.figure)


def main():
    bundle = joblib.load(BUNDLE)
    dataset = pd.read_csv(ROOT / "data" / "behavioral_daily.csv", parse_dates=["date"])
    heldout = pd.read_csv(ROOT / "data" / "step4a_heldout_seed20261020" / "behavioral_daily.csv",
                          parse_dates=["date"])
    validation = dataset.loc[dataset["split"].eq("validation"), "date"]
    candidates = pd.concat([controlled_benchmark(dataset, bundle, w, "validation",
                                                 validation.min(), validation.max()) for w in (7, 14)])
    ranking = candidates.assign(correct=candidates["temporal_status"].eq(candidates["expected"]))
    scores = ranking.groupby("window")["correct"].mean().to_dict()
    selected = max(scores, key=lambda w: (scores[w], -w))
    test = heldout.loc[heldout["split"].eq("test"), "date"]
    controlled_test = controlled_benchmark(heldout, bundle, selected, "heldout_controlled", test.min(), test.max())
    natural = pd.DataFrame([
        benchmark(child.loc[child["date"].le(test.max())], bundle, selected, "heldout_cohort", "unmodified")
        for _, child in heldout.groupby("child_id")
    ])
    OUT.mkdir(exist_ok=True)
    all_results = pd.concat([candidates, controlled_test, natural], ignore_index=True)
    all_results.to_csv(OUT / "comparisons.csv", index=False)
    metrics = {
        "selected_window": selected, "validation": {w: summary(group) for w, group in candidates.groupby("window")},
        "heldout_controlled": summary(controlled_test), "heldout_cohort": summary(natural),
    }
    (OUT / "metrics.json").write_text(
        json.dumps(json_value(metrics), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_results(controlled_test, natural, selected)
    print(json.dumps({"selected_window": selected, "validation_accuracy": scores}))


if __name__ == "__main__":
    main()

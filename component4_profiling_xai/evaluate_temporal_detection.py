"""Chronological validation and frozen Step 05 synthetic holdout."""
import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import numpy as np
import pandas as pd

from .evaluate_anomaly_optimization import digest, serializable
from .evaluate_adaptive_profile import scenario_rows
from .src.behavioral_explanation_reliability import temporal_evidence, assess_temporal
from .src.behavioral_profile import BUNDLE
from .src.behavioral_synthetic import build_dataset
from .src.behavioral_temporal_challengers import Candidate, evaluate_episodes, paired_intervals, prepare, replay, resources
from .src.urgent_alerts import urgent_alerts
from .train_function02 import temporal_metrics

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "temporal_detection_optimization_step05"
DEV = ROOT / "data" / "behavioral_daily.csv"
EPISODES = ROOT / "data" / "behavioral_episodes.csv"
SEED = 2026112627
BUDGET = .02
SCENARIOS = ("stable", "gradual", "abrupt", "sustained", "bullying", "recurring",
             "isolated", "legitimate", "recovery", "incomplete")


def save(name, value):
    (OUT / name).write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def configurations():
    for rule, parameters, cutoffs in (("ewma", (.3, .6), (1.5, 2.5, 3.5)),
                                       ("cusum", (.5, .8), (4., 7., 10.))):
        for parameter, cutoff in product(parameters, cutoffs):
            yield Candidate(f"score_{rule}_{parameter}_{cutoff}", "score", rule, cutoff, parameter)
    for signal, cutoff in product(("score", "fixed", "guarded", "short_long", "combined"), (1.5, 2.5, 3.5)):
        for run in (2, 3):
            yield Candidate(f"{signal}_persistence_{cutoff}_{run}", signal, "persistence", cutoff, persistence=run)


def reference(history):
    for rule in ("ewma", "cusum"):
        yield "deployed_" + rule, history[[rule + "_alert", rule + "_active"]].set_axis(["alert", "active"], axis=1)
    yield "deployed_union", pd.DataFrame({
        "alert": history["ewma_alert"] | history["cusum_alert"],
        "active": history["ewma_active"] | history["cusum_active"],
    })
    flags = history["flag"].fillna(False).astype(bool)
    yield "daily_svm", pd.DataFrame({"alert": flags, "active": flags})


def normalized(daily):
    return daily.assign(date=pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d"))


def controlled(daily):
    """Construct independent streams from earlier normal donor records."""
    earliest = daily.loc[daily["split"].eq("train") & daily["is_anomaly"].eq(0)]
    anchors = earliest.sort_values(["child_id", "date"]).groupby("child_id").tail(1).head(4)
    rows, cases = [], []
    for offset, (_, base) in enumerate(anchors.iterrows()):
        for scenario in SCENARIOS:
            kind = "abrupt" if scenario == "legitimate" else "spike" if scenario == "isolated" else "stable"
            frame, _ = scenario_rows(base, kind, pd.Timestamp(daily["date"].max()), offset, 4)
            child = f"controlled-{offset}-{scenario}"
            frame["child_id"], frame["scenario"] = child, scenario
            frame["is_anomaly"], frame["episode_id"] = 0, ""
            frame["anomaly_type"] = "none"
            if scenario in {"gradual", "abrupt", "sustained", "bullying", "recurring", "recovery", "incomplete"}:
                end = 25 if scenario == "abrupt" else 35
                points = np.arange(17, end + 1)
                if scenario == "recurring":
                    points = points[np.arange(len(points)) % 5 <= 1]
                severity = (1 + (points - 17) // 4 if scenario == "gradual"
                            else 1 if scenario in {"bullying", "incomplete"} else 3)
                original_language = frame.loc[points, "language_flags"].copy()
                frame.loc[points, "language_flags"] += severity
                frame.loc[points, "language_bullying_flags"] = severity
                frame.loc[points, "language_max_risk"] = frame.loc[points, "language_max_risk"].clip(lower=.62)
                frame.loc[points, "language_mean_risk"] = (
                    frame.loc[points, "language_mean_risk"] * original_language + .62 * severity
                ) / frame.loc[points, "language_flags"]
                if scenario in {"sustained", "recovery"}:
                    frame.loc[points, "language_threat_flags"] = 1
                if scenario == "incomplete":
                    frame.loc[[17, 18, 19, 23, 24], "language_monitor_observed"] = False
                    # A recorded urgent signal remains independently visible even with incomplete monitoring.
                    frame.at[19, "language_urgent_flags"] = 1
                frame.loc[points, "is_anomaly"] = 1
                frame.loc[points, "episode_id"] = child
                frame.loc[points, "anomaly_type"] = scenario
                cases.append(dict(episode_id=child, child_id=child, anomaly_type=scenario,
                                  start_date=frame.at[17, "date"].strftime("%Y-%m-%d"),
                                  end_date=frame.at[end, "date"].strftime("%Y-%m-%d")))
            rows.append(frame)
    return normalized(pd.concat(rows, ignore_index=True)), pd.DataFrame(cases)


def validate():
    bundle = joblib.load(BUNDLE)
    original = normalized(pd.read_csv(DEV))
    cases = pd.read_csv(EPISODES)
    test_cases = cases.copy()
    test_cases["split"] = test_cases["split"].replace("val", "validation")
    _, deployed, _, _, _ = prepare(original, bundle)
    save("reference_audit.json", {
        method: temporal_metrics(deployed, original, test_cases, "test", method) for method in ("ewma", "cusum")
    })
    daily = original.loc[original["split"].ne("test")].reset_index(drop=True)
    valid = daily["split"].isin(["val", "validation"])
    if daily.loc[~valid, "date"].max() >= daily.loc[valid, "date"].min():
        raise ValueError("Validation overlaps training dates")
    episodes = test_cases.loc[test_cases["split"].eq("validation")]
    _, history, signals, _, _ = prepare(daily, bundle)
    eligible = history["score"].notna()
    baseline = [dict(name=name, **evaluate_episodes(daily, episodes, result, valid, eligible)[0])
                for name, result in reference(history)]
    results = []
    for candidate in configurations():
        stats, _, _ = evaluate_episodes(daily, episodes, replay(daily, signals[candidate.signal], candidate),
                                        valid, eligible)
        results.append({**asdict(candidate), **stats,
                        "signal_coverage": float(signals.loc[valid & eligible, candidate.signal].notna().mean())})
    ranked = pd.DataFrame(results).sort_values(["episode_recall", "episode_f1", "false_alert_rate", "name"],
                                              ascending=[False, False, True, True])
    affordable = ranked.loc[ranked["false_alert_rate"].le(BUDGET)]
    if affordable.empty:
        raise RuntimeError("No candidate satisfies the prespecified validation budget")
    best = affordable.iloc[0]
    config = next(asdict(c) for c in configurations() if c.name == best["name"])
    ranked.to_csv(OUT / "validation_search.csv", index=False)
    pd.DataFrame(baseline).to_csv(OUT / "validation_reference.csv", index=False)
    lock = {
        "candidate": config, "validation_summary": best.to_dict(), "seed": SEED, "alert_budget": BUDGET,
        "selection": "Highest validation episode recall within 2% normal-day onsets, then episode F1, FPR, name.",
        "matching": "One onset per episode on an assessable labeled injected day. No carry-in or date grace.",
        "recovery": "First assessable inactive normal date after end before next episode, among detected episodes.",
        "horizons": [1, 3, 7],
        "holdout": "Final 34 days of new draw plus separate controlled cases from earlier normal donors.",
        "deployment_gate": "Positive lower 95% paired child recall bound vs deployed union, FPR ≤ 2%, "
                           "no more than 0.75 points above union, with reviewed delay, reset and XAI.",
        "sha256": {path.name: digest(path) for path in (DEV, EPISODES, BUNDLE, Path(__file__),
                    ROOT / "src/behavioral_temporal_challengers.py")},
    }
    save("selection_lock.json", lock)
    print(config, flush=True)


def evaluate_cohort(daily, episodes, scope, bundle, candidate, label):
    matrix, history, signals, leaders, guarded = prepare(daily, bundle)
    eligible = history["score"].notna()
    challenger = replay(daily, signals[candidate.signal], candidate)
    summaries, case_rows, people, category = {}, [], [], []
    run_list = list(reference(history)) + [("candidate", challenger)]
    # Compare the unguarded and guarded references using the same fixed control recipe.
    for source in ("fixed", "guarded"):
        spec = Candidate(source + "_audit", source, "persistence", 2.5)
        run_list.append((spec.name, replay(daily, signals[source], spec)))
    for name, result in run_list:
        stats, cases, children = evaluate_episodes(daily, episodes, result, scope, eligible)
        summaries[name] = stats
        case_rows.append(cases.assign(arm=name))
        people.append(children.assign(arm=name))
        for kind, group in cases.groupby("anomaly_type"):
            category.append(dict(arm=name, category=kind, total=len(group), detected=int(group["delay"].count()),
                                 median_delay=group["delay"].median()))
    child = pd.concat(people)
    summaries["uncertainty"] = paired_intervals(child.loc[child["arm"].eq("candidate")],
                                               child.loc[child["arm"].eq("deployed_union")])
    summaries["coverage"] = {"assessable": int((scope & eligible).sum()), "rows": int(scope.sum()),
                            **{name: int((scope & signals[name].notna()).sum()) for name in signals}}
    # Compare predictor evidence on positive days without using labels in any replay.
    positive = scope & daily["is_anomaly"].eq(1)
    diagnostics = daily.loc[positive, ["episode_id", "child_id", "date", "anomaly_type"]].join(signals)
    diagnostics = diagnostics.join(history[["flag"]]).join(leaders.add_suffix("_leading_feature"))
    diagnostics["guarded_admission"] = guarded["decision"]
    diagnostics["candidate_alert"] = challenger["alert"]
    diagnostics.to_csv(OUT / f"{label}_positive_diagnostics.csv", index=False)
    pd.concat(case_rows).to_csv(OUT / f"{label}_episodes.csv", index=False)
    child.to_csv(OUT / f"{label}_per_child.csv", index=False)
    pd.DataFrame(category).to_csv(OUT / f"{label}_categories.csv", index=False)
    if label == "controlled":
        observed = daily.loc[scope, ["child_id", "date", "scenario", "is_anomaly"]].copy()
        observed["assessable"] = eligible
        observed["urgent_match"] = urgent_alerts(daily)["is_urgent"]
        observed["guard_decision"] = guarded["decision"]
        for name, result in run_list:
            observed[name + "_alert"] = result["alert"]
            observed[name + "_active"] = result["active"]
        observed.to_csv(OUT / "controlled_daily_trace.csv", index=False)
        summaries["urgent_matches_on_unassessable_days"] = int(
            (observed["urgent_match"] & ~observed["assessable"]).sum()
        )
        summaries["normal_scenarios"] = {
            name: observed.loc[observed["is_anomaly"].eq(0)].groupby("scenario")[name + "_alert"].sum().to_dict()
            for name, _ in run_list
        }
    return summaries, (matrix, history, signals, challenger)


def holdout():
    lock = json.loads((OUT / "selection_lock.json").read_text())
    for path in (DEV, EPISODES, BUNDLE, Path(__file__), ROOT / "src/behavioral_temporal_challengers.py"):
        if digest(path) != lock["sha256"][path.name]:
            raise RuntimeError(f"Lock mismatch for {path.name}")
    candidate = Candidate(**lock["candidate"])
    bundle = joblib.load(BUNDLE)
    generated = build_dataset(lock["seed"])
    daily = normalized(generated.daily)
    scope = daily["split"].ne("train")
    episodes = generated.episodes.loc[generated.episodes["split"].ne("train")]
    daily.to_csv(OUT / "fresh_synthetic.csv", index=False)
    controlled_daily, controls = controlled(daily)
    controlled_daily.to_csv(OUT / "controlled_inputs.csv", index=False)
    controls.to_csv(OUT / "controlled_labels.csv", index=False)
    normal, (matrix, history, signals, challenger) = evaluate_cohort(daily, episodes, scope, bundle, candidate, "generator")
    control_scope = pd.Series(controlled_daily.groupby("child_id").cumcount().ge(14), index=controlled_daily.index)
    stress, _ = evaluate_cohort(controlled_daily, controls, control_scope, bundle, candidate, "controlled")
    sample = daily.loc[scope & challenger["alert"]].groupby("child_id").head(1).head(8)
    checks = []
    for row in sample.itertuples():
        start = perf_counter()
        result, variants = temporal_evidence(history, matrix, bundle, row.child_id, row.date)
        review = assess_temporal(result, variants)
        legacy = history.loc[row.Index, ["ewma_alert", "cusum_alert"]].any()
        checks.append({"child": row.child_id, "date": row.date, "legacy_onset_overlap": bool(legacy),
                       "score_pattern": result.status, "score_review": review.status,
                       "score_group": review.group, "score_delta_error": result.reconstruction_error,
                       "score_explanation_ms": (perf_counter() - start) * 1000})
    probe = pd.DataFrame(checks)
    probe.to_csv(OUT / "temporal_xai_compatibility.csv", index=False)
    one_child = daily.loc[daily["child_id"].eq(daily["child_id"].iloc[0])]
    efficiencies = {"candidate_replay_90_rows": resources(lambda: replay(one_child, signals[candidate.signal], candidate)),
                    "reference_replay_90_rows": resources(lambda: replay(
                        one_child, signals["score"], Candidate("reference", "score", "ewma", 3., .3))),
                    "all_signal_preparation_90_rows": resources(lambda: prepare(one_child, bundle))}
    save("holdout_metrics.json", {"generator": normal, "controlled": stress, "resources": efficiencies,
                                 "fresh_sha256": digest(OUT / "fresh_synthetic.csv"),
                                 "xai": {"sample": len(probe), "supported": int(probe["score_review"].eq("supported").sum()),
                                         "median_ms": probe["score_explanation_ms"].median(),
                                         "max_reconstruction_error": probe["score_delta_error"].max()}
                                 if not probe.empty else {"sample": 0}})
    labels = ["deployed_ewma", "deployed_cusum", "deployed_union", "candidate"]
    data = pd.DataFrame({label: {"Generator": normal[label]["detected"], "Controlled": stress[label]["detected"]}
                         for label in labels}).T
    axis = data.plot.barh(figsize=(8, 4), color=["#2166ac", "#73a99e"])
    axis.set(xlabel="Matched episodes", title="Step 05 · episode detections by cohort")
    axis.figure.tight_layout()
    axis.figure.savefig(OUT / "episode_comparison.png", dpi=160)
    plt.close(axis.figure)
    print(json.dumps(serializable({"generator": {name: normal[name] for name in labels},
                                   "controlled": {name: stress[name] for name in labels}}), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "holdout"))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (validate if args.stage == "validate" else holdout)()


if __name__ == "__main__":
    main()

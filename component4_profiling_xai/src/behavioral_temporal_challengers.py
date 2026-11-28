"""Research-only temporal replay and exact-day episode evaluation."""
from dataclasses import dataclass
from time import perf_counter
import tracemalloc

import numpy as np
import pandas as pd

from .behavioral_adaptive_profile import adaptive_profiles
from .behavioral_profile import ANCHORS, ProfileConfig, TrendConfig, profiles, score_history, temporal

FEATURES = ("screen_incidents", "screen_night_incidents", "language_flags", "language_bullying_flags")


@dataclass(frozen=True)
class Candidate:
    name: str
    signal: str
    rule: str
    cutoff: float
    parameter: float = 0.3
    persistence: int = 2


def prepare(daily, bundle):
    matrix, details = profiles(daily, ProfileConfig(**bundle["profile"]))
    history = score_history(daily, matrix, bundle)
    guarded = adaptive_profiles(daily, matrix, history["flag"])
    valid = history["score"].notna()
    fixed_z = details["robust_z"][list(FEATURES)].where(valid)
    guarded_z = pd.DataFrame(guarded["robust_z"].tolist(), index=daily.index, columns=ANCHORS)
    guarded_z = guarded_z[list(FEATURES)].where(valid & guarded["available"])
    clipped = lambda values: values.clip(-10, 10)
    score = clipped((history["score"] - bundle["center"]) / bundle["scale"])
    short_long = pd.Series(np.nan, index=daily.index, dtype=float)
    for _, child in daily.groupby("child_id"):
        ordered = child.sort_values("date")
        days = pd.DatetimeIndex(pd.to_datetime(ordered["date"]))
        short = matrix.loc[ordered.index, list(FEATURES)].where(valid).set_axis(days)
        short = short.rolling("3D", min_periods=3).mean().set_axis(ordered.index)
        long = details.loc[ordered.index, "median"][list(FEATURES)]
        scale = details.loc[ordered.index, "mad"][list(FEATURES)].mul(1.4826).clip(lower=1)
        short_long.loc[ordered.index] = ((short - long) / scale).max(axis=1)
    signals = pd.DataFrame({
        "score": score, "fixed": clipped(fixed_z.max(axis=1)), "guarded": clipped(guarded_z.max(axis=1)),
        "short_long": clipped(short_long.where(valid)),
    })
    signals["combined"] = pd.concat([score, signals["guarded"] - 1], axis=1).max(axis=1).where(valid)
    leaders = pd.DataFrame({"fixed": fixed_z.fillna(-np.inf).idxmax(axis=1).where(fixed_z.notna().any(axis=1)),
                            "guarded": guarded_z.fillna(-np.inf).idxmax(axis=1).where(guarded_z.notna().any(axis=1))})
    return matrix, history, signals, leaders, guarded


def replay(daily, values, candidate):
    blocks = []
    for _, child in daily.groupby("child_id", sort=False):
        index = child.sort_values("date").index
        dates = pd.DatetimeIndex(pd.to_datetime(daily.loc[index, "date"]))
        if dates.has_duplicates:
            raise ValueError("Expected one record per child and date")
        signal = values.loc[index]
        if candidate.rule in {"ewma", "cusum"}:
            config = TrendConfig(threshold=candidate.cutoff, cusum_threshold=candidate.cutoff,
                                 alpha=candidate.parameter if candidate.rule == "ewma" else .3,
                                 drift=candidate.parameter if candidate.rule == "cusum" else .5,
                                 min_run=candidate.persistence)
            result = temporal(dates, signal.to_numpy(), np.zeros(len(index), dtype=bool), 0., 1., config)
            active = result[candidate.rule + "_active"].to_numpy()
            alert = result[candidate.rule + "_alert"].to_numpy()
        else:
            # A missing record or calendar gap starts a new evidence window.
            finite = signal.notna().to_numpy()
            breaks = ~finite | np.asarray(pd.Series(dates).diff().dt.days.ne(1))
            breaks[1:] |= ~finite[:-1]
            runs = np.cumsum(breaks)
            high = signal.ge(candidate.cutoff).astype(int).groupby(runs).rolling(5, min_periods=1).sum()
            active = high.to_numpy() >= candidate.persistence
            active &= finite
            alert = active & (breaks | ~np.r_[False, active[:-1]])
        blocks.append(pd.DataFrame({"alert": alert, "active": active}, index=index))
    return pd.concat(blocks).reindex(daily.index)


def evaluate_episodes(daily, episodes, result, scope, assessable):
    """Credit one onset per labeled observed episode day. No carry-in or grace."""
    selected = daily.loc[scope].assign(alert=result["alert"], active=result["active"], assessable=assessable)
    onsets = selected.loc[selected["alert"] & selected["assessable"]]
    first = onsets.loc[onsets["is_anomaly"].eq(1)].groupby("episode_id")["date"].min()
    cases = episodes[["episode_id", "child_id", "anomaly_type", "start_date", "end_date"]].copy()
    cases["observed_labels"] = cases["episode_id"].map(
        selected.loc[selected["is_anomaly"].eq(1)].groupby("episode_id").size()).fillna(0).astype(int)
    cases["assessable_labels"] = cases["episode_id"].map(
        selected.loc[selected["is_anomaly"].eq(1) & selected["assessable"]].groupby("episode_id").size()
    ).fillna(0).astype(int)
    cases["first_alert"] = cases["episode_id"].map(first)
    cases["delay"] = (pd.to_datetime(cases["first_alert"]) - pd.to_datetime(cases["start_date"])).dt.days
    cases["recovery_delay"] = np.nan
    for case in cases.loc[cases["delay"].notna()].itertuples():
        later = selected.loc[selected["child_id"].eq(case.child_id) & selected["date"].gt(case.end_date)]
        eligible = later.loc[later["assessable"] & ~later["active"] & later["is_anomaly"].eq(0)]
        next_start = cases.loc[cases["child_id"].eq(case.child_id) & cases["start_date"].gt(case.end_date),
                               "start_date"].min()
        if pd.notna(next_start):
            eligible = eligible.loc[eligible["date"].lt(next_start)]
        if not eligible.empty:
            cases.loc[case.Index, "recovery_delay"] = (pd.Timestamp(eligible["date"].min())
                                                      - pd.Timestamp(case.end_date)).days
    normal = selected.loc[selected["assessable"] & selected["is_anomaly"].eq(0)]
    children = selected[["child_id"]].drop_duplicates().set_index("child_id")
    for key, counts in {
        "episodes": cases.groupby("child_id").size(),
        "hits": cases.loc[cases["delay"].notna()].groupby("child_id").size(),
        "normal_days": normal.groupby("child_id").size(),
        "false_alerts": normal.loc[normal["alert"]].groupby("child_id").size(),
        "alert_onsets": onsets.groupby("child_id").size(),
    }.items():
        children[key] = counts.reindex(children.index, fill_value=0)
    return summarize(cases, children), cases, children.reset_index()


def summarize(cases, children):
    n = len(cases)
    delay = cases["delay"].dropna()
    hits, alerts = len(delay), int(children["alert_onsets"].sum())
    false = int(children["false_alerts"].sum())
    normal = int(children["normal_days"].sum())
    return dict(episodes=n, detected=hits, missed=n-hits, episode_precision=hits/alerts if alerts else None,
                episode_recall=hits/n if n else None, episode_f1=2*hits/(n+alerts) if n+alerts else None,
                normal_days=normal, false_alerts=false, false_alert_rate=false/normal if normal else None,
                false_alerts_per_child=false/len(children) if len(children) else None, alert_onsets=alerts,
                repeat_on_episode_alerts=alerts-false-hits, mean_delay=delay.mean(), median_delay=delay.median(),
                within_1_day=int(delay.le(1).sum()), within_3_days=int(delay.le(3).sum()),
                within_7_days=int(delay.le(7).sum()), recovered=int(cases["recovery_delay"].count()),
                median_recovery=cases["recovery_delay"].median(),
                episodes_without_assessable_labels=int(cases["assessable_labels"].eq(0).sum()))


def paired_intervals(candidate, reference):
    merged = candidate.set_index("child_id").join(reference.set_index("child_id"), rsuffix="_ref")
    fields = ["hits", "episodes", "false_alerts", "normal_days"]
    values = merged[fields + [key + "_ref" for key in fields]].to_numpy()
    rng = np.random.default_rng(805)
    draws = values[rng.integers(len(values), size=(2000, len(values)))].sum(axis=1)
    recall = draws[:, 0]/draws[:, 1] - draws[:, 4]/draws[:, 5]
    fpr = draws[:, 2]/draws[:, 3] - draws[:, 6]/draws[:, 7]
    return {"recall_delta_95": np.quantile(recall, [.025, .975]).tolist(),
            "false_alert_rate_delta_95": np.quantile(fpr, [.025, .975]).tolist()}


def resources(operation):
    times = []
    for _ in range(3):
        start = perf_counter()
        operation()
        times.append((perf_counter() - start) * 1000)
    tracemalloc.start()
    operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"median_ms": float(np.median(times)), "peak_traced_bytes": peak}

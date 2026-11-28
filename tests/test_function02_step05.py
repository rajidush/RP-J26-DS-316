"""Research detector causality, coverage, matching and compatibility."""
import socket

import joblib
import numpy as np
import pandas as pd
import pytest

from component4_profiling_xai.evaluate_temporal_detection import DEV, controlled, normalized
from component4_profiling_xai.src.behavioral_explanation_reliability import assess_temporal, temporal_evidence
from component4_profiling_xai.src.behavioral_profile import BUNDLE
from component4_profiling_xai.src.behavioral_temporal_challengers import Candidate, evaluate_episodes, prepare, replay
from component4_profiling_xai.src.urgent_alerts import urgent_alerts


def calendar(n, child="trial"):
    return pd.DataFrame({"date": pd.date_range("2026-02-01", periods=n).strftime("%Y-%m-%d"), "child_id": child})


@pytest.mark.parametrize("rule,cutoff,parameter", [("persistence", 2., .3), ("ewma", 1., .6), ("cusum", 3.2, .5)])
def test_gradual_sustained_changes_and_single_spikes(rule, cutoff, parameter):
    daily = calendar(15)
    values = pd.Series([0, 0, 4, 0, 0, 0, 1, 2, 3, 4, 4, 4, 0, 0, 0])
    spec = Candidate("experiment", "score", rule, cutoff, parameter)
    result = replay(daily, values, spec)
    assert not result.loc[:6, "alert"].any()
    assert result.loc[7:11, "alert"].any()


def test_missing_reset_recovery_and_child_isolation():
    daily = pd.concat([calendar(14), calendar(14, "other")], ignore_index=True)
    signal = pd.Series([3, 3, 3, np.nan, 3, 0, 0, 0, 0, 0, 3, 3, 0, 0] + [0] * 14)
    rule = Candidate("experiment", "score", "persistence", 2)
    result = replay(daily, signal, rule)
    assert result.loc[1, "alert"] and result.loc[11, "alert"]
    assert not result.loc[3:10, "active"].any()
    assert not result.loc[14:, "alert"].any()
    # Dropping a row still breaks the calendar window.
    absent = replay(daily.drop(index=3), signal, rule)
    assert not absent.loc[4, "active"]
    prefix = daily["date"].le(daily.at[7, "date"])
    pd.testing.assert_frame_equal(result.loc[prefix], replay(daily.loc[prefix], signal, rule))


def test_exact_episode_onsets_no_grace_and_recovery_censoring():
    daily = calendar(9).assign(is_anomaly=[0, 0, 1, 1, 1, 0, 0, 1, 0],
                               episode_id=["", "", "a", "a", "a", "", "", "b", ""])
    episodes = pd.DataFrame({"episode_id": ["a", "b"], "child_id": ["trial"] * 2,
                             "anomaly_type": ["kind"] * 2, "start_date": daily.loc[[2, 7], "date"].tolist(),
                             "end_date": daily.loc[[4, 7], "date"].tolist()})
    alerts = pd.DataFrame({"alert": [False, True, True, True, True, False, False, False, False],
                           "active": [False, True, False, True, True, True, False, True, False]})
    eligible = pd.Series(True, index=daily.index)
    eligible.at[2] = False
    stats, cases, _ = evaluate_episodes(daily, episodes, alerts, pd.Series(True, index=daily.index), eligible)
    assert stats["detected"] == 1 and stats["missed"] == 1
    assert stats["false_alerts"] == 1 and stats["repeat_on_episode_alerts"] == 1
    assert stats["episode_precision"] == pytest.approx(1/3)
    assert cases.loc[0, "delay"] == 1 and cases.loc[0, "recovery_delay"] == 2
    assert pd.isna(cases.loc[1, "delay"])


@pytest.fixture(scope="module")
def control():
    daily, _ = controlled(normalized(pd.read_csv(DEV)))
    selected = daily["child_id"].isin(["controlled-0-incomplete", "controlled-0-legitimate",
                                      "controlled-0-recovery"])
    return daily.loc[selected].reset_index(drop=True)


def test_guarded_evidence_and_independent_urgent_rules_offline(control, monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Research evaluation attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    model = joblib.load(BUNDLE)
    before = urgent_alerts(control)
    matrix, history, signals, _, guard = prepare(control, model)
    missing = ~control["language_monitor_observed"]
    assert signals.loc[missing].isna().all(axis=None)
    assert (missing & before["is_urgent"]).any()
    pd.testing.assert_frame_equal(before, urgent_alerts(control))
    harmful = control["scenario"].eq("recovery") & control["is_anomaly"].eq(1)
    assert guard.loc[harmful, "decision"].eq("concern").all()
    assert not any(guard.loc[harmful, "admitted_indices"])
    assert guard.loc[control["scenario"].eq("legitimate"), "shift"].any()

    # Replaying another outcome never replaces the deployed score explanation state.
    child = "controlled-0-legitimate"
    day = control.loc[control["child_id"].eq(child), "date"].max()
    original, variants = temporal_evidence(history, matrix, model, child, day)
    replay(control, signals["combined"], Candidate("trial", "combined", "persistence", 2.5))
    again, again_variants = temporal_evidence(history, matrix, model, child, day)
    assert (original, variants) == (again, again_variants)
    assert assess_temporal(original, variants).status != "invalid"


def test_all_feature_and_score_inputs_are_prior_causal_and_child_specific(control):
    mdl = joblib.load(BUNDLE)
    _, _, original, _, _ = prepare(control, mdl)
    selected = control["child_id"].eq("controlled-0-recovery")
    cutoff = control.loc[selected, "date"].iloc[28]
    scoped = selected & control["date"].le(cutoff)
    changed = control.assign(is_anomaly=0, anomaly_type="audit", episode_id="ignored")
    changed.loc[~scoped, "language_flags"] += 200
    _, _, actual, _, _ = prepare(changed, mdl)
    pd.testing.assert_frame_equal(original.loc[scoped], actual.loc[scoped])

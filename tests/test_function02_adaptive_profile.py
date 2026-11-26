"""Adaptive profile admission and as-of replay against the deployed data shape."""
import joblib
import pandas as pd
import pytest

from component4_profiling_xai.evaluate_adaptive_profile import ONSET, RECOVERY, scenario_rows
from component4_profiling_xai.src.behavioral_adaptive_profile import AdaptiveConfig, adaptive_profiles
from component4_profiling_xai.src.behavioral_profile import BUNDLE, ProfileConfig, profiles, score_history
from ui import services
from ui.pages._behavioral_upgrade import adaptive_summary, adaptive_update_caption, historical_comparison


@pytest.fixture(scope="module")
def anchor():
    return services.behavioral_daily_source()["daily"].iloc[0]


def replay(anchor, scenario, level=0, flag=None):
    data, labels = scenario_rows(anchor, scenario, pd.Timestamp("2026-09-28"), level, 5)
    matrix, _ = profiles(data)
    flags = pd.Series(False, index=data.index, dtype="boolean")
    if flag:
        flags.loc[list(flag)] = True
    result = adaptive_profiles(data, matrix, flags)
    return data, matrix, flags, result, labels


@pytest.mark.parametrize("scenario", ["gradual", "abrupt"])
def test_legitimate_change_adapts(anchor, scenario):
    _, _, _, result, _ = replay(anchor, scenario)
    assert result.iloc[-1]["median"][0] == 5
    assert result["shift"].any()
    assert result.loc[ONSET, "median"][0] == 0
    if scenario == "abrupt":
        assert result.loc[ONSET, "decision"] == "probation"


def test_isolated_spike_is_not_admitted(anchor):
    _, _, _, result, _ = replay(anchor, "spike")
    assert result.loc[ONSET, "decision"] == "probation"
    assert ONSET not in {i for items in result["admitted_indices"] for i in items}
    assert result.iloc[-1]["median"][0] == 0 and not result["shift"].any()


@pytest.mark.parametrize("scenario", ["screen_concern", "language_concern", "repeated_anomaly"])
def test_persistent_concern_and_incoherent_anomalies_are_excluded(anchor, scenario):
    _, _, _, result, affected = replay(anchor, scenario, flag=range(ONSET, RECOVERY))
    assert not affected & {i for items in result["admitted_indices"] for i in items}
    if scenario != "repeated_anomaly":
        assert result.loc[list(affected), "frozen_after"].all()
        assert adaptive_summary(result.loc[ONSET]).startswith("More ")
        assert "set aside" in adaptive_update_caption(result.loc[ONSET])
        assert "reliable comparison" in adaptive_summary(result.iloc[-1])
        assert "set aside" in adaptive_update_caption(result.iloc[-1])


def test_urgent_freeze_recovers_without_admitting_concerning_days(anchor):
    _, _, _, result, affected = replay(anchor, "recovery")
    assert result.loc[ONSET, "decision"] == "urgent"
    assert result.loc[RECOVERY, "decision"] == "recovery_wait"
    assert result.loc[RECOVERY + AdaptiveConfig().recovery - 1, "decision"] == "recovered"
    assert not affected & {i for items in result["admitted_indices"] for i in items}
    assert result.iloc[-1]["median"][0] == 0


def test_false_positive_flags_can_be_followed_by_a_confirmed_update(anchor):
    start = range(ONSET, ONSET + AdaptiveConfig().confirmation)
    _, _, _, result, _ = replay(anchor, "false_positive", flag=start)
    assert result.loc[ONSET, "decision"] == "probation"
    assert result.loc[max(start), "admitted_indices"] == tuple(start)
    assert "similar to" in adaptive_summary(result.loc[ONSET])
    assert "similar pattern on later days" in adaptive_update_caption(result.loc[ONSET])
    admitted = result.loc[max(start)]
    assert adaptive_update_caption(admitted) == "Recent suitable records were included in the usual activity profile."
    assert adaptive_update_caption({**admitted.to_dict(), "admitted_indices": ()}) is None
    assert adaptive_update_caption(result.loc[ONSET - 1]) is None
    assert not result["shift"].any()


def test_missing_days_are_excluded_but_observed_zeros_and_cold_start_are_retained(anchor):
    _, _, _, result, _ = replay(anchor, "missing")
    assert result.loc[0, "admitted_indices"] == (0,)
    assert not set(range(ONSET, ONSET + 9)) & set(result.index[result["available"]])
    assert result.iloc[-1]["median"][0] == 0
    _, _, _, cold, _ = replay(anchor, "cold_start")
    assert all(pd.isna(value) for value in cold.iloc[-1]["median"])
    assert "too few usable earlier records" in adaptive_summary(cold.iloc[-1])
    assert adaptive_update_caption(cold.iloc[-1]) is None


def test_child_isolation_determinism_and_no_future_leakage(anchor):
    a, _, _, expected, _ = replay(anchor, "abrupt", 1)
    b, _, _, _, _ = replay(anchor, "stable", 9)
    b = b.assign(child_id="other")
    pooled = pd.concat([a, b], ignore_index=True)
    matrix, _ = profiles(pooled)
    flags = pd.Series(False, index=pooled.index)
    actual = adaptive_profiles(pooled, matrix, flags)
    pd.testing.assert_frame_equal(expected, actual.loc[a.index])
    assert actual.iloc[-1]["median"][0] == 9
    eligible = pooled["date"].le(a.loc[23, "date"])
    cutoff = adaptive_profiles(pooled.loc[eligible], matrix.loc[eligible], flags)
    pd.testing.assert_frame_equal(actual.loc[eligible], cutoff)
    pd.testing.assert_frame_equal(actual, adaptive_profiles(pooled.sample(frac=1, random_state=23),
                                                           matrix, flags).reindex(actual.index))


def test_deployed_scores_and_urgent_rules_are_independent(anchor):
    data, _, _, _, _ = replay(anchor, "recovery")
    matrix, detail = profiles(data)
    bundle = joblib.load(BUNDLE)
    before = score_history(data, matrix, bundle)
    urgent_before = services.behavioral_urgent_report(data, data["child_id"].iloc[0])["urgent"]
    wording, caption, ready = historical_comparison(data, matrix, detail, before["flag"], ONSET,
                                                    data.loc[ONSET, "date"], ProfileConfig())
    assert ready
    assert wording.startswith("More ")
    assert "set aside" in caption
    pd.testing.assert_frame_equal(before, score_history(data, matrix, bundle))
    pd.testing.assert_frame_equal(urgent_before,
                                  services.behavioral_urgent_report(data, data["child_id"].iloc[0])["urgent"])

"""Unit and offline Streamlit integration tests for the Function 02 upgrade."""
from datetime import timedelta
from io import BytesIO
import socket

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from component4_profiling_xai.src.behavioral_anomaly import anomaly_scores
from component4_profiling_xai.src.behavioral_explain import explain
from component4_profiling_xai.src.behavioral_profile import (
    BUNDLE, ProfileConfig, TrendConfig, priority, profiles, score_history, temporal,
)
from ui import services


@pytest.fixture
def sample():
    source = services.behavioral_daily_source()
    return source["daily"].query("child_id in ['C001', 'C009']").reset_index(drop=True)


def test_calendar_window_missing_zero_and_child_isolation(sample):
    child = sample[sample.child_id.eq("C001")].head(8).copy()
    anchor = "screen_incidents"
    child[anchor] = [0, 0, 0, 0, 5000, 4, 12, 9000]
    child.loc[child.index[3], "screen_monitor_observed"] = 0
    day = child.index[5]
    cfg = ProfileConfig(minimum=3)
    features, detail = profiles(child, cfg)
    assert detail.loc[day, ("n", anchor)] == 4
    assert detail.loc[day, ("median", anchor)] == detail.loc[day, ("mad", anchor)] == 0
    assert features.loc[day, f"{anchor}__robust_z"] == 4
    assert features.loc[day, f"{anchor}__active_days"] == 1
    assert np.isnan(features.loc[child.index[3], anchor])
    assert np.isnan(features.iloc[0][f"{anchor}__robust_z"])
    combined = pd.concat([child, sample[sample.child_id.eq("C009")]])
    pd.testing.assert_series_equal(profiles(combined, cfg)[0].loc[day],
                                   profiles(child.loc[:day], cfg)[0].loc[day])
    child.loc[child.index[-1], "date"] += timedelta(days=30)
    assert profiles(child, cfg)[1].iloc[-1][("n", anchor)] == 0


def test_trends_single_repeat_sustained_and_missing():
    dates = pd.date_range("2026-01-01", periods=12)
    scores = [0, 3, 0, 0, 3, 3, 3, 3, np.nan, 3, 3, 0]
    flags = [s > 1 if np.isfinite(s) else pd.NA for s in scores]
    trend = temporal(dates, scores, flags, 0, 1, TrendConfig(threshold=1, alpha=0.6))
    assert not trend.loc[1, "ewma_alert"]
    assert trend.loc[5, "ewma_alert"] and trend.loc[5, "prior_streak"] == 1
    assert trend.loc[6, "ewma_active"] and not trend.loc[6, "ewma_alert"]
    assert trend.loc[9, "prior_streak"] == 0
    assert trend.loc[10, "ewma_alert"]
    assert trend.loc[5, "prior_flagged"] == 2
    assert priority(True, 0, False, 0, False)[0] == "Urgent safety review"
    assert priority(False, 0, False, 0, False)[0] == "No unusual activity detected"
    assert priority(False, 3, True, 0, False)[0] == "Needs attention"
    assert priority(False, 3, True, 2, False)[0] == "Priority review"
    assert priority(False, np.nan, False, 0, False)[0] == "Insufficient data"


def test_group_explanations_match_actual_scoring(sample):
    model = joblib.load(BUNDLE)
    matrix, _ = profiles(sample)
    history = score_history(sample, matrix, model)
    i = history.index[history["flag"].fillna(False)][0]
    row = matrix.loc[[i], model["features"]]
    result = explain(model["pipeline"], row, model["background"])
    assert result["score"] == pytest.approx(anomaly_scores(model["pipeline"], row)[0])
    assert result["score"] == pytest.approx(result["reference"] + sum(result["shap"].values()))
    assert np.isfinite(result["lime_r2"])
    assert {"Screen activity", "Language activity"} <= result["shap"].keys()
    assert result["lime"].keys() == result["shap"].keys()


def test_offline_parent_page_child_date_and_urgent_override(monkeypatch, tmp_path):
    from component4_profiling_xai.src.behavioral_explanation_reliability import assess_temporal, temporal_evidence
    from ui.pages._behavioral_upgrade import recent_behavior, temporal_text

    def blocked(*args, **kwargs):
        raise AssertionError("Outbound connection attempted during local inference")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.delenv("C4_F2_LEGACY_VIEW", raising=False)
    monkeypatch.delenv("C4_F2_ARTIFACT_DIR", raising=False)
    source = services.behavioral_daily_source()
    daily = source["daily"].copy()
    model = joblib.load(BUNDLE)
    child = daily[daily.child_id.eq("C001")]
    matrix, _ = profiles(child)
    scored = score_history(child, matrix, model)
    index = scored.index[scored["flag"].eq(False).fillna(False)][-2]
    urgent_date = daily.loc[index, "date"]
    # A curated safety signal outside the detector's input features.
    daily.loc[index, "screen_self_harm_imagery_incidents"] = 1
    missing_index = child.index[0]
    daily.loc[missing_index, "screen_monitor_observed"] = 0
    missing_date = daily.at[missing_index, "date"]
    monkeypatch.setattr(services, "behavioral_daily_source", lambda: {**source, "daily": daily})
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'offline.db').as_posix()}")
    page = AppTest.from_file(str(services.ROOT / "ui/app.py"), default_timeout=45).run()
    assert not page.exception
    navigation = " ".join(str(link.proto) for link in page.get("page_link"))
    assert 'label: "Emotional Pattern"' in navigation
    assert 'label: "Behavioral Anomaly"' in navigation
    assert "Research" not in navigation and "Child Profile" not in navigation
    page.switch_page("pages/behavioral_anomaly.py").run()
    assert not page.exception
    date_input = page.selectbox(key="f2_activity_C001")
    date_input.set_value(urgent_date).run()
    assert not page.exception
    assert any("Urgent safety review" in text.value and "Images that may show self-harm" in text.value
               for text in page.warning)
    assert any("No unusual pattern was detected on this day" in item.value for item in page.markdown)
    assert not page.metric and not page.dataframe and not page.get("plotly_chart")
    flagged_date = child.loc[scored.index[scored["flag"].eq(True).fillna(False)][-1], "date"]
    page.selectbox(key="f2_activity_C001").set_value(flagged_date).run()
    full_text = " ".join(
        [item.value for item in page.markdown] + [item.value for item in page.caption]
    )
    assert "made this day look more unusual" in full_text
    assert "compares the recorded activity" in full_text
    assert "An unusual activity pattern was detected on this day" in full_text
    recent = recent_behavior(scored, flagged_date, model["profile"]["minimum"])
    temporal, probes = temporal_evidence(scored, matrix, model, "C001", flagged_date)
    expected = temporal_text(temporal, recent[1], assess_temporal(temporal, probes))
    assert any(item.value == expected for item in page.caption)
    assert "No detected anomaly does not guarantee safety." in full_text
    for research_metric in ("F1-score", "Precision", "Recall", "confusion matrix", "weighted R²"):
        assert research_metric not in full_text
    page.selectbox(key="f2_activity_C001").set_value(missing_date).run()
    assert any("Not enough information to assess activity" in item.value for item in page.markdown)
    assert any("Some activity records are missing" in item.value for item in page.markdown)
    assert any("too few usable records to explain changes" in item.value for item in page.caption)
    page.selectbox(key="f2_child").set_value("C009").run()
    assert not page.exception
    assert page.selectbox(key="f2_activity_C009").value == daily.loc[daily.child_id.eq("C009"), "date"].max()
    assert page.selectbox(key="f2_activity_C009").value != urgent_date
    other_child = daily[daily.child_id.eq("C009")]
    other_date = page.selectbox(key="f2_activity_C009").value
    other_matrix, _ = profiles(other_child)
    other_scores = score_history(other_child, other_matrix, model)
    recent = recent_behavior(other_scores, other_date, model["profile"]["minimum"])
    temporal, probes = temporal_evidence(other_scores, other_matrix, model, "C009", other_date)
    expected = temporal_text(temporal, recent[1], assess_temporal(temporal, probes))
    assert any(item.value == expected for item in page.caption)

def test_local_research_interface():
    page = AppTest.from_file(str(services.ROOT / "component4_profiling_xai/experiments/archived_research_function02.py"), default_timeout=45).run()
    assert not page.exception
    assert len(page.header) == 6
    assert any("Synthetic benchmark" in message.value for message in page.warning)

def test_qualitative_temporal_summary_distinguishes_c014_carryover_and_active_change():
    from ui.pages._behavioral_upgrade import recent_behavior

    source = services.behavioral_daily_source()
    rows = services._behavioral_child_rows(source["daily"], "C014")
    matrix, _ = profiles(rows)
    history = score_history(rows, matrix, joblib.load(BUNDLE))
    latest = history["date"].iloc[-1]
    assert history["cusum_active"].iloc[-1]
    assert not history.tail(30)["flag"].any()
    summary, active, earlier = recent_behavior(history, latest, 5)
    assert not active and earlier < latest - timedelta(days=29)
    assert "No unusual patterns" in summary

    onset = history.loc[history["cusum_alert"], "date"].iloc[0]
    summary, active, earlier = recent_behavior(history, onset, 5)
    assert active and earlier == onset
    assert "continued across recent days" in summary


def test_qualitative_explanations_distinguish_counts_from_attributions():
    from component4_profiling_xai.src.behavioral_profile import ANCHORS
    from ui.pages._behavioral_upgrade import explanation_text, guidance, observed_summary

    row = pd.Series(0.0, index=ANCHORS)
    result = {"shap": {"Screen activity": 0.8, "Language activity": -0.1},
              "lime": {"Screen activity": 0.2, "Language activity": -0.2}}
    message = explanation_text(result, flagged=True)
    assert "screen activity records" in message and "second explanation" in message
    assert "incidents" not in message and "0.8" not in message
    assert "less unusual" in explanation_text(result, flagged=False)
    assert "online evenings" not in guidance(row, result, False)[0]
    assert "similar" in observed_summary(row, {"median": row.copy()})
    assert "enough earlier activity" in observed_summary(row, {"median": row * np.nan})
    row["screen_night_incidents"] = 1
    assert observed_summary(row, {"median": row * 0}) == "More nighttime screen incidents were recorded than usual."
    assert "online evenings" in guidance(row, result, False)[0]

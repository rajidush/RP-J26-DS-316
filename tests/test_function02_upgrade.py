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
from parent_dashboard_test_helpers import follow_page_link
from ui import child_selection, services


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
    from ui.legacy_pages._behavioral_upgrade import recent_behavior, temporal_text

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
    page = AppTest.from_file(str(services.ROOT / "ui/app.py"), default_timeout=45)
    page.session_state[child_selection.DEMO_WIDGET] = True
    page.run()
    assert not page.exception
    navigation = " ".join(str(link.proto) for link in page.get("page_link"))
    assert 'label: "Emotional Pattern"' in navigation
    assert 'label: "Behavioral Anomaly"' in navigation
    assert "Research" not in navigation and 'label: "Child Profile"' not in navigation
    follow_page_link(page, page.sidebar, "Behavioral Anomaly")
    assert not page.exception
    date_input = page.selectbox(key="f2_activity_C001")
    date_input.set_value(urgent_date).run()
    assert not page.exception
    assert any("Urgent safety review" in text.value and "Images that may show self-harm" in text.value
               for text in page.warning)
    assert any("The AI detected no unusual pattern" in item.value for item in page.markdown)
    exclusion_text = "were set aside when updating the usual pattern"
    assert any(exclusion_text in item.value for item in page.caption)
    assert all(exclusion_text not in item.value for item in page.markdown)
    assert any(item.value.startswith(("This day’s recorded activity", "Your child’s usual recorded pattern", "More ", "Fewer ",
                                      "There are too few usable earlier records"))
               for item in page.markdown)
    assert not page.metric and not page.dataframe and not page.get("plotly_chart")
    flagged_date = child.loc[scored.index[scored["flag"].eq(True).fillna(False)][-1], "date"]
    page.selectbox(key="f2_activity_C001").set_value(flagged_date).run()
    full_text = " ".join(
        [item.value for item in page.markdown] + [item.value for item in page.caption]
    )
    assert "made the recorded activity look more unusual" in full_text
    assert "uses your child’s own earlier records" in full_text
    assert "The AI detected an unusual pattern in this day’s records" in full_text
    recent = recent_behavior(scored, flagged_date, model["profile"]["minimum"])
    temporal, probes = temporal_evidence(scored, matrix, model, "C001", flagged_date)
    expected = temporal_text(temporal, recent[1], assess_temporal(temporal, probes))
    assert any(item.value == expected for item in page.caption)
    assert "cannot tell you whether your child is safe." in full_text
    for research_metric in ("F1-score", "Precision", "Recall", "confusion matrix", "weighted R²"):
        assert research_metric not in full_text
    page.selectbox(key="f2_activity_C001").set_value(missing_date).run()
    assert any("couldn’t assess this day" in item.value for item in page.markdown)
    assert any("This day is missing records needed for an AI check" in item.value for item in page.markdown)
    assert any("too little activity recorded for this day" in item.value
               for item in page.markdown)
    assert any("too few usable records to explain the AI’s assessments" in item.value for item in page.caption)
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
    from ui.legacy_pages._behavioral_upgrade import recent_behavior

    source = services.behavioral_daily_source()
    rows = services._behavioral_child_rows(source["daily"], "C014")
    matrix, _ = profiles(rows)
    history = score_history(rows, matrix, joblib.load(BUNDLE))
    latest = history["date"].iloc[-1]
    assert history["cusum_active"].iloc[-1]
    assert not history.tail(30)["flag"].any()
    summary, active, earlier = recent_behavior(history, latest, 5)
    assert not active and earlier < latest - timedelta(days=29)
    assert "No continuing change" in summary

    onset = history.loc[history["cusum_alert"], "date"].iloc[0]
    summary, active, earlier = recent_behavior(history, onset, 5)
    assert active and earlier == onset
    assert "continued across recent days" in summary


def test_qualitative_explanations_distinguish_counts_from_attributions():
    from component4_profiling_xai.src.behavioral_explanation_reliability import Review
    from component4_profiling_xai.src.behavioral_profile import ANCHORS
    from ui.legacy_pages._behavioral_upgrade import explanation_text, guidance, observed_summary

    row = pd.Series(0.0, index=ANCHORS)
    result = {"shap": {"Screen activity": 0.8, "Language activity": -0.1},
              "lime": {"Screen activity": 0.2, "Language activity": -0.2}}
    message = explanation_text(result, True, Review("supported", "Screen activity", True))
    assert "screen activity records" in message and "Both explanation checks" in message
    assert "incidents" not in message and "0.8" not in message
    assert "less unusual" in explanation_text(result, False, Review("supported", "Language activity"))
    supported = Review("supported", "Screen activity")
    assert "online evenings" not in guidance(row, supported, False)[0]
    assert "similar" in observed_summary(row, {"median": row.copy()})
    assert "enough earlier activity" in observed_summary(row, {"median": row * np.nan})
    row["screen_night_incidents"] = 1
    assert observed_summary(row, {"median": row * 0}) == "More nighttime screen incidents were recorded than usual."
    assert "online evenings" in guidance(row, supported, False)[0]
    assert "AI explanation" in guidance(row, supported, False)[0]
    assert "Pick a calm moment" in guidance(row, Review("limited", "Screen activity"), False)[0]
    row["language_bullying_flags"] = 1
    assert "records include language flagged" in guidance(row, Review("invalid"), False)[0]
    assert "AI explanation" not in guidance(row, Review("invalid"), False)[0]
    advice, question = guidance(row, Review("invalid"), False, urgent=True)
    assert "seek appropriate human support" in advice and question is None


def test_dashboard_distinguishes_safety_coverage_and_adaptive_history(monkeypatch):
    from component4_profiling_xai.src.urgent_alerts import URGENT_RULE_COLUMNS

    source = services.behavioral_daily_source()
    child = source["child_ids"][0]
    rows = services._behavioral_child_rows(source["daily"], child).head(16).copy()
    for column in URGENT_RULE_COLUMNS:
        rows[column] = 0
    for column in ("screen_monitor_observed", "language_monitor_observed"):
        rows[column] = 1
    rows["screen_escalated_incidents"] = 1
    partial_without_hit, partial_with_hit, complete = rows.index[-3:]
    rows.loc[[partial_without_hit, partial_with_hit], "screen_monitor_observed"] = 0
    rows.at[partial_with_hit, "screen_self_harm_imagery_incidents"] = 1
    monkeypatch.setattr(services, "behavioral_daily_source",
                        lambda: {"source": "demo", "daily": rows, "child_ids": [child]})
    monkeypatch.setattr(services, "enhanced_bundle_bytes", lambda: None)
    app = AppTest.from_string(
        "from ui import services\n"
        "from ui.legacy_pages._behavioral_upgrade import render\n"
        "render(services.behavioral_daily_source())"
    ).run()
    assert not app.exception
    selector = f"f2_activity_{child}"

    def texts(elements):
        return "\n".join(item.value for item in elements)

    assert app.selectbox(key=selector).value == rows.at[complete, "date"]
    assert "No urgent alert was recorded for this day." in texts(app.markdown)
    assert "Safety monitoring was incomplete for the selected day." not in texts(app.warning)
    assert "Safety monitoring was incomplete on other recorded days." in texts(app.warning)
    assert "Earlier activity records are available." in texts(app.markdown)
    assert "The activity records needed for an AI check are available." in texts(app.markdown)
    assert "Too few earlier records could be used to establish the usual pattern." in texts(app.markdown)

    app.selectbox(key=selector).set_value(rows.at[partial_without_hit, "date"]).run()
    assert not app.exception
    assert "Safety monitoring was incomplete for the selected day." in texts(app.warning)
    assert "The available records contained no urgent alert" in texts(app.markdown)
    assert "missing records needed for an AI check" in texts(app.markdown)

    app.selectbox(key=selector).set_value(rows.at[partial_with_hit, "date"]).run()
    assert not app.exception
    assert "Safety monitoring was incomplete for the selected day." in texts(app.warning)
    assert f"Selected day ({rows.at[partial_with_hit, 'date']})" in texts(app.warning)
    assert "Images that may show self-harm" in texts(app.warning)
    assert "seek appropriate human support" in texts(app.markdown)
    assert "The available records contained no urgent alert" not in texts(app.markdown)

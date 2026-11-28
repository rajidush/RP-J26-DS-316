from dataclasses import replace
from datetime import date, timedelta
from uuid import uuid4

import joblib
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from component4_profiling_xai.src import behavioral_anomaly, behavioral_profile, urgent_alerts
from test_dashboard_navigation import APP, main_text, profiles  # noqa: F401
from test_safety_notifications import isolated_app_database  # noqa: F401
from ui import activity_summary as activity, child_selection as selection, safety_summary as safety, services
from ui.legacy_pages import _behavioral_upgrade


END = date(2026, 8, 30)
CHILD = "fictional-partial"


@pytest.fixture
def records():
    base = {**dict.fromkeys(behavioral_anomaly.FEATURE_SET_A, 0.0),
            **dict.fromkeys(urgent_alerts.URGENT_RULE_COLUMNS, 0.0),
            "child_id": CHILD, "screen_monitor_observed": 1.0, "language_monitor_observed": 1.0}
    scenarios = {
        29: {"language_flags": 2},  # First day of the 30-day window.
        7: {"language_flags": 99},  # Outside the 7-day window.
        6: {},  # Recorded zero in both sources.
        5: {"language_monitor_observed": 0, "language_flags": 98, "screen_self_harm_imagery_incidents": 1},
        4: {"screen_monitor_observed": 0, "language_monitor_observed": 0},
        # Day 3 is absent.
        2: {"screen_flagged_seconds": float("nan"), "language_flags": 3},
        1: {"screen_monitor_observed": float("nan"), "language_flags": 1},
        0: {"screen_monitor_observed": 0, "language_flags": 4, "language_urgent_flags": 1},
    }
    return pd.DataFrame([{**base, **values, "date": END - timedelta(days=offset)}
                         for offset, values in scenarios.items()])


@pytest.mark.parametrize("missing_flag", [pd.NA, None, float("nan")])
def test_coverage_and_nullable_flags_match_research_contract(records, missing_flag):
    bundle = joblib.load(behavioral_profile.BUNDLE)
    matrix, _ = behavioral_profile.profiles(records, behavioral_profile.ProfileConfig(**bundle["profile"]))
    expected = behavioral_profile.score_history(records, matrix, bundle)
    result = services.enhanced_behavioral_assessment(records, CHILD, END, services.enhanced_bundle_bytes())
    pd.testing.assert_frame_equal(result.history, expected)
    assert result.history["flag"].notna().tolist() == [True, True, True, False, False, False, False, False]

    nullable = result.history["flag"].astype(object).where(result.history["flag"].notna(), missing_flag)
    summary = safety.summarize(replace(result, history=result.history.assign(flag=nullable)))
    assert summary.latest_unusual is None and safety.LATEST_UNASSESSED in summary.behavioral_message
    assert summary.assessed_dates == frozenset(expected.loc[expected["flag"].notna(), "date"])
    rule_matches = urgent_alerts.urgent_alerts(records)
    assert {item.date for item in summary.history(safety.URGENT)} == set(
        rule_matches.loc[rule_matches["is_urgent"], "date"]
    )
    for days, total, language_days, assessed_days in [(7, 8, 4, 1), (30, 109, 6, 3)]:
        window = activity.summarize(records, END, days)
        counts = window.assessment_counts(summary)
        active = expected["date"].between(END - timedelta(days=days - 1), END)
        assert counts.unusual_days == expected.loc[active, "flag"].eq(True).sum()
        assert counts.assessed_days == assessed_days and counts.urgent_days == 2
        assert (window.language_flags, window.language_days, window.average_flags) == (
            total, language_days, total / language_days)
        chart = activity.language_history_figure(window, "blue")
        plotted = dict(zip(chart.data[0].x, chart.data[0].y))
        assert plotted[str(END - timedelta(days=3))] is None  # Absent row.
        assert plotted[str(END - timedelta(days=5))] is None  # Unavailable language.
        assert plotted[str(END - timedelta(days=6))] == 0  # Recorded zero.


@pytest.mark.parametrize("status", ["model_unavailable", "failed"])
def test_partial_activity_and_urgent_review_survive_assessment_unavailability(records, status, monkeypatch):
    def fail(*args):
        raise RuntimeError("synthetic scoring failure")

    def forbid_legacy(*args):
        pytest.fail("Unavailable enhanced assessment must not fall back")

    monkeypatch.setattr(services, "behavioral_report", forbid_legacy)
    contents = None
    if status == "failed":
        contents = services.enhanced_bundle_bytes()
        monkeypatch.setattr(behavioral_profile, "score_history", fail)
    result = services.enhanced_behavioral_assessment(records.tail(1), CHILD, END, contents)
    summary = safety.summarize(result)
    assert result.status == status and result.history is None
    assert summary.behavioral_message == (
        safety.MODEL_UNAVAILABLE if status == "model_unavailable" else safety.BEHAVIOR_FAILED
    )
    for days in (7, 30):
        window = activity.summarize(result.rows, END, days)
        counts = window.assessment_counts(summary)
        assert window.language_flags == 4 and window.average_flags == 4
        assert counts.unusual_days is None and counts.assessed_days == 0 and counts.urgent_days == 1


@pytest.mark.usefixtures("isolated_app_database")
def test_partial_monitoring_pages_and_cache(profiles, records, monkeypatch, navigate_dashboard):
    profile = profiles.create("Fictional local profile", submission_token=uuid4())
    rows = records.tail(1).copy()
    rows.loc[:, "screen_monitor_observed"] = 1.0
    source = {"source": "demo", "child_ids": [CHILD], "daily": rows}
    monkeypatch.setattr(services, "behavioral_daily_source", lambda: source)
    original = services.enhanced_behavioral_assessment
    calls = []

    def assess(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(services, "enhanced_behavioral_assessment", assess)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox(key=selection.PROFILE_WIDGET).set_value(profile.id).run()
    navigate_dashboard(at, "Behavioral Anomaly")
    assert "Monitoring data has not been connected" in main_text(at) and not calls
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    assert len(calls) == 1 and "couldn’t assess" not in main_text(at)
    assert at.warning  # Independently recorded urgent match on an assessed day.
    at.run()
    assert len(calls) == 1

    def unavailable_xai(*args):
        pytest.fail("An unassessed endpoint should not run XAI")

    monkeypatch.setattr(_behavioral_upgrade, "ExplanationRequest", unavailable_xai)
    rows.loc[:, "screen_monitor_observed"] = 0.0
    at.run()
    assert len(calls) == 2 and not at.exception
    assert "couldn’t assess this day" in main_text(at)
    assert "An AI explanation is unavailable for this day." in main_text(at)
    assert "Language that needs urgent review" in at.warning[0].value
    navigate_dashboard(at, "Safety Notifications")
    assert not at.exception and len(calls) == 2
    assert [item.label for item in at.get("status")] == [f"{END} · {safety.URGENT}"]
    navigate_dashboard(at, "Overview")
    at.button(key="overview-run-behavioral").click().run()
    assert "insufficient monitoring information to assess unusual activity" in main_text(at)
    assert "Most recent urgent safety concern" in main_text(at)
    navigate_dashboard(at, "Activity Summary")
    for period in ("Last 7 days", "Last 30 days"):
        at.radio(key="activity-period-widget").set_value(period).run()
        assert [metric.value for metric in at.metric] == ["4", "—", "1"]
        assert "No days could be assessed" in main_text(at)

    rows.loc[:, ["language_monitor_observed", "language_urgent_flags"]] = 0.0
    navigate_dashboard(at, "Behavioral Anomaly")
    assert safety.URGENT_INSUFFICIENT in main_text(at)
    assert "The available records contained no urgent alert" not in main_text(at)
    assert len(calls) == 3 and not at.exception
    at.toggle(key=selection.DEMO_WIDGET).set_value(False).run()
    assert "Monitoring data has not been connected" in main_text(at)
    assert selection._ENHANCED_RESULT not in at.session_state

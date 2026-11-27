from dataclasses import replace
from datetime import date
from uuid import uuid4

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from parent_dashboard_test_helpers import follow_page_link
from test_dashboard_navigation import APP, main_text, profiles  # noqa: F401
from test_safety_notifications import (  # noqa: F401
    FIRST, LAST, MIDDLE, daily, isolated_app_database, predictions, source_for,
)
from ui import activity_summary as activity, child_selection as selection, safety_summary as safety, services


def test_inclusive_calendar_windows_and_observed_average():
    end = date(2026, 8, 30)
    records = pd.DataFrame([
        ("a", date(2026, 7, 31), 1, 500),
        ("a", date(2026, 8, 1), 1, 8),
        ("a", date(2026, 8, 23), 1, 4),
        ("a", date(2026, 8, 24), 1, 2),
        ("a", date(2026, 8, 26), 1, 0),
        ("a", date(2026, 8, 28), 0, 100),
        ("a", end, 1, 10),
        ("a", date(2026, 8, 31), 1, 900),
    ], columns=["child_id", "date", "language_monitor_observed", "language_flags"])
    week = activity.summarize(records, end, 7)
    assert (week.start, week.end, week.days) == (date(2026, 8, 24), end, 7)
    assert (week.records, week.language_days, week.language_flags, week.average_flags) == (4, 3, 12, 4)
    assert [count for _, count in week.language_history] == [2, None, 0, None, None, None, 10]

    month = activity.summarize(records, end, 30)
    assert (month.start, month.end, month.days) == (date(2026, 8, 1), end, 30)
    assert (month.records, month.language_days, month.language_flags, month.average_flags) == (6, 5, 24, 4.8)

    earlier = activity.summarize(records, date(2026, 8, 26), 7)
    assert earlier.language_flags == 6
    assert not any(day > earlier.end for day, _ in earlier.language_history)
    chart = activity.language_history_figure(week, "blue")
    assert list(chart.data[0].x) == [str(day) for day, _ in week.language_history]
    assert list(chart.data[0].y) == [2, None, 0, None, None, None, 10]
    assert list(chart.data[1].x) == ["2026-08-26"] and list(chart.data[1].y) == [0]


@pytest.mark.parametrize("value", [None, pd.NA, "bad", -1, 1.5, float("inf")])
def test_invalid_counts_and_missing_source_fields(daily, value):
    rows = daily[daily["child_id"] == "demo-a"].assign(language_flags=value)
    assert activity.summarize(rows, LAST, 7).language_flags is None
    assert activity.summarize(rows.drop(columns="language_flags"), LAST, 7).language_flags is None
    assert activity.summarize(rows.drop(columns="language_monitor_observed"), LAST, 7).language_flags is None
    assert activity.summarize(rows.iloc[:0], LAST, 7).records == 0


def test_duplicate_days_and_mixed_identities_rejected(daily):
    with pytest.raises(ValueError, match="one row"):
        activity.summarize(daily, LAST, 7)
    rows = daily[daily["child_id"] == "demo-a"]
    with pytest.raises(ValueError, match="one row"):
        activity.summarize(pd.concat([rows, rows]), LAST, 7)


def test_assessment_counts_use_unique_dates_and_coverage(daily, predictions):
    rows = safety.records_for(source_for(daily), "demo-a")
    reviewed = safety.summarize(rows)
    duplicate = reviewed.history(safety.URGENT)[0]
    distinct = safety.Notification(duplicate.date, safety.URGENT, duplicate.context, "Another rule")
    reviewed = replace(reviewed, notifications=reviewed.notifications + (duplicate, distinct))
    result = activity.summarize(rows, MIDDLE, 7).assessment_counts(reviewed)
    assert (result.unusual_days, result.urgent_days, result.assessed_days, result.complete_urgent_days) == (1, 2, 2, 2)
    result = activity.summarize(rows, LAST, 7).assessment_counts(reviewed)
    assert (result.unusual_days, result.urgent_days) == (1, 3)
    unavailable = activity.summarize(rows, LAST, 7).assessment_counts(safety.SafetySummary())
    assert unavailable.unusual_days is None and unavailable.urgent_days is None


def test_missing_model_preserves_safety_counts(daily, tmp_path, monkeypatch):
    monkeypatch.setenv("C4_F2_ARTIFACT_DIR", str(tmp_path))
    rows = safety.records_for(source_for(daily), "demo-a")
    reviewed = safety.summarize(rows)
    result = activity.summarize(rows, LAST, 7).assessment_counts(reviewed)
    assert reviewed.behavioral_message == safety.BEHAVIOR_FAILED
    assert result.unusual_days is None and result.assessed_days == 0
    assert result.urgent_days == 3


@pytest.mark.usefixtures("isolated_app_database")
def test_period_context_navigation_and_inference_reuse(profiles, daily, predictions, monkeypatch, navigate_dashboard):
    daily.loc[daily["date"] == LAST, "language_flags"] = 5
    daily.loc[daily["child_id"] == "demo-b", "language_flags"] = 23
    monkeypatch.setattr(services, "behavioral_daily_source", lambda: source_for(daily))
    profile = profiles.create("Local child", submission_token=uuid4())
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox(key=selection.PROFILE_WIDGET).set_value(profile.id).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    at.selectbox(key="overview-demo-emotional").set_value("C001").run()
    at.selectbox(key="overview-demo-behavioral").set_value("demo-a").run()
    assert "Recorded language flags: 5 across 3 of 7 days" in main_text(at)
    assert not predictions
    follow_page_link(at, at.main, "Open Activity Summary")
    assert not at.exception
    assert at.radio(key="activity-period-widget").value == "Last 7 days"
    assert [metric.value for metric in at.metric] == ["5", "1", "3"]
    assert len(predictions) == 1 and len(at.get("plotly_chart")) == 1

    at.radio(key="activity-period-widget").set_value("Last 30 days").run()
    assert "2026-06-05 to 2026-07-04" in main_text(at) and len(predictions) == 1
    at.selectbox(key="activity-assessment-date-demo-a").set_value(MIDDLE).run()
    assert [metric.value for metric in at.metric] == ["0", "1", "2"]
    assert predictions[-1] == ("demo-a", (FIRST, MIDDLE))
    follow_page_link(at, at.main, "Open Safety Notifications")
    assert at.selectbox(key="safety-assessment-date-demo-a").value == MIDDLE
    assert len([item for item in at.get("status") if safety.URGENT in item.label]) == 2
    assert len(predictions) == 2

    navigate_dashboard(at, "Overview")
    assert "Recorded language flags: 0 across 2 of 30 days" in main_text(at)
    assert "2026-06-04 to 2026-07-03" in main_text(at)
    follow_page_link(at, at.main, "Open Activity Summary")
    at.selectbox(key="activity-demo-child").set_value("demo-b").run()
    assert at.selectbox(key="activity-assessment-date-demo-b").value == FIRST
    assert [metric.value for metric in at.metric] == ["23", "0", "0"]
    assert len(predictions) == 3
    at.toggle(key=selection.DEMO_WIDGET).set_value(False).run()
    assert "Local child" in main_text(at) and "not been connected" in main_text(at)
    assert not at.metric and not at.get("plotly_chart")
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    assert at.selectbox(key="activity-demo-child").value is None
    assert not at.metric


@pytest.mark.usefixtures("isolated_app_database")
def test_local_selection_and_demo_source_errors(profiles, monkeypatch, navigate_dashboard):
    at = AppTest.from_file(APP, default_timeout=60).run()
    navigate_dashboard(at, "Activity Summary")
    assert "There are no active child profiles" in main_text(at)
    follow_page_link(at, at.main, "Go to Child Profiles")
    assert "Child Profiles" in main_text(at)

    first = profiles.create("First local child", submission_token=uuid4())
    second = profiles.create("Second local child", submission_token=uuid4())

    def unavailable():
        raise OSError("private file path")

    monkeypatch.setattr(services, "behavioral_daily_source", unavailable)
    navigate_dashboard(at, "Activity Summary")
    assert "Select a child profile in the sidebar" in main_text(at) and not at.error
    at.selectbox(key=selection.PROFILE_WIDGET).set_value(first.id).run()
    assert "First local child" in main_text(at) and not at.metric
    at.selectbox(key=selection.PROFILE_WIDGET).set_value(second.id).run()
    assert "Second local child" in main_text(at) and "First local child" not in main_text(at)
    profiles.archive(second.id)
    at.run()
    assert "Select a child profile in the sidebar" in main_text(at)
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    assert [item.value for item in at.error] == ["Fictional activity records could not be loaded."]

    monkeypatch.setattr(services, "behavioral_daily_source", lambda: {"source": "demo", "child_ids": []})
    at.run()
    assert "No fictional activity records are available" in main_text(at)
    assert not at.metric and not at.get("plotly_chart")

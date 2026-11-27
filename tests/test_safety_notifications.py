from datetime import date
from uuid import uuid4

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from conftest import follow_page_link
from test_dashboard_navigation import APP, main_text, profiles  # noqa: F401
from ui import child_selection as selection, safety_summary as safety, services


FIRST = date(2026, 7, 1)
MIDDLE = date(2026, 7, 3)
LAST = date(2026, 7, 4)
REAL_BEHAVIORAL_REPORT = services.behavioral_report


@pytest.fixture
def isolated_app_database(temp_db, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", str(temp_db.url))


@pytest.fixture
def daily():
    features = {name: 0.0 for name in services._f2_module().FEATURE_SET_A}
    days = [
        ("demo-a", LAST, 0, 1),
        ("demo-b", FIRST, 0, 0),
        ("demo-a", FIRST, 1, 0),
        ("demo-a", MIDDLE, 2, 1),
    ]
    return pd.DataFrame([
        dict(features, child_id=child, date=day, language_urgent_flags=urgent,
             screen_self_harm_imagery_incidents=imagery, screen_monitor_observed=1,
             language_monitor_observed=1)
        for child, day, urgent, imagery in days
    ])


@pytest.fixture
def predictions(monkeypatch):
    calls = []

    def report(rows, child):
        calls.append((child, tuple(rows["date"])))
        explanation = pd.DataFrame([dict(feature="language_flags", observed=2.0, typical=1.0,
                                         direction="higher than typical")])
        return {
            "daily": rows[["date"]].assign(unusual=rows["date"].eq(MIDDLE), imputed=False),
            "explanations": {MIDDLE: explanation},
        }

    monkeypatch.setattr(services, "behavioral_report", report)
    return calls


def source_for(daily):
    return {"source": "demo", "child_ids": ["demo-a", "demo-b"], "daily": daily}


def test_order_categories_and_independent_rules(daily, predictions):
    rows = safety.records_for(source_for(daily), "demo-a")
    result = safety.summarize(rows)
    assert [(item.date, item.category) for item in result.notifications] == [
        (LAST, safety.URGENT), (MIDDLE, safety.URGENT), (MIDDLE, safety.BEHAVIORAL),
        (FIRST, safety.URGENT),
    ]
    assert "urgent language flag (2)" in result.history(safety.URGENT)[1].description
    assert "self-harm imagery on screen" in result.history(safety.URGENT)[1].description
    assert "more language flags" in result.history(safety.BEHAVIORAL)[0].description
    assert result.latest_unusual is False
    assert "no monitoring record" in result.coverage
    assert str(LAST) in result.urgent_message
    assert predictions == [("demo-a", (FIRST, MIDDLE, LAST))]
    with pytest.raises(ValueError):
        safety.records_for(source_for(daily), "unrelated-profile")


def test_deduplication_preserves_distinct_same_day_results(daily, predictions, monkeypatch):
    original = services.behavioral_urgent_review

    def repeated(rows):
        alerts, available, complete = original(rows)
        distinct = alerts.iloc[[0]].assign(reasons="grooming language flag (1)")
        return pd.concat([alerts, alerts.iloc[[0]], distinct]), available, complete

    monkeypatch.setattr(services, "behavioral_urgent_review", repeated)
    result = safety.summarize(safety.records_for(source_for(daily), "demo-a"))
    same_day = [item for item in result.history(safety.URGENT) if item.date == FIRST]
    assert len(same_day) == 2
    assert len({item.description for item in same_day}) == 2
    assert str(LAST) in result.urgent_message


def test_failed_model_keeps_real_urgent_matches(daily, tmp_path, monkeypatch):
    monkeypatch.setenv("C4_F2_ARTIFACT_DIR", str(tmp_path))
    result = safety.summarize(safety.records_for(source_for(daily), "demo-a"))
    assert result.behavioral_message == safety.BEHAVIOR_FAILED
    assert result.latest_unusual is None
    assert not result.history(safety.BEHAVIORAL)
    assert [item.date for item in result.history(safety.URGENT)] == [LAST, MIDDLE, FIRST]


def test_coverage_no_records_insufficient_and_no_matches(daily, predictions):
    assert safety.summarize(daily.iloc[:0]).urgent_message == safety.NO_RECORDS

    rows = daily[daily["child_id"] == "demo-b"]
    valid = safety.summarize(rows)
    assert "No urgent safety concerns were identified" in valid.urgent_message
    assert not valid.notifications and not valid.coverage

    missing = safety.summarize(rows.assign(screen_monitor_observed=0, language_monitor_observed=0))
    assert "insufficient" in missing.behavioral_message
    assert "insufficient" in missing.urgent_message
    assert missing.coverage and not missing.notifications
    assert "0" not in missing.urgent_message

    partial = safety.summarize(rows.assign(screen_self_harm_imagery_incidents=float("nan")))
    assert partial.coverage and "available records" in partial.urgent_message

    urgent_only = daily[daily["date"] == LAST].assign(screen_monitor_observed=0, language_monitor_observed=0)
    result = safety.summarize(urgent_only)
    assert "insufficient" in result.behavioral_message
    assert result.history(safety.URGENT)[0].limitation


@pytest.mark.usefixtures("isolated_app_database")
def test_demo_navigation_filters_context_and_cache(profiles, daily, predictions, monkeypatch, navigate_dashboard):
    local = profiles.create("Demo-a local profile", submission_token=uuid4())
    monkeypatch.setattr(services, "behavioral_daily_source", lambda: source_for(daily))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox(key=selection.PROFILE_WIDGET).set_value(local.id).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    at.selectbox(key="overview-demo-behavioral").set_value("demo-a").run()
    at.button(key="overview-run-behavioral").click().run()
    message = f"Most recent urgent safety concern: {LAST}."
    assert message in main_text(at)
    follow_page_link(at, at.main, "Open Safety Notifications")
    assert not at.exception and len(predictions) == 1
    assert at.selectbox(key="safety-demo-child").value == "demo-a"
    assert message in main_text(at) and local.display_name not in main_text(at)
    assert len(at.get("status")) == 4

    at.selectbox(key="safety-assessment-date-demo-a").set_value(MIDDLE).run()
    assert str(LAST) not in main_text(at)
    assert predictions[-1] == ("demo-a", (FIRST, MIDDLE))
    at.selectbox(key="safety-history-category").set_value(safety.BEHAVIORAL).run()
    assert [item.label for item in at.get("status")] == [f"{MIDDLE} · {safety.BEHAVIORAL}"]
    at.selectbox(key="safety-history-category").set_value(safety.URGENT).run()
    assert [item.label for item in at.get("status")] == [f"{MIDDLE} · {safety.URGENT}", f"{FIRST} · {safety.URGENT}"]
    assert len(predictions) == 2
    navigate_dashboard(at, "Overview")
    at.button(key="overview-run-behavioral").click().run()
    assert f"Most recent urgent safety concern: {MIDDLE}." in main_text(at)
    assert str(LAST) not in main_text(at)
    assert len(predictions) == 2

    follow_page_link(at, at.main, "Open Safety Notifications")
    at.selectbox(key="safety-demo-child").set_value("demo-b").run()
    assert len(predictions) == 3
    assert not at.get("status")
    assert "No urgent safety concerns were identified" in main_text(at)
    monkeypatch.setattr(services, "behavioral_report", REAL_BEHAVIORAL_REPORT)
    follow_page_link(at, at.main, "Open Behavioral Anomaly")
    assert not at.exception
    assert at.selectbox(key=selection.F2_DEMO_CHILD).value == "demo-b"
    navigate_dashboard(at, "Safety Notifications")
    at.toggle(key=selection.DEMO_WIDGET).set_value(False).run()
    assert local.display_name in main_text(at)
    assert "Monitoring data has not been connected" in main_text(at)
    assert not at.get("status")
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    assert at.selectbox(key="safety-demo-child").value is None
    assert not at.get("status")


@pytest.mark.usefixtures("isolated_app_database")
def test_empty_profiles_and_source_failure(profiles, monkeypatch, navigate_dashboard):
    at = AppTest.from_file(APP, default_timeout=30).run()
    navigate_dashboard(at, "Safety Notifications")
    assert "There are no active child profiles" in main_text(at)
    follow_page_link(at, at.main, "Go to Child Profiles")
    assert "Child Profiles" in main_text(at)

    def unavailable():
        raise OSError("private path")

    monkeypatch.setattr(services, "behavioral_daily_source", unavailable)
    navigate_dashboard(at, "Safety Notifications")
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    assert [item.value for item in at.error] == ["Fictional monitoring records could not be loaded."]

    monkeypatch.setattr(services, "behavioral_daily_source", lambda: {"source": "demo", "child_ids": []})
    at.run()
    assert safety.NO_RECORDS in main_text(at)
    assert not at.get("status")


@pytest.mark.usefixtures("isolated_app_database")
def test_failed_model_keeps_urgent_cards_and_overview_summary(
    profiles, daily, tmp_path, monkeypatch,
):
    monkeypatch.setenv("C4_F2_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(services, "behavioral_daily_source", lambda: source_for(daily))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    at.selectbox(key="overview-demo-behavioral").set_value("demo-a").run()
    at.button(key="overview-run-behavioral").click().run()
    urgent_message = f"Most recent urgent safety concern: {LAST}."
    assert safety.BEHAVIOR_FAILED in main_text(at)
    assert urgent_message in main_text(at)
    follow_page_link(at, at.main, "Open Safety Notifications")
    assert not at.exception
    assert safety.BEHAVIOR_FAILED in main_text(at)
    assert urgent_message in main_text(at)
    assert len(at.get("status")) == 3
    assert all(safety.URGENT in item.label for item in at.get("status"))

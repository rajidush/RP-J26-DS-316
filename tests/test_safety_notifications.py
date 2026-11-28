import os
from datetime import date
from uuid import uuid4

import joblib
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from component4_profiling_xai.src import behavioral_profile

from parent_dashboard_test_helpers import follow_page_link
from test_dashboard_navigation import APP, main_text, profiles  # noqa: F401
from ui import child_selection as selection, safety_summary as safety, services


FIRST = date(2026, 7, 1)
MIDDLE = date(2026, 7, 3)
LAST = date(2026, 7, 4)


def summarize_rows(rows):
    child = rows["child_id"].iloc[0] if not rows.empty else "demo-a"
    through = rows["date"].max() if not rows.empty else LAST
    return safety.summarize(services.enhanced_behavioral_assessment(
        rows, child, through, services.enhanced_bundle_bytes(),
    ))


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
    original = behavioral_profile.score_history

    def report(rows, matrix, bundle):
        calls.append((rows.iloc[0]["child_id"], tuple(rows["date"])))
        history = original(rows, matrix, bundle)
        history.loc[history["flag"].notna(), "flag"] = history["date"].eq(MIDDLE)
        return history

    monkeypatch.setattr(behavioral_profile, "score_history", report)
    return calls


def source_for(daily):
    return {"source": "demo", "child_ids": ["demo-a", "demo-b"], "daily": daily}


def test_order_categories_and_independent_rules(daily, predictions):
    rows = safety.records_for(source_for(daily), "demo-a")
    result = summarize_rows(rows)
    assert [(item.date, item.category) for item in result.notifications] == [
        (LAST, safety.URGENT), (MIDDLE, safety.URGENT), (MIDDLE, safety.BEHAVIORAL),
        (FIRST, safety.URGENT),
    ]
    assert "urgent language flag (2)" in result.history(safety.URGENT)[1].description
    assert "self-harm imagery on screen" in result.history(safety.URGENT)[1].description
    assert "differed from typical patterns" in result.history(safety.BEHAVIORAL)[0].description
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
    result = summarize_rows(safety.records_for(source_for(daily), "demo-a"))
    same_day = [item for item in result.history(safety.URGENT) if item.date == FIRST]
    assert len(same_day) == 2
    assert len({item.description for item in same_day}) == 2
    assert str(LAST) in result.urgent_message


def test_failed_model_keeps_real_urgent_matches(daily, tmp_path, monkeypatch):
    monkeypatch.setattr(behavioral_profile, "BUNDLE", tmp_path / "selected.joblib")
    result = summarize_rows(safety.records_for(source_for(daily), "demo-a"))
    assert result.behavioral_message == safety.MODEL_UNAVAILABLE
    assert result.latest_unusual is None
    assert not result.history(safety.BEHAVIORAL)
    assert [item.date for item in result.history(safety.URGENT)] == [LAST, MIDDLE, FIRST]


def test_coverage_no_records_insufficient_and_no_matches(daily, predictions):
    assert summarize_rows(daily.iloc[:0]).urgent_message == safety.NO_RECORDS

    rows = daily[daily["child_id"] == "demo-b"]
    valid = summarize_rows(rows)
    assert "No urgent safety concerns were identified" in valid.urgent_message
    assert not valid.notifications and not valid.coverage

    missing = summarize_rows(rows.assign(screen_monitor_observed=0, language_monitor_observed=0))
    assert "insufficient" in missing.behavioral_message
    assert "insufficient" in missing.urgent_message
    assert missing.coverage and not missing.notifications
    assert "0" not in missing.urgent_message

    partial = summarize_rows(rows.assign(screen_self_harm_imagery_incidents=float("nan")))
    assert partial.coverage and "available records" in partial.urgent_message

    urgent_only = daily[daily["date"] == LAST].assign(screen_monitor_observed=0, language_monitor_observed=0)
    result = summarize_rows(urgent_only)
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
    profiles, daily, tmp_path, monkeypatch, navigate_dashboard,
):
    monkeypatch.setattr(behavioral_profile, "BUNDLE", tmp_path / "selected.joblib")
    monkeypatch.setattr(services, "behavioral_daily_source", lambda: source_for(daily))
    monkeypatch.setattr(services, "behavioral_report",
                        lambda *args: pytest.fail("Missing enhanced model fell back to the legacy model"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    at.selectbox(key="overview-demo-behavioral").set_value("demo-a").run()
    at.button(key="overview-run-behavioral").click().run()
    urgent_message = f"Most recent urgent safety concern: {LAST}."
    assert safety.MODEL_UNAVAILABLE in main_text(at)
    assert urgent_message in main_text(at)
    follow_page_link(at, at.main, "Open Safety Notifications")
    assert not at.exception
    assert safety.MODEL_UNAVAILABLE in main_text(at)
    assert urgent_message in main_text(at)
    assert len(at.get("status")) == 3
    assert all(safety.URGENT in item.label for item in at.get("status"))
    navigate_dashboard(at, "Activity Summary")
    assert safety.MODEL_UNAVAILABLE in main_text(at)
    urgent = next(item for item in at.metric if item.label == "Days with urgent safety flags")
    assert urgent.value == "3"
    navigate_dashboard(at, "Behavioral Anomaly")
    assert not at.exception
    assert safety.MODEL_UNAVAILABLE in main_text(at)
    assert str(LAST) in "\n".join(item.value for item in at.warning)


@pytest.mark.usefixtures("isolated_app_database")
def test_overview_refreshes_for_source_date_and_artifact_revisions(
    profiles, daily, predictions, tmp_path, monkeypatch, navigate_dashboard,
):
    source = source_for(daily)
    monkeypatch.setattr(services, "behavioral_daily_source", lambda: source)
    bundle = joblib.load(behavioral_profile.BUNDLE)
    artifact = tmp_path / "selected.joblib"
    joblib.dump({**bundle, "test_revision": 1}, artifact, compress=0)
    monkeypatch.setattr(behavioral_profile, "BUNDLE", artifact)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    at.selectbox(key="overview-demo-behavioral").set_value("demo-a").run()
    at.button(key="overview-run-behavioral").click().run()
    assert f"Most recent urgent safety concern: {LAST}." in main_text(at)
    assert len(predictions) == 1

    child_rows = daily["child_id"] == "demo-a"
    rule_columns = list(services._urgent_module().URGENT_RULE_COLUMNS)
    daily.loc[child_rows, rule_columns] = 0
    daily.loc[child_rows & daily["date"].eq(LAST), "language_flags"] = 6
    at.run()
    assert "No urgent safety concerns were identified" in main_text(at)
    assert "Recorded language flags: 6" in main_text(at)
    assert len(predictions) == 2
    follow_page_link(at, at.main, "Open Safety Notifications")
    assert "No urgent safety concerns were identified" in main_text(at)
    assert len(predictions) == 2

    daily.loc[child_rows & daily["date"].eq(MIDDLE), "language_urgent_flags"] = 1
    navigate_dashboard(at, "Overview")
    assert f"Most recent urgent safety concern: {MIDDLE}." in main_text(at)
    assert len(predictions) == 3
    follow_page_link(at, at.main, "Open Activity Summary")
    assert [item.value for item in at.main.metric] == ["6", "1", "1"]
    assert len(predictions) == 3

    at.selectbox(key="activity-assessment-date-demo-a").set_value(FIRST).run()
    navigate_dashboard(at, "Overview")
    assert f"Assessment date: {FIRST}" in main_text(at)
    assert f"Most recent urgent safety concern: {MIDDLE}." not in main_text(at)
    assert "Recorded language flags: 0" in main_text(at)
    assert len(predictions) == 4
    stamp = artifact.stat()
    joblib.dump({**bundle, "test_revision": 2}, artifact, compress=0)
    os.utime(artifact, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert artifact.stat().st_size == stamp.st_size
    at.run()
    assert len(predictions) == 5
    at.run()
    assert len(predictions) == 5

    at.toggle(key=selection.DEMO_WIDGET).set_value(False).run()
    assert "Recorded language flags" not in main_text(at)
    assert selection.OVERVIEW_DEMO_RESULTS not in at.session_state

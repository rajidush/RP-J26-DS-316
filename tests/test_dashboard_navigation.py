from contextlib import closing
from uuid import uuid4

import pytest
from streamlit.testing.v1 import AppTest

from parent_dashboard_test_helpers import follow_page_link
from db.child_profiles import PROFILE_DB_ENV, ChildProfileRepository
from ui import child_selection as selection, services, theme


APP = str(services.ROOT / "ui" / "app.py")
PAGES = (
    ("Overview", "", "overview"),
    ("Child Profiles", "child-profile", "child-profile"),
    ("Emotional Pattern", "emotional-pattern", "emotional-pattern"),
    ("Behavioral Anomaly", "behavioral-anomaly", "behavioral-anomaly"),
    ("Safety Notifications", "safety-notifications", "safety-notifications"),
    ("Activity Summary", "activity-summary", "activity-summary"),
)


@pytest.fixture
def profiles(temp_db, tmp_path, monkeypatch):
    monkeypatch.setenv(PROFILE_DB_ENV, str(tmp_path / "profiles.db"))
    with closing(ChildProfileRepository()) as store:
        yield store


def main_text(at):
    return " ".join(str(item.proto) for item in at.main.get("html")) + " " + " ".join(
        item.value for item in [*at.main.info, *at.main.text, *at.main.caption, *at.main.markdown]
    )


def test_default_overview_and_empty_profiles_link(profiles):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "Overview" in main_text(at)
    assert "There are no active child profiles" in main_text(at)
    assert any("Parent Dashboard" in str(item.proto) for item in at.sidebar.get("html"))
    links = [item.proto for item in at.sidebar.get("page_link")]
    assert [(link.label, link.page) for link in links] == [(title, url) for title, url, _ in PAGES]
    assert len({link.icon for link in links if link.icon}) == 6
    sidebar = list(at.sidebar)
    profile_index = next(i for i, item in enumerate(sidebar) if item.type == "selectbox")
    link_indexes = [i for i, item in enumerate(sidebar) if item.type == "page_link"]
    toggle_index = next(i for i, item in enumerate(sidebar) if item.type == "toggle")
    assert profile_index < min(link_indexes) < max(link_indexes) < toggle_index
    assert not at.toggle(key=selection.DEMO_WIDGET).value
    follow_page_link(at, at.main, "Go to Child Profiles")
    assert not at.exception
    assert "Child Profiles" in main_text(at)


def test_streamlit_cold_discovery_has_no_legacy_pages():
    assert not list((services.ROOT / "ui" / "pages").glob("*.py"))
    assert (services.ROOT / "ui" / "legacy_pages" / "add_record.py").is_file()


def test_all_pages_keep_local_selection_and_use_only_profile_context(profiles, navigate_dashboard, monkeypatch):
    first = profiles.create("First child", age_group="8–10", avatar="owl", submission_token=uuid4())
    second = profiles.create("Second child", submission_token=uuid4())
    highlights = []
    monkeypatch.setattr(theme, "active_nav_css", highlights.append)

    def forbidden(*args, **kwargs):
        pytest.fail("A local dashboard page queried global records or assessments")

    for name in ("dashboard_kpis", "escalated_sessions", "risk_level_counts", "list_sessions",
                 "child_session_source", "behavioral_daily_source", "emotional_pattern_report",
                 "behavioral_report"):
        monkeypatch.setattr(services, name, forbidden)

    at = AppTest.from_file(APP, default_timeout=30).run()
    assert "Select a child profile in the sidebar" in main_text(at)
    at.selectbox(key=selection.PROFILE_WIDGET).set_value(first.id).run()
    assert "First child" in main_text(at)
    assert "🦉" in main_text(at)
    assert "Age group: 8–10" in main_text(at)
    assert first.id not in main_text(at)
    assert [link.proto.label for link in at.main.get("page_link")] == [
        "Open Emotional Pattern", "Open Behavioral Anomaly", "Open Safety Notifications", "Open Activity Summary",
    ]
    for title in ("Open Emotional Pattern", "Open Behavioral Anomaly", "Open Safety Notifications", "Open Activity Summary"):
        follow_page_link(at, at.main, title)
        assert not at.exception
        assert at.selectbox(key=selection.PROFILE_WIDGET).value == first.id
        navigate_dashboard(at, "Overview")

    for title, _, slug in PAGES:
        navigate_dashboard(at, title)
        assert not at.exception, [error.value for error in at.exception]
        content = main_text(at)
        assert title in content
        assert highlights[-1] == f"nav-{slug}"
        assert at.selectbox(key=selection.PROFILE_WIDGET).value == first.id
        assert not at.main.metric and not at.main.dataframe and not at.main.get("plotly_chart")
        if title in {"Overview", "Emotional Pattern", "Behavioral Anomaly", "Safety Notifications", "Activity Summary"}:
            assert "First child" in content and "Second child" not in content
            assert "not been connected" in content

    navigate_dashboard(at, "Safety Notifications")
    at.selectbox(key=selection.PROFILE_WIDGET).set_value(second.id).run()
    assert "Safety Notifications" in main_text(at)
    assert "Second child" in main_text(at) and "First child" not in main_text(at)
    profiles.archive(second.id)
    at.run()
    assert at.selectbox(key=selection.PROFILE_WIDGET).value is None
    assert "No safety alerts" not in main_text(at)


def test_demo_mode_persists_between_shells_without_attributing_data(profiles, navigate_dashboard):
    child = profiles.create("Local child", submission_token=uuid4())
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox(key=selection.PROFILE_WIDGET).set_value(child.id).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    assert "separate fictional records" in main_text(at)
    for title in ("Safety Notifications", "Activity Summary", "Overview"):
        navigate_dashboard(at, title)
        assert not at.exception
        assert at.toggle(key=selection.DEMO_WIDGET).value
        assert at.selectbox(key=selection.PROFILE_WIDGET).proto.disabled
        assert "Local child" not in main_text(at)
        assert not at.main.metric and not at.main.dataframe and not at.main.get("plotly_chart")
    at.toggle(key=selection.DEMO_WIDGET).set_value(False).run()
    assert at.selectbox(key=selection.PROFILE_WIDGET).value == child.id
    assert "Local child" in main_text(at)


def test_overview_demo_summaries_use_independent_reports_and_clear_stale_results(profiles, monkeypatch):
    child = profiles.create("Local child", submission_token=uuid4())
    calls = []
    for name in ("emotional_pattern_report", "behavioral_report"):
        report = getattr(services, name)

        def record(data, child_id, *, report=report, name=name):
            calls.append((name, child_id))
            return report(data, child_id)

        monkeypatch.setattr(services, name, record)

    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key=selection.PROFILE_WIDGET).set_value(child.id).run()
    assert "Local child" in main_text(at)
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    at.selectbox(key="overview-demo-emotional").set_value("C004").run()
    at.selectbox(key="overview-demo-behavioral").set_value("C009").run()
    assert not calls
    assert "Local child" not in main_text(at)

    at.button(key="overview-run-emotional").click().run()
    at.button(key="overview-run-behavioral").click().run()
    assert not at.exception, [error.value for error in at.exception]
    assert calls == [("emotional_pattern_report", "C004"), ("behavioral_report", "C009")]
    content = main_text(at)
    assert "Guarded responses" in content and "Assessment period" in content
    assert "The overall trend" in content
    assert "Assessment date" in content and "flagged for unusual activity" in content
    assert "Emotional Pattern demo · Fictional child C004" in content
    assert "Behavioral Anomaly demo · Fictional child C009" in content
    assert "Local child" not in content

    at.run()
    assert len(calls) == 2
    at.selectbox(key="overview-demo-behavioral").set_value("C001").run()
    assert "Behavioral Anomaly demo · Fictional child C009" not in main_text(at)
    assert "Assessment date" not in main_text(at)
    assert "Guarded responses" in main_text(at)
    at.toggle(key=selection.DEMO_WIDGET).set_value(False).run()
    assert "Local child" in main_text(at)
    assert "Guarded responses" not in main_text(at)
    assert "Safety information is unavailable" in main_text(at)
    assert selection.OVERVIEW_DEMO_RESULTS not in at.session_state


def test_overview_distinguishes_unavailable_insufficient_and_failed_demo_assessments(profiles, monkeypatch):
    def failed(*args):
        raise services.ModelUnavailableError({"reason": "Model path should stay private"})

    monkeypatch.setattr(services, "behavioral_daily_source", lambda: {"source": "demo", "child_ids": []})
    monkeypatch.setattr(services, "emotional_pattern_report", lambda *args: {"enough_data": False})
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    assert "Fictional records for this example are unavailable" in main_text(at)
    at.selectbox(key="overview-demo-emotional").set_value("C004").run()
    at.button(key="overview-run-emotional").click().run()
    assert "too few fictional sessions" in main_text(at)

    monkeypatch.setattr(services, "emotional_pattern_report", failed)
    at.button(key="overview-run-emotional").click().run()
    assert not at.exception
    assert "could not be generated" in main_text(at)
    assert "Model path should stay private" not in main_text(at)
    assert "No urgent rules matched" not in main_text(at)

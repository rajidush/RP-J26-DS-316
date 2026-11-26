from contextlib import closing
from uuid import uuid4

import pytest
from streamlit.testing.v1 import AppTest

from conftest import follow_page_link
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
    ("Settings", "settings", "settings"),
)


@pytest.fixture
def profiles(temp_db, tmp_path, monkeypatch):
    monkeypatch.setenv(PROFILE_DB_ENV, str(tmp_path / "profiles.db"))
    with closing(ChildProfileRepository()) as store:
        yield store


def main_text(at):
    return " ".join(str(item.proto) for item in at.main.get("html")) + " " + " ".join(
        item.value for item in at.main.info
    )


def test_default_overview_and_empty_profiles_link(profiles):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "Overview" in main_text(at)
    assert "There are no active child profiles" in main_text(at)
    assert any("Parent Dashboard" in str(item.proto) for item in at.sidebar.get("html"))
    links = [item.proto for item in at.sidebar.get("page_link")]
    assert [(link.label, link.page) for link in links] == [(title, url) for title, url, _ in PAGES]
    assert len({link.icon for link in links if link.icon}) == 7
    sidebar = list(at.sidebar)
    profile_index = next(i for i, item in enumerate(sidebar) if item.type == "selectbox")
    link_indexes = [i for i, item in enumerate(sidebar) if item.type == "page_link"]
    toggle_index = next(i for i, item in enumerate(sidebar) if item.type == "toggle")
    assert profile_index < min(link_indexes) < max(link_indexes) < toggle_index
    assert not at.toggle(key=selection.DEMO_WIDGET).value
    follow_page_link(at, at.main, "Go to Child Profiles")
    assert not at.exception
    assert "Child Profiles" in main_text(at)


def test_all_pages_keep_local_selection_and_use_only_profile_context(profiles, navigate_dashboard, monkeypatch):
    first = profiles.create("First child", avatar="owl", submission_token=uuid4())
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
    assert "fictional data" in main_text(at)
    for title in ("Safety Notifications", "Activity Summary", "Settings", "Overview"):
        navigate_dashboard(at, title)
        assert not at.exception
        assert at.toggle(key=selection.DEMO_WIDGET).value
        assert at.selectbox(key=selection.PROFILE_WIDGET).proto.disabled
        assert "Local child" not in main_text(at)
        assert not at.main.metric and not at.main.dataframe and not at.main.get("plotly_chart")
    at.toggle(key=selection.DEMO_WIDGET).set_value(False).run()
    assert at.selectbox(key=selection.PROFILE_WIDGET).value == child.id
    assert "Local child" in main_text(at)

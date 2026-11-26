from contextlib import closing
from uuid import uuid4

import pytest
from streamlit.testing.v1 import AppTest

from db.child_profiles import PROFILE_DB_ENV, ChildProfileRepository
from ui import child_selection as selection, services


APP = str(services.ROOT / "ui" / "app.py")
EMOTIONAL = "Emotional Pattern"
BEHAVIORAL = "Behavioral Anomaly"


@pytest.fixture
def profiles(temp_db, tmp_path, monkeypatch):
    monkeypatch.setenv(PROFILE_DB_ENV, str(tmp_path / "profiles.db"))
    with closing(ChildProfileRepository()) as repository:
        yield repository


def app():
    return AppTest.from_file(APP, default_timeout=120)


def selector(at):
    return at.selectbox(key=selection.PROFILE_WIDGET)


def assert_no_assessments(at, message):
    assert not at.exception
    assert message in at.main.info[0].value
    assert not at.main.selectbox
    assert not at.main.dataframe and not at.main.metric
    assert not at.main.get("plotly_chart")
    html = " ".join(str(item.proto) for item in at.main.get("html"))
    assert "Urgent Alerts" not in html
    assert "Emotional Wellbeing" not in html


def forbid_assessments(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Local mode attempted to load or evaluate assessment data")

    for name in ("child_session_source", "emotional_pattern_report",
                 "behavioral_daily_source", "behavioral_report"):
        monkeypatch.setattr(services, name, unexpected)


def test_empty_profiles_and_invalid_selection(profiles, monkeypatch, navigate_dashboard):
    forbid_assessments(monkeypatch)
    at = app().run()
    assert not selector(at).options
    assert selector(at).proto.disabled
    assert not at.toggle(key=selection.DEMO_WIDGET).value
    assert any("Child Profiles page" in item.value for item in at.sidebar.caption)
    navigate_dashboard(at, EMOTIONAL)
    assert_no_assessments(at, "Select a child profile")


@pytest.mark.parametrize("invalid", ["invalid UUID", str(uuid4())])
def test_invalid_selection_is_not_replaced_with_active_profile(profiles, monkeypatch, invalid, navigate_dashboard):
    forbid_assessments(monkeypatch)
    profiles.create("Available", submission_token=uuid4())
    at = app()
    at.session_state[selection.SELECTED_PROFILE] = invalid
    at.run()
    navigate_dashboard(at, EMOTIONAL)
    assert selector(at).value is None
    assert at.session_state[selection.SELECTED_PROFILE] is None
    assert_no_assessments(at, "Select a child profile")


def test_navigation_rename_switch_archive_restore_and_clear(profiles, monkeypatch, navigate_dashboard):
    forbid_assessments(monkeypatch)
    first = profiles.create("Alex", avatar="owl", submission_token=uuid4())
    second = profiles.create("Alex", avatar="owl", submission_token=uuid4())
    at = app().run()
    navigate_dashboard(at, EMOTIONAL)
    assert selector(at).options == ["🦉 Alex · Profile 1", "🦉 Alex · Profile 2"]
    assert at.session_state[selection.SELECTED_PROFILE] is None
    selector(at).set_value(first.id).run()
    assert_no_assessments(at, "Monitoring data has not been connected")
    navigate_dashboard(at, BEHAVIORAL)
    assert not at.exception, [error.value for error in at.exception]
    assert selector(at).value == first.id
    assert_no_assessments(at, "Monitoring data has not been connected")

    profiles.edit(first.id, "Renamed Alex")
    at.run()
    assert selector(at).value == first.id
    assert "Renamed Alex" in selector(at).options
    assert any("Renamed Alex" in str(item.proto) for item in at.main.get("html"))
    selector(at).set_value(second.id).run()
    assert at.session_state[selection.SELECTED_PROFILE] == second.id
    assert not any("Renamed Alex" in str(item.proto) for item in at.main.get("html"))

    profiles.archive(second.id)
    at.run()
    assert selector(at).value is None
    assert "🦉 Alex" not in selector(at).options
    assert_no_assessments(at, "Select a child profile")
    profiles.restore(second.id)
    at.run()
    assert selector(at).value is None
    assert "🦉 Alex" in selector(at).options
    selector(at).set_value(second.id).run()
    selector(at).set_value(None).run()
    assert at.session_state[selection.SELECTED_PROFILE] is None
    assert_no_assessments(at, "Select a child profile")


def test_demo_mode_is_explicit_and_does_not_reuse_monitoring_or_demo_results(
    profiles, monkeypatch, navigate_dashboard,
):
    first = profiles.create("C001", submission_token=uuid4())
    second = profiles.create("Local child", submission_token=uuid4())

    def unexpected_database_read():
        pytest.fail("Demo Mode attempted to list monitoring data")

    monkeypatch.setattr(services.comp3, "list_child_ids", unexpected_database_read)
    at = app().run()
    navigate_dashboard(at, EMOTIONAL)
    selector(at).set_value(first.id).run()
    assert_no_assessments(at, "Monitoring data has not been connected")
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    assert at.selectbox(key=selection.F1_DEMO_CHILD).value == "C001"
    assert any("SYNTHETIC DEMO DATA" in item.value for item in at.main.caption)
    assert at.main.get("plotly_chart")
    at.selectbox(key=selection.F1_DEMO_CHILD).set_value("C004").run()

    navigate_dashboard(at, BEHAVIORAL)
    assert at.selectbox(key=selection.F2_DEMO_CHILD).value == "C001"
    at.selectbox(key=selection.F2_DEMO_CHILD).set_value("C009").run()
    assert not at.exception, [error.value for error in at.exception]
    assert at.selectbox(key="f2_day_C009").value is not None
    assert at.main.dataframe
    at.toggle(key=selection.DEMO_WIDGET).set_value(False).run()
    assert selector(at).value == first.id
    assert_no_assessments(at, "Monitoring data has not been connected")
    assert "f2_day_C009" not in at.session_state
    selector(at).set_value(second.id).run()
    assert_no_assessments(at, "Monitoring data has not been connected")
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    assert at.selectbox(key=selection.F2_DEMO_CHILD).value == "C001"
    navigate_dashboard(at, EMOTIONAL)
    assert at.selectbox(key=selection.F1_DEMO_CHILD).value == "C001"
    at.toggle(key=selection.DEMO_WIDGET).set_value(False).run()
    assert selector(at).value == second.id
    assert_no_assessments(at, "Monitoring data has not been connected")

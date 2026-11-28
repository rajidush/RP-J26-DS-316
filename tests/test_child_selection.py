from contextlib import closing
from datetime import date
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
    assert "Behavior Overview" not in html
    assert not any("Urgent safety review" in item.value for item in at.main.warning)


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
    profiles, monkeypatch, navigate_dashboard, tmp_path,
):
    first = profiles.create("C001", submission_token=uuid4())
    second = profiles.create("Local child", submission_token=uuid4())

    def unexpected_database_read():
        pytest.fail("Demo Mode attempted to list monitoring data")

    monkeypatch.setattr(services.comp3, "list_child_ids", unexpected_database_read)
    monkeypatch.delenv("C4_F2_DAILY_CSV", raising=False)
    monkeypatch.delenv("C4_F2_LEGACY_VIEW", raising=False)
    original_source = services.behavioral_daily_source
    source = original_source()
    daily = source["daily"]
    prior_date = daily.loc[daily["child_id"].eq("C001"), "date"].min()
    # Give the two fictional children different valid dates without changing the bundled file.
    rows = daily.loc[~(daily["child_id"].eq("C009") & daily["date"].eq(prior_date))]
    c009_dates = sorted(rows.loc[rows["child_id"].eq("C009"), "date"])
    monkeypatch.setattr(services, "behavioral_daily_source", lambda uploaded_csv=None: (
        {**source, "daily": rows} if uploaded_csv is None else original_source(uploaded_csv)
    ))
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
    at.selectbox(key="f2_activity_C001").set_value(prior_date).run()
    assert at.session_state[selection.F2_SAFETY_DATE] == prior_date
    at.selectbox(key=selection.F2_DEMO_CHILD).set_value("C009").run()
    assert not at.exception, [error.value for error in at.exception]
    date_control = at.selectbox(key="f2_activity_C009")
    assert date_control.label == "Activity date"
    assert date_control.value == c009_dates[-1]
    assert str(prior_date) not in date_control.options
    historical_date = c009_dates[-2]
    date_control.set_value(historical_date).run()
    result = services.enhanced_behavioral_assessment(
        rows, "C009", historical_date, services.enhanced_bundle_bytes(),
    )
    flag = result.history.iloc[-1]["flag"]
    assert result.history["flag"].notna().iloc[-1]
    expected = "detected an unusual pattern" if flag else "detected no unusual pattern"
    assert any(expected in item.value for item in at.main.markdown)
    cached = at.session_state[selection._ENHANCED_RESULT][2]
    navigate_dashboard(at, "Overview")
    assert at.selectbox(key="overview-demo-behavioral").value == "C009"
    at.button(key="overview-run-behavioral").click().run()
    assert f"Assessment date: {historical_date}" in [item.value for item in at.main.caption]
    for page, child_key, date_key in (
        ("Safety Notifications", "safety-demo-child", "safety-assessment-date-C009"),
        ("Activity Summary", "activity-demo-child", "activity-assessment-date-C009"),
    ):
        navigate_dashboard(at, page)
        assert at.selectbox(key=child_key).value == "C009"
        assert at.selectbox(key=date_key).value == historical_date
        assert at.session_state[selection._ENHANCED_RESULT][2] is cached
    navigate_dashboard(at, BEHAVIORAL)
    assert at.selectbox(key="f2_activity_C009").value == historical_date
    at.toggle(key=selection.DEMO_WIDGET).set_value(False).run()
    assert selector(at).value == first.id
    assert_no_assessments(at, "Monitoring data has not been connected")
    # Streamlit may retain a dormant widget value. Assert the assessment context
    # is cleared, then revisit the same child to check it cannot restore the old date.
    assert selection.F2_SAFETY_DATE not in at.session_state
    assert selection._ENHANCED_RESULT not in at.session_state
    selector(at).set_value(second.id).run()
    assert_no_assessments(at, "Monitoring data has not been connected")
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    assert at.selectbox(key=selection.F2_DEMO_CHILD).value == "C001"
    at.selectbox(key=selection.F2_DEMO_CHILD).set_value("C009").run()
    assert at.selectbox(key="f2_activity_C009").value == c009_dates[-1]
    navigate_dashboard(at, EMOTIONAL)
    assert at.selectbox(key=selection.F1_DEMO_CHILD).value == "C001"
    at.toggle(key=selection.DEMO_WIDGET).set_value(False).run()
    assert selector(at).value == second.id
    assert_no_assessments(at, "Monitoring data has not been connected")

    csv_path = tmp_path / "selection-fixture.csv"
    csv_date = date(2030, 1, 1)
    daily.loc[daily["child_id"].eq("C009")].head(1).assign(
        child_id="csv-only-child", date=csv_date,
    ).to_csv(csv_path, index=False)
    monkeypatch.setenv("C4_F2_DAILY_CSV", str(csv_path))
    navigate_dashboard(at, BEHAVIORAL)
    assert_no_assessments(at, "Monitoring data has not been connected")
    at.toggle(key=selection.CSV_ASSESSMENT_WIDGET).set_value(True).run()
    csv_child = next(item for item in at.main.selectbox if item.label == "CSV child identifier")
    csv_day = next(item for item in at.main.selectbox if item.label == "Activity date")
    assert (csv_child.value, csv_day.value) == ("csv-only-child", csv_date)
    csv_result = at.session_state[selection._ENHANCED_RESULT][2]
    assert csv_result.rows["child_id"].tolist() == ["csv-only-child"]
    assert csv_result.rows["date"].tolist() == [csv_date]
    assert selection.F2_SAFETY_DATE not in at.session_state

    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    demo_child = at.selectbox(key=selection.F2_DEMO_CHILD)
    assert demo_child.value == "C001" and "csv-only-child" not in demo_child.options
    assert at.selectbox(key="f2_activity_C001").value == rows.loc[rows["child_id"].eq("C001"), "date"].max()
    assert str(csv_date) not in at.selectbox(key="f2_activity_C001").options
    demo_result = at.session_state[selection._ENHANCED_RESULT][2]
    assert demo_result is not csv_result
    assert set(demo_result.rows["child_id"]) == {"C001"}
    assert csv_date not in set(demo_result.rows["date"])
    assert csv_child.key not in at.session_state and csv_day.key not in at.session_state

    at.toggle(key=selection.DEMO_WIDGET).set_value(False).run()
    assert selector(at).value == second.id
    assert not at.toggle(key=selection.CSV_ASSESSMENT_WIDGET).value
    assert selection._ENHANCED_RESULT not in at.session_state
    assert_no_assessments(at, "Monitoring data has not been connected")

from datetime import timedelta
from uuid import uuid4

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from test_dashboard_navigation import APP, main_text, profiles  # noqa: F401
from test_partial_dashboard_monitoring import CHILD, END, records  # noqa: F401
from test_safety_notifications import isolated_app_database  # noqa: F401
from ui import child_selection as selection, safety_summary as safety, services


SINGLE_CHILD = "fictional-single"
SINGLE_DATE = END - timedelta(days=40)
OVERLAPPING_CHILD = "fictional-overlap"


@pytest.fixture
def source(records, monkeypatch):
    historical = END - timedelta(days=6)
    records.loc[records["date"].eq(historical), "language_urgent_flags"] = 1.0
    single = records.iloc[[0]].assign(child_id=SINGLE_CHILD, date=SINGLE_DATE)
    overlap = records.loc[records["date"].isin([historical, END])].assign(child_id=OVERLAPPING_CHILD)
    source = {"source": "demo", "child_ids": [CHILD, SINGLE_CHILD, OVERLAPPING_CHILD],
              "daily": pd.concat([records, single, overlap], ignore_index=True)}
    monkeypatch.setattr(services, "behavioral_daily_source", lambda: source)
    return source


@pytest.mark.usefixtures("isolated_app_database")
def test_date_controls_share_cutoff_and_reuse_assessment(source, profiles, monkeypatch, navigate_dashboard):
    calls = []
    assess = services.enhanced_behavioral_assessment

    def capture(rows, child, through, artifact):
        calls.append((child, through))
        return assess(rows, child, through, artifact)

    monkeypatch.setattr(services, "enhanced_behavioral_assessment", capture)
    at = AppTest.from_file(APP, default_timeout=35).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    at.selectbox(key="overview-demo-emotional").set_value("C004").run()
    at.button(key="overview-run-emotional").click().run()
    emotional = at.session_state[selection.OVERVIEW_DEMO_RESULTS]["emotional"]

    navigate_dashboard(at, "Behavioral Anomaly")
    detail_key = f"f2_activity_{CHILD}"
    safety_key = f"safety-assessment-date-{CHILD}"
    activity_key = f"activity-assessment-date-{CHILD}"
    assert at.selectbox(key=detail_key).value == END
    historical = END - timedelta(days=6)
    at.selectbox(key=detail_key).set_value(historical).run()
    at.run()
    assert at.selectbox(key=detail_key).value == historical
    assert calls == [(CHILD, END), (CHILD, historical)]
    assert at.session_state[selection.OVERVIEW_DEMO_RESULTS]["emotional"] == emotional

    navigate_dashboard(at, "Overview")
    at.button(key="overview-run-behavioral").click().run()
    assert f"Assessment date: {historical}" in main_text(at)
    assert f"Most recent urgent safety concern: {historical}" in main_text(at)
    assert "Recorded language flags: 99 across 2 of 7 days." in main_text(at)

    navigate_dashboard(at, "Safety Notifications")
    assert at.selectbox(key=safety_key).value == historical
    at.selectbox(key="safety-history-category").set_value(safety.URGENT).run()
    assert [item.label for item in at.get("status")] == [f"{historical} · {safety.URGENT}"]
    navigate_dashboard(at, "Activity Summary")
    assert at.selectbox(key=activity_key).value == historical
    for label, days, total, usable in [("Last 7 days", 7, 99, 2), ("Last 30 days", 30, 101, 3)]:
        at.radio(key="activity-period-widget").set_value(label).run()
        start = historical - timedelta(days=days - 1)
        assert f"{start} to {historical}" in main_text(at)
        assert at.metric[0].value == str(total)
        assert f"{total / usable:.2f} per day with a usable count ({usable} days)" in main_text(at)
        assert at.metric[2].value == "1"
    assert len(calls) == 2

    second = END - timedelta(days=2)
    at.selectbox(key=activity_key).set_value(second).run()
    navigate_dashboard(at, "Behavioral Anomaly")
    assert at.selectbox(key=detail_key).value == second
    navigate_dashboard(at, "Safety Notifications")
    assert at.selectbox(key=safety_key).value == second
    at.selectbox(key=safety_key).set_value(historical).run()
    navigate_dashboard(at, "Behavioral Anomaly")
    assert at.selectbox(key=detail_key).value == historical
    assert calls[-2:] == [(CHILD, second), (CHILD, historical)]
    assert not at.exception


@pytest.mark.usefixtures("isolated_app_database")
def test_date_validation_child_changes_source_changes_and_demo_isolation(source, profiles, navigate_dashboard):
    local = profiles.create("Date isolation fixture", submission_token=uuid4())
    at = AppTest.from_file(APP, default_timeout=35).run()
    assert at.toggle(key=selection.DEMO_WIDGET).value is False
    at.selectbox(key=selection.PROFILE_WIDGET).set_value(local.id).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    navigate_dashboard(at, "Behavioral Anomaly")
    key = f"f2_activity_{CHILD}"
    historical = END - timedelta(days=6)
    absent = END - timedelta(days=3)
    assert str(absent) not in at.selectbox(key=key).options
    # The browser can submit a child change and a value from the outgoing
    # date widget in one rerun. The new child's own dates must win.
    at.selectbox(key=selection.F2_DEMO_CHILD).set_value(OVERLAPPING_CHILD)
    at.selectbox(key=key).set_value(historical).run()
    assert at.selectbox(key=f"f2_activity_{OVERLAPPING_CHILD}").value == END
    at.selectbox(key=selection.F2_DEMO_CHILD).set_value(CHILD).run()
    at.selectbox(key=key).set_value(historical).run()
    at.selectbox(key=selection.F2_DEMO_CHILD).set_value(SINGLE_CHILD).run()
    assert at.selectbox(key=f"f2_activity_{SINGLE_CHILD}").value == SINGLE_DATE
    at.selectbox(key=selection.F2_DEMO_CHILD).set_value(CHILD).run()
    assert at.selectbox(key=key).value == END
    at.selectbox(key=key).set_value(historical).run()
    source["daily"] = source["daily"].loc[source["daily"]["date"].ne(historical)]
    navigate_dashboard(at, "Overview")
    at.button(key="overview-run-behavioral").click().run()
    assert f"Assessment date: {END}" in main_text(at)
    assert at.session_state[selection.F2_SAFETY_DATE] == END
    at.session_state[selection.F2_SAFETY_DATE] = pd.NA
    navigate_dashboard(at, "Behavioral Anomaly")
    assert at.selectbox(key=key).value == END

    rows = source["daily"]
    source.update(daily=rows.iloc[:0], child_ids=[])
    at.run()
    assert not at.main.selectbox and "No fictional monitoring records are available" in main_text(at)
    assert "The AI detected" not in main_text(at) and not at.exception
    at.toggle(key=selection.DEMO_WIDGET).set_value(False).run()
    assert selection.F2_SAFETY_DATE not in at.session_state
    assert "Monitoring data has not been connected" in main_text(at)
    source.update(daily=rows, child_ids=[CHILD, SINGLE_CHILD])
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    assert at.selectbox(key=key).value == END
    assert not at.exception

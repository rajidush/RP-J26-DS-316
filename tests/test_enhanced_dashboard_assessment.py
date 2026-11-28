from datetime import date, timedelta

import joblib
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from component4_profiling_xai.src import behavioral_profile
from ui import child_selection as selection, safety_summary as safety, services
from test_dashboard_navigation import APP, main_text, profiles  # noqa: F401
from test_safety_notifications import isolated_app_database  # noqa: F401


@pytest.mark.usefixtures("isolated_app_database")
@pytest.mark.parametrize(("child", "day"), [
    ("C003", date(2026, 8, 21)),
    ("C004", date(2026, 9, 1)),
    ("C004", date(2026, 9, 8)),
])
def test_enhanced_classifications_agree_across_dashboard(child, day, profiles, monkeypatch, navigate_dashboard):
    source = services.behavioral_daily_source()
    rows = services._behavioral_child_rows(source["daily"], child)
    bundle = joblib.load(behavioral_profile.BUNDLE)
    config = behavioral_profile.ProfileConfig(**bundle["profile"])
    original_matrix, original_details = behavioral_profile.profiles(rows, config)
    original_history = behavioral_profile.score_history(rows, original_matrix, bundle)
    expected = original_history.loc[original_history["date"].le(day)]
    flagged = bool(expected.iloc[-1]["flag"])
    calls = []
    assess = services.enhanced_behavioral_assessment

    def capture(daily, selected_child, through, contents):
        result = assess(daily, selected_child, through, contents)
        if (selected_child, through) == (child, day):
            calls.append(result)
            pd.testing.assert_frame_equal(result.history, expected)
            pd.testing.assert_frame_equal(result.matrix, original_matrix.loc[expected.index])
            pd.testing.assert_frame_equal(result.details, original_details.loc[expected.index])
        return result

    def forbid_legacy(*args, **kwargs):
        pytest.fail("Default dashboard used the legacy model")

    monkeypatch.setattr(services, "enhanced_behavioral_assessment", capture)
    monkeypatch.setattr(services, "behavioral_report", forbid_legacy)
    at = AppTest.from_file(APP, default_timeout=45).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    navigate_dashboard(at, "Behavioral Anomaly")
    at.selectbox(key=selection.F2_DEMO_CHILD).set_value(child).run()
    at.selectbox(key=f"f2_activity_{child}").set_value(day).run()
    assert not at.exception
    assert ("detected an unusual pattern" if flagged else "detected no unusual pattern") in main_text(at)
    assert len(calls) == 1

    navigate_dashboard(at, "Safety Notifications")
    at.selectbox(key=f"safety-assessment-date-{child}").set_value(day).run()
    at.selectbox(key="safety-history-category").set_value(safety.BEHAVIORAL).run()
    assert not at.exception
    assert (f"{day} · {safety.BEHAVIORAL}" in [event.label for event in at.get("status")]) == flagged
    assert len(calls) == 2  # The initial Safety date differed, replacing the single cached context.

    navigate_dashboard(at, "Overview")
    at.button(key="overview-run-behavioral").click().run()
    wording = "was flagged" if flagged else "was not flagged"
    assert f"The latest recorded day {wording} for unusual activity." in main_text(at)
    assert f"Assessment date: {day}" in main_text(at)

    navigate_dashboard(at, "Activity Summary")
    for period, days in (("Last 7 days", 7), ("Last 30 days", 30)):
        at.radio(key="activity-period-widget").set_value(period).run()
        window = expected["date"].between(day - timedelta(days=days - 1), day)
        metric = next(item for item in at.metric if item.label == "Days with unusual activity")
        assert metric.value == str(expected.loc[window, "flag"].sum())
    assert not at.exception and len(calls) == 2

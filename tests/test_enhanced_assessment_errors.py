import pytest
from streamlit.testing.v1 import AppTest

from component4_profiling_xai.src import behavioral_profile
from test_dashboard_navigation import APP, main_text, profiles  # noqa: F401
from test_partial_dashboard_monitoring import CHILD, END, records  # noqa: F401
from test_safety_notifications import isolated_app_database  # noqa: F401
from ui import child_selection as selection, safety_summary as safety, services
from ui.legacy_pages import _behavioral_upgrade as view


SECRET = "private path E:/private/story.csv"
DAILY_UNAVAILABLE = "An AI explanation is unavailable for this day."
TEMPORAL_UNAVAILABLE = "An AI explanation across days is unavailable."


def fail(*args, **kwargs):
    raise ValueError(SECRET)


def assert_parent_safe(at):
    assert not at.exception
    assert SECRET not in main_text(at)
    assert "Traceback" not in main_text(at)


@pytest.fixture
def source(records, monkeypatch):
    rows = records.tail(1).assign(screen_monitor_observed=1.0)
    source = {"source": "demo", "daily": rows, "child_ids": [CHILD]}
    monkeypatch.setattr(services, "behavioral_daily_source", lambda: source)
    return source


def prediction_message(source):
    result = services.enhanced_behavioral_assessment(
        source["daily"], CHILD, END, services.enhanced_bundle_bytes(),
    )
    flag = result.history["flag"].iloc[-1]
    assert result.history["flag"].notna().iloc[-1]
    return "The AI detected an unusual pattern" if flag else "The AI detected no unusual pattern"


@pytest.mark.usefixtures("isolated_app_database")
def test_scoring_failure_drops_stale_assessment_and_recovers(source, profiles, monkeypatch, navigate_dashboard, caplog):
    initial_prediction = prediction_message(source)
    at = AppTest.from_file(APP, default_timeout=35).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    navigate_dashboard(at, "Behavioral Anomaly")
    assert initial_prediction in main_text(at)

    source["daily"].loc[:, "language_flags"] = 5.0
    monkeypatch.setattr(services, "behavioral_report", lambda *args: pytest.fail("Unexpected legacy fallback"))
    with monkeypatch.context() as injected:
        injected.setattr(behavioral_profile, "score_history", fail)
        injected.setattr(view, "ExplanationRequest", lambda *args: pytest.fail("Unexpected XAI without a score"))
        at.run()
        assert_parent_safe(at)
        assert safety.BEHAVIOR_FAILED in main_text(at)
        assert "The AI detected" not in main_text(at) and DAILY_UNAVAILABLE in main_text(at)
        assert "Language that needs urgent review" in at.warning[0].value
        assert selection._ENHANCED_RESULT not in at.session_state

        navigate_dashboard(at, "Safety Notifications")
        assert_parent_safe(at)
        assert safety.BEHAVIOR_FAILED in main_text(at)
        assert [item.label for item in at.get("status")] == [f"{END} · {safety.URGENT}"]
        navigate_dashboard(at, "Overview")
        at.button(key="overview-run-behavioral").click().run()
        assert_parent_safe(at)
        assert safety.BEHAVIOR_FAILED in main_text(at)
        assert f"Most recent urgent safety concern: {END}" in main_text(at)
        navigate_dashboard(at, "Activity Summary")
        assert_parent_safe(at)
        assert safety.BEHAVIOR_FAILED in main_text(at)
        assert [metric.value for metric in at.metric] == ["5", "—", "1"]

    navigate_dashboard(at, "Behavioral Anomaly")
    assert_parent_safe(at)
    assert prediction_message(source) in main_text(at) and safety.BEHAVIOR_FAILED not in main_text(at)
    assert at.selectbox(key=f"f2_activity_{CHILD}").value == END
    assert "enhanced activity assessment (ValueError)" in caplog.text and SECRET not in caplog.text


@pytest.mark.usefixtures("isolated_app_database")
@pytest.mark.parametrize(("operation", "log_stage", "daily_lost", "temporal_lost"), [
    ("ExplanationRequest", "activity explanation setup", True, True),
    ("daily", "daily activity explanation", True, False),
    ("review_daily", "daily activity explanation", True, False),
    ("temporal", "activity explanation across days", False, True),
    ("assess_temporal", "activity explanation across days", False, True),
])
def test_explanation_failure_keeps_prediction_and_recovers(
    source, profiles, monkeypatch, navigate_dashboard, caplog, operation, log_stage, daily_lost, temporal_lost,
):
    expected_prediction = prediction_message(source)
    at = AppTest.from_file(APP, default_timeout=35).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    navigate_dashboard(at, "Behavioral Anomaly")
    assert_parent_safe(at)
    before = main_text(at)
    assert DAILY_UNAVAILABLE not in before and TEMPORAL_UNAVAILABLE not in before
    before_copy = [item.value for item in (*at.main.markdown, *at.main.caption)]
    cached = at.session_state[selection._ENHANCED_RESULT][2]

    with monkeypatch.context() as injected:
        owner = view if operation in {"ExplanationRequest", "assess_temporal"} else view.ExplanationRequest
        injected.setattr(owner, operation, fail)
        at.run()
        assert_parent_safe(at)
        content = main_text(at)
        assert expected_prediction in content
        assert at.selectbox(key=selection.F2_DEMO_CHILD).value == CHILD
        assert at.selectbox(key=f"f2_activity_{CHILD}").value == END
        assert (DAILY_UNAVAILABLE in content) == daily_lost
        assert (TEMPORAL_UNAVAILABLE in content) == temporal_lost
        assert "The comparison uses your child’s own earlier records." in content
        assert "Language that needs urgent review" in at.warning[0].value
        assert at.session_state[selection._ENHANCED_RESULT][2] is cached

    at.run()
    assert_parent_safe(at)
    assert [item.value for item in (*at.main.markdown, *at.main.caption)] == before_copy
    assert at.session_state[selection._ENHANCED_RESULT][2] is cached
    assert f"{log_stage} (ValueError)" in caplog.text and SECRET not in caplog.text

from datetime import date

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from component4_profiling_xai.src import behavioral_profile, urgent_alerts
from test_dashboard_navigation import APP, main_text, profiles  # noqa: F401
from test_partial_dashboard_monitoring import CHILD, END, records  # noqa: F401
from test_safety_notifications import isolated_app_database  # noqa: F401
from ui import child_selection as selection, safety_summary as safety, services
from ui.legacy_pages import _behavioral_upgrade as view


def banners(at):
    return [item.value for item in at.warning if "Urgent safety review" in item.value]


def earlier_alerts(at):
    return [item.value for group in at.get("expander") if group.label == "Earlier safety alerts"
            for item in group.markdown]


@pytest.mark.usefixtures("isolated_app_database")
def test_historical_safety_follows_shared_cutoff(profiles, navigate_dashboard):
    source = services.behavioral_daily_source()
    child = "C003"
    rows = source["daily"].loc[source["daily"]["child_id"].eq(child)]
    latest = rows["date"].max()
    key = f"f2_activity_{child}"
    dates = [date(2026, 9, 21), date(2026, 8, 21), latest, date(2026, 9, 21)]
    rules = urgent_alerts.urgent_alerts(rows)
    hits = rules.loc[rules["is_urgent"]]

    at = AppTest.from_file(APP, default_timeout=45).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    navigate_dashboard(at, "Behavioral Anomaly")
    at.selectbox(key=selection.F2_DEMO_CHILD).set_value(child).run()
    for day in dates:
        at.selectbox(key=key).set_value(day).run()
        assert not at.exception
        assessment = services.enhanced_behavioral_assessment(rows, child, day, services.enhanced_bundle_bytes())
        flag = bool(assessment.history["flag"].iloc[-1])
        prediction = "detected an unusual pattern" if flag else "detected no unusual pattern"
        assert prediction in main_text(at)
        current = hits.loc[hits["date"].eq(day)]
        assert len(banners(at)) == len(current)
        for banner, hit in zip(banners(at), current.itertuples(), strict=True):
            assert f"Selected day ({day})" in banner and view.safety_text(hit.reasons) in banner
        guidance = "Review the safety alert above and seek appropriate human support."
        assert (guidance in main_text(at)) == (not current.empty)
        earlier = hits.loc[hits["date"].lt(day)].sort_values("date", ascending=False)
        assert earlier_alerts(at) == [
            f"**{hit.date}** · {view.safety_text(hit.reasons)}" for hit in earlier.itertuples()
        ]

    day = dates[-1]
    most_recent = hits.loc[hits["date"].le(day), "date"].max()
    navigate_dashboard(at, "Overview")
    at.button(key="overview-run-behavioral").click().run()
    assert f"Assessment date: {day}" in main_text(at)
    assert f"Most recent urgent safety concern: {most_recent}" in main_text(at)
    navigate_dashboard(at, "Safety Notifications")
    assert at.selectbox(key=f"safety-assessment-date-{child}").value == day
    assert all(date.fromisoformat(item.label.split(" · ")[0]) <= day for item in at.get("status"))
    navigate_dashboard(at, "Activity Summary")
    assert at.selectbox(key=f"activity-assessment-date-{child}").value == day
    navigate_dashboard(at, "Behavioral Anomaly")
    assert at.selectbox(key=key).value == day and not banners(at)
    at.selectbox(key=selection.F2_DEMO_CHILD).set_value("C001").run()
    other_rows = source["daily"].loc[source["daily"]["child_id"].eq("C001")]
    other_rules = urgent_alerts.urgent_alerts(other_rows)
    assert len(banners(at)) == other_rules.loc[other_rules["date"].eq(other_rows["date"].max()), "is_urgent"].sum()


@pytest.mark.usefixtures("isolated_app_database")
def test_later_urgent_record_cannot_change_historical_presentation(records, profiles, monkeypatch, navigate_dashboard):
    selected = records.iloc[[2]].assign(language_urgent_flags=1.0)
    day = selected["date"].iloc[0]
    source = {"source": "demo", "child_ids": [CHILD], "daily": selected}
    monkeypatch.setattr(services, "behavioral_daily_source", lambda: source)
    at = AppTest.from_file(APP, default_timeout=35).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    navigate_dashboard(at, "Behavioral Anomaly")
    assert banners(at) and not at.exception

    def snapshot():
        return [item.value for item in (*at.main.markdown, *at.main.caption, *at.warning)]

    before = snapshot()
    history = at.session_state[selection._ENHANCED_RESULT][2].history.copy()
    later = records.tail(1).assign(language_grooming_flags=1.0, screen_monitor_observed=0.0)
    source["daily"] = pd.concat([selected, later], ignore_index=True)
    at.run()
    assert at.selectbox(key=f"f2_activity_{CHILD}").value == day
    assert not at.exception and snapshot() == before
    pd.testing.assert_frame_equal(at.session_state[selection._ENHANCED_RESULT][2].history, history)
    at.selectbox(key=f"f2_activity_{CHILD}").set_value(END).run()
    assert any(str(END) in banner and "grooming" in banner for banner in banners(at))


@pytest.mark.usefixtures("isolated_app_database")
def test_distinct_urgent_matches_survive_on_the_same_day(records, profiles, monkeypatch, navigate_dashboard):
    rows = records.tail(1)
    monkeypatch.setattr(services, "behavioral_daily_source",
                        lambda: {"source": "demo", "child_ids": [CHILD], "daily": rows})
    review = services.behavioral_urgent_review

    def with_second_match(daily):
        alerts, available, complete = review(daily)
        return pd.concat([alerts, alerts.assign(reasons="grooming language flag (1)")]), available, complete

    monkeypatch.setattr(services, "behavioral_urgent_review", with_second_match)
    at = AppTest.from_file(APP, default_timeout=35).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    navigate_dashboard(at, "Behavioral Anomaly")
    assert not at.exception and len(banners(at)) == 2
    assert "urgent review" in banners(at)[0] and "grooming" in banners(at)[1]
    assert "couldn’t assess this day" in main_text(at)


@pytest.mark.usefixtures("isolated_app_database")
@pytest.mark.parametrize("condition", ["screen", "language", "required_input", "bundle", "scoring", "explanation"])
def test_selected_day_safety_survives_degraded_assessment(
    condition, records, profiles, monkeypatch, navigate_dashboard,
):
    def fail(*args):
        raise ValueError("injected failure")

    rows = records.tail(1).assign(screen_monitor_observed=1.0)
    if condition == "screen":
        rows = rows.assign(screen_monitor_observed=0.0)
    elif condition == "language":
        rows = rows.assign(language_monitor_observed=0.0, language_urgent_flags=0.0,
                           screen_self_harm_imagery_incidents=1.0)
    elif condition == "required_input":
        rows = rows.assign(screen_flagged_seconds=float("nan"))
    elif condition == "bundle":
        monkeypatch.setattr(services, "enhanced_bundle_bytes", lambda: None)
    elif condition == "scoring":
        monkeypatch.setattr(behavioral_profile, "score_history", fail)
    else:
        monkeypatch.setattr(view.ExplanationRequest, "daily", fail)

    expected = urgent_alerts.urgent_alerts(rows).iloc[-1]
    assert expected["is_urgent"]
    monkeypatch.setattr(services, "behavioral_daily_source",
                        lambda: {"source": "demo", "child_ids": [CHILD], "daily": rows})
    at = AppTest.from_file(APP, default_timeout=35).run()
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    navigate_dashboard(at, "Behavioral Anomaly")
    assert not at.exception and len(banners(at)) == 1
    assert view.safety_text(expected["reasons"]) in banners(at)[0]
    assert "Review the safety alert above and seek appropriate human support." in main_text(at)
    if condition in {"screen", "language", "required_input"}:
        assert "couldn’t assess this day" in main_text(at)
    elif condition == "bundle":
        assert safety.MODEL_UNAVAILABLE in main_text(at)
    elif condition == "scoring":
        assert safety.BEHAVIOR_FAILED in main_text(at)
    else:
        assert "The AI detected" in main_text(at)
        assert "An AI explanation is unavailable for this day." in main_text(at)

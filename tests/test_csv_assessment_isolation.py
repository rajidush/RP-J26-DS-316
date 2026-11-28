from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest
from streamlit.testing.v1 import AppTest

from test_dashboard_navigation import APP, main_text, profiles  # noqa: F401
from test_safety_notifications import isolated_app_database  # noqa: F401
from ui import child_selection as selection, services


EARLIER = date(2030, 1, 1)
LATEST = date(2030, 1, 2)


@pytest.fixture
def external_csv(tmp_path):
    path = tmp_path / "private-fixture.csv"
    daily = services.behavioral_daily_source()["daily"].query("child_id == 'C001'").head(2).copy()
    daily["date"] = [EARLIER, LATEST]
    daily["screen_monitor_observed"] = [0.0, 1.0]
    daily[list(services._urgent_module().URGENT_RULE_COLUMNS)] = 0.0
    daily["language_urgent_flags"] = [1.0, 0.0]
    daily["language_grooming_flags"] = [0.0, 2.0]
    daily.to_csv(path, index=False)
    return path, daily


def open_behavioral(navigate_dashboard):
    at = AppTest.from_file(APP, default_timeout=40).run()
    navigate_dashboard(at, "Behavioral Anomaly")
    return at


@pytest.mark.usefixtures("isolated_app_database")
@pytest.mark.parametrize(("config", "legacy"), [
    ("unset", False), ("valid", False), ("missing", False), ("malformed", False),
    ("unset", True), ("valid", True), ("missing", True), ("malformed", True),
])
def test_demo_uses_only_bundled_fiction(config, legacy, external_csv, profiles, monkeypatch, navigate_dashboard):
    path, _ = external_csv
    monkeypatch.setenv("C4_F2_LEGACY_VIEW", "1" if legacy else "0")
    if config == "unset":
        monkeypatch.delenv("C4_F2_DAILY_CSV", raising=False)
    else:
        monkeypatch.setenv("C4_F2_DAILY_CSV", str(path))
        if config == "missing":
            path.unlink()
        elif config == "malformed":
            path.write_text("bad,columns\n1,2\n")
    at = open_behavioral(navigate_dashboard)
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    assert not at.exception and not at.main.error
    assert "Fictional data for this assessment" in main_text(at)
    assert "Behavior Overview" in main_text(at)
    assert "Latest Prediction" not in main_text(at)
    assert str(LATEST) not in main_text(at)
    assert at.selectbox(key=selection.F2_DEMO_CHILD).options == services.behavioral_daily_source()["child_ids"]
    assert selection.CSV_ASSESSMENT_WIDGET not in [item.key for item in at.main.toggle]


@pytest.mark.usefixtures("isolated_app_database")
def test_csv_assessments_are_opt_in_source_keyed_and_cut_off(
    external_csv, profiles, monkeypatch, navigate_dashboard,
):
    path, daily = external_csv
    monkeypatch.setenv("C4_F2_DAILY_CSV", str(path))
    monkeypatch.setenv("C4_F2_LEGACY_VIEW", "1")
    profile = profiles.create("C001", submission_token=uuid4())
    at = AppTest.from_file(APP, default_timeout=40).run()
    at.selectbox(key=selection.PROFILE_WIDGET).set_value(profile.id).run()
    navigate_dashboard(at, "Behavioral Anomaly")
    assert "Monitoring data has not been connected" in main_text(at)
    assert not at.main.selectbox
    at.toggle(key=selection.CSV_ASSESSMENT_WIDGET).set_value(True).run()
    assert not at.exception
    assert "Local CSV assessment" in main_text(at)
    assert "Behavior Overview" in main_text(at)
    assert "Demo Mode · Fictional data" not in main_text(at)
    assert any(box.label == "CSV child identifier" and box.value == "C001" for box in at.main.selectbox)
    original = at.session_state[selection._ENHANCED_RESULT][2]
    at.run()
    assert at.session_state[selection._ENHANCED_RESULT][2] is original
    date_picker = next(box for box in at.main.selectbox if box.label == "Activity date")
    date_picker.set_value(EARLIER).run()
    alerts = " ".join(w.value for w in at.warning)
    assert "couldn’t assess this day" in main_text(at)
    assert str(EARLIER) in alerts and str(LATEST) not in alerts
    assert "Language that needs urgent review" in alerts
    original = at.session_state[selection._ENHANCED_RESULT][2]

    # Even an unused input column changes the file revision at the same path.
    daily.assign(unused="changed bytes").to_csv(path, index=False)
    at.run()
    updated = at.session_state[selection._ENHANCED_RESULT][2]
    assert updated is not original and updated.rows["date"].max() == EARLIER
    assert not any(str(LATEST) in w.value for w in at.warning)
    next(box for box in at.main.selectbox if box.label == "Activity date").set_value(LATEST).run()
    assert any(str(LATEST) in w.value and "grooming" in w.value for w in at.warning)
    at.toggle(key=selection.DEMO_WIDGET).set_value(True).run()
    assert "Local CSV assessment" not in main_text(at)
    assert not any(str(LATEST) in w.value for w in at.warning)
    demo_context, _, demo_result = at.session_state[selection._ENHANCED_RESULT]
    assert demo_context[0] == "demo"
    assert demo_result is not updated and LATEST not in set(demo_result.rows["date"])
    navigate_dashboard(at, "Safety Notifications")
    assert "Fictional" in main_text(at) and str(LATEST) not in main_text(at)
    navigate_dashboard(at, "Activity Summary")
    assert str(LATEST) not in main_text(at)
    navigate_dashboard(at, "Overview")
    assert str(LATEST) not in main_text(at)
    navigate_dashboard(at, "Behavioral Anomaly")
    at.toggle(key=selection.DEMO_WIDGET).set_value(False).run()
    assert not at.toggle(key=selection.CSV_ASSESSMENT_WIDGET).value
    assert "Monitoring data has not been connected" in main_text(at)


@pytest.mark.usefixtures("isolated_app_database")
@pytest.mark.parametrize("kind", ["missing", "unreadable", "empty", "headers", "malformed"])
def test_external_source_errors_do_not_fall_back(
    kind, external_csv, profiles, monkeypatch, navigate_dashboard,
):
    path, daily = external_csv
    monkeypatch.setenv("C4_F2_DAILY_CSV", str(path))
    at = open_behavioral(navigate_dashboard)
    at.toggle(key=selection.CSV_ASSESSMENT_WIDGET).set_value(True).run()
    assert selection._ENHANCED_RESULT in at.session_state
    if kind == "missing":
        path.unlink()
    elif kind == "unreadable":
        read = Path.read_bytes

        def denied(file):
            if file == path:
                raise PermissionError("private location")
            return read(file)

        monkeypatch.setattr(Path, "read_bytes", denied)
    elif kind == "empty":
        path.write_text("")
    elif kind == "headers":
        daily.iloc[:0].to_csv(path, index=False)
    else:
        path.write_text("bad,columns\n1,2\n")
    at.run()
    expected = {
        "missing": "could not be opened", "unreadable": "could not be opened",
        "empty": "contains no assessment records", "headers": "contains no assessment records",
        "malformed": "contains invalid assessment records",
    }
    assert any(expected[kind] in item.value for item in (*at.main.error, *at.main.info))
    assert not at.exception and not at.main.selectbox
    assert str(path) not in main_text(at) and "private location" not in main_text(at)
    assert selection._ENHANCED_RESULT not in at.session_state

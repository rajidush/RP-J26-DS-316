"""Render the real Streamlit pages (streamlit.testing AppTest) and check what
a viewer would see when the trained models load or fail to load."""
import pytest
from streamlit.testing.v1 import AppTest

from ui import services

APP = str(services.ROOT / "ui" / "app.py")
F1_ERROR = "Function 1 model could not be loaded. No emotional pattern is shown."


def html_of(at, area):
    return " ".join(str(h.proto) for h in area.get("html"))


def run_page(page=None):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    if page:
        at.switch_page(f"pages/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.fixture
def child_with_sessions(temp_db):
    for i, emo in enumerate(["calm", "calm", "defensive", "calm", "calm"]):
        services.save_record({
            "child_id": "K1", "session_id": f"K1-{i}", "timestamp": f"2026-07-0{i + 1}T10:00:00Z",
            "risk_level": "low", "emotional_state": emo, "self_regulation_shown": False,
            "escalation_flag": False, "dialogue_turns": 2})


def test_emotional_pattern_page_with_real_model(child_with_sessions):
    at = run_page()
    main, sidebar = html_of(at, at.main), html_of(at, at.sidebar)
    assert "c4-badge" in main and "Calm" in main
    assert at.get("plotly_chart")
    assert "Why this pattern?" in [e.label for e in at.expander]
    assert F1_ERROR not in main
    assert "Function 1 model" in sidebar and "not loaded" not in sidebar


def test_emotional_pattern_page_missing_model_shows_only_the_error(child_with_sessions,
                                                                  tmp_path, monkeypatch):
    monkeypatch.setenv("C4_F1_ARTIFACT_DIR", str(tmp_path))  # empty folder
    at = run_page()
    main, sidebar = html_of(at, at.main), html_of(at, at.sidebar)
    assert F1_ERROR in main
    assert "emotional_pattern_kmeans.pkl" in main          # the reason names the files
    assert "scikit-learn==1.6.1" in main                   # and says how to fix it
    assert "c4-badge" not in main                          # no badge
    assert not at.get("plotly_chart")                      # no chart
    assert "Why this pattern?" not in [e.label for e in at.expander]  # no features
    assert not at.selectbox                                # nothing to select
    assert "not loaded" in sidebar


def test_emotional_pattern_page_unloadable_model(child_with_sessions, tmp_path, monkeypatch):
    for name in services.F1_ARTIFACT_FILES:
        (tmp_path / name).write_text("garbage")
    monkeypatch.setenv("C4_F1_ARTIFACT_DIR", str(tmp_path))
    at = run_page()
    main = html_of(at, at.main)
    assert F1_ERROR in main and "Could not load the model files" in main
    assert "c4-badge" not in main


# --- Function 2: Behavioral Anomaly page ----------------------------------------------

F2_ERROR = "Function 2 model could not be loaded. No anomaly prediction is shown."
F2_URGENT_TITLE = "Urgent Safety Alerts (rule-based, not the anomaly model)"


def test_emotional_pattern_is_still_the_default_page(temp_db):
    at = run_page()
    main = html_of(at, at.main)
    assert "Emotional Pattern (Function 1)" in main
    assert "Behavioral Anomaly (Function 2)" not in main


def test_sidebar_shows_both_model_statuses(temp_db):
    sidebar = html_of(at := run_page(), at.sidebar)
    assert "Function 1 model" in sidebar and "Function 2 model" in sidebar
    assert "not loaded" not in sidebar


def test_behavioral_anomaly_page_with_real_model(temp_db):
    at = run_page("behavioral_anomaly")
    main = html_of(at, at.main)
    assert "SYNTHETIC DATA" in main and "DEMO PREDICTIONS" in main
    assert at.selectbox(key="f2_child").options[0] == "C001"
    assert len(at.selectbox(key="f2_child").options) == 40
    assert "c4-badge" in main
    assert "It does not prove that the child is safe" in main
    assert len(at.get("plotly_chart")) == 1
    assert len(at.dataframe) >= 3                         # daily, weekly, urgent / explanation
    assert F2_URGENT_TITLE in main
    assert "not an exact cause" in main
    labels = [e.label for e in at.expander]
    assert "About this model" in labels and "Use your own data (daily-feature CSV)" in labels
    assert "One-Class SVM" in " ".join(m.value for m in at.markdown)
    assert F2_ERROR not in main


def test_behavioral_anomaly_page_switches_child(temp_db):
    at = run_page("behavioral_anomaly")
    at.selectbox(key="f2_child").set_value("C009").run()
    assert not at.exception
    assert at.selectbox(key="f2_day_C009").value is not None   # C009 has unusual days


def test_behavioral_anomaly_page_missing_model_shows_only_the_error(temp_db, tmp_path, monkeypatch):
    monkeypatch.setenv("C4_F2_ARTIFACT_DIR", str(tmp_path))   # empty folder
    at = run_page("behavioral_anomaly")
    main = html_of(at, at.main)
    assert F2_ERROR in main
    assert "behavioral_anomaly_pipeline.pkl" in main and "scikit-learn==1.6.1" in main
    assert "c4-badge" not in main
    assert not at.get("plotly_chart")
    assert not at.selectbox and not at.dataframe
    assert F2_URGENT_TITLE not in main


def test_function2_failure_leaves_function1_page_working(child_with_sessions, tmp_path, monkeypatch):
    monkeypatch.setenv("C4_F2_ARTIFACT_DIR", str(tmp_path))
    at = run_page()
    main, sidebar = html_of(at, at.main), html_of(at, at.sidebar)
    assert "c4-badge" in main and F1_ERROR not in main
    assert "Function 2 model" in sidebar and "not loaded" in sidebar


@pytest.mark.parametrize("page", ["emotional_pattern", "behavioral_anomaly", "child_profile",
                                  "dashboard", "sessions", "escalations", "add_record"])
def test_every_page_renders(temp_db, page):
    run_page(page)

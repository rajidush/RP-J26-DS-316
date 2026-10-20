"""Render the real Streamlit pages (streamlit.testing AppTest) and check what
a viewer would see when the trained models load or fail to load."""
import pytest
from streamlit.testing.v1 import AppTest

from ui import services

APP = str(services.ROOT / "ui" / "app.py")
F1_ERROR = "Function 1 model could not be loaded. No emotional pattern is shown."
F2_ERROR = "Function 2 model could not be loaded. No behavioral anomaly result is shown."


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


def test_emotional_pattern_page_has_no_upload_label_or_contract_notice(child_with_sessions):
    at = run_page()
    main = html_of(at, at.main)
    assert "Emotional Pattern" in main and "(Function 1)" not in main
    assert "Use your own data (CSV with child_id)" not in [e.label for e in at.expander]
    assert not at.get("file_uploader")
    assert "pending a contract update" not in main
    assert "from the database" not in main                 # no child-count note
    assert at.selectbox[0].value == "K1"                   # database children still loaded


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


def test_behavioral_anomaly_page_with_real_model(temp_db):
    at = run_page("behavioral_anomaly")
    at.button[0].click().run()
    main = html_of(at, at.main)
    assert "c4-badge" in main and F2_ERROR not in main


def test_behavioral_anomaly_page_missing_model_shows_only_the_error(temp_db, monkeypatch):
    def missing():
        raise FileNotFoundError("Behavioral anomaly artifact not found: x.pkl")

    monkeypatch.setattr(services._f2_module(), "load_artifacts", missing)
    at = run_page("behavioral_anomaly")
    main = html_of(at, at.main)
    assert F2_ERROR in main and "x.pkl" in main
    assert "c4-badge" not in main
    assert not at.button  # no form to submit

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
F2_URGENT_TITLE = "Urgent Alerts (rule-based)"
F2_ABOUT = "About the data and model"
F2_BANNED = ("Normal", "Unusual", "Threshold", "Synthetic Data")   # never visible on the main page


def visible_main_text(at):
    """What a sighted viewer reads on the main page, outside the About expander:
    HTML text (screen-reader-only text and tag attributes removed), Markdown and
    metric labels."""
    import re
    html = html_of(at, at.main)
    html = re.sub(r'<p class=\\?"c4-sr-only\\?">.*?</p>', " ", html)
    text = re.sub(r"<[^>]+>", " ", html)
    about = next(e for e in at.expander if e.label == F2_ABOUT)
    about_md = {m.value for m in about.markdown}
    text += " ".join(m.value for m in at.markdown if m.value not in about_md)
    text += " ".join(m.label for m in at.metric)
    return text


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
    assert at.selectbox(key="f2_child").options[0] == "C001"
    assert len(at.selectbox(key="f2_child").options) == 40
    assert "c4-flag" in main and "c4-badge" not in main
    assert len(at.get("plotly_chart")) == 1
    assert len(at.dataframe) >= 3                         # daily, weekly, urgent / details
    assert F2_URGENT_TITLE in main and "Weekly Summary" in main
    assert [m.label for m in at.metric] == ["Days shown", "Flagged days"]
    labels = [e.label for e in at.expander]
    assert F2_ABOUT in labels and "Use your own data (daily-feature CSV)" in labels
    assert F2_ERROR not in main


def test_main_page_has_no_status_words_threshold_or_synthetic_banner(temp_db):
    at = run_page("behavioral_anomaly")
    text = visible_main_text(at).lower()
    # The helper sees real visible text (so the check below is not vacuous) ...
    assert "flagged days" in text and "weekly summary" in text and "score" in text
    # ... but not the screen-reader-only chart description.
    assert "amber diamonds" not in text
    for word in F2_BANNED:
        assert word.lower() not in text, word
    weekly = next(d.value for d in at.dataframe if "week (Mon)" in d.value.columns)
    assert "flagged days" in weekly.columns


def test_status_icon_has_a_text_description(temp_db):
    import re
    at = run_page("behavioral_anomaly")                    # C001: latest day not flagged
    main = html_of(at, at.main)
    assert re.search(r'aria-label=\\?"2026-09-28: Not flagged: the model found no unusual pattern', main)
    assert re.search(r'alt=\\?"2026-09-28: Not flagged', main)
    assert "This is not proof that the child is safe" in main          # part of the description
    assert "c4-sr-only" in main and "amber diamonds" in main            # chart description


def test_flagged_latest_day_uses_the_flagged_icon(temp_db):
    daily = services.behavioral_daily_source()["daily"]
    flagged_last = next(c for c in services.child_ids(daily)
                        if services.behavioral_report(daily, c)["latest"]["unusual"])
    at = run_page("behavioral_anomaly")
    at.selectbox(key="f2_child").set_value(flagged_last).run()
    assert not at.exception
    assert "Flagged: the model found an unusual pattern" in html_of(at, at.main)


def test_about_section_discloses_synthetic_data_and_threshold(temp_db):
    at = run_page("behavioral_anomaly")
    about = " ".join(m.value for m in next(e for e in at.expander if e.label == F2_ABOUT).markdown)
    assert "synthetic demo data" in about and "no real child is involved" in about
    assert "Not flagged is not proof that a child is safe" in about
    assert "-12.5158" in about and "decision threshold" in about
    assert "not an exact cause" in about and "never merged" in about
    assert "One-Class SVM" in about


def test_behavioral_anomaly_page_switches_child(temp_db):
    at = run_page("behavioral_anomaly")
    at.selectbox(key="f2_child").set_value("C009").run()
    assert not at.exception
    assert at.selectbox(key="f2_day_C009").value is not None   # C009 has flagged days


def test_behavioral_anomaly_page_missing_model_shows_only_the_error(temp_db, tmp_path, monkeypatch):
    monkeypatch.setenv("C4_F2_ARTIFACT_DIR", str(tmp_path))   # empty folder
    at = run_page("behavioral_anomaly")
    main = html_of(at, at.main)
    assert F2_ERROR in main
    assert "behavioral_anomaly_pipeline.pkl" in main and "scikit-learn==1.6.1" in main
    assert "c4-flag" not in main
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

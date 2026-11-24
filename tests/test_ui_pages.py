"""Render the real Streamlit pages (streamlit.testing AppTest) and check what
a viewer would see when the trained models load or fail to load."""
import pytest
from streamlit.testing.v1 import AppTest

from ui import services

APP = str(services.ROOT / "ui" / "app.py")
F1_ERROR = "The emotional wellbeing summary is temporarily unavailable."
HISTORY = "View Emotional History"
AI_DETAILS = "Why Did the AI Identify This Pattern?"


def html_of(at, area):
    return " ".join(str(h.proto) for h in area.get("html"))


def run_page(page=None):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    if page:
        at.switch_page(f"pages/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def wellbeing_card(at):
    return next(container for container in at.get("flex_container")
                if container.proto.id.endswith("-card-emotional-wellbeing"))


def emotional_expanders(at):
    # AppTest classifies expandable blocks with icons as status blocks.
    return [block for block in at.main if block.type in ("expander", "status")]


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
    assert len(at.get("plotly_chart")) == 1
    assert [e.label for e in emotional_expanders(at)] == [HISTORY, AI_DETAILS]
    assert not at.info
    assert "Function 1 model" in sidebar and "not loaded" not in sidebar
    card = wellbeing_card(at)
    assert "Your Child&#x27;s Emotional Wellbeing" in html_of(at, card)
    lines = [item.value for item in card.markdown]
    assert [line for line in lines if line.startswith("#### ")] == [
        "#### Current Emotional Pattern", "#### Has Anything Changed?",
        "#### What Can You Do?", "#### Try Asking",
    ]
    assert "**Calm and reflective**" in lines
    assert "More recorded sessions are needed to tell whether the pattern has changed." in lines
    assert lines.count("“How have things felt online lately?”") == 1


def test_emotional_pattern_page_keeps_prediction_when_xai_is_missing(child_with_sessions,
                                                                      monkeypatch):
    monkeypatch.setattr(services._f1_module(), "explain_pattern", lambda *args: None)
    at = run_page()
    assert "Calm and reflective" in visible_main_text(at)
    assert "There isn’t enough clear supporting information to explain this pattern." in visible_main_text(at)
    assert "This describes how your child responded in the latest period" in visible_main_text(at)


def test_emotional_pattern_page_labels_synthetic_guidance(temp_db):
    at = run_page()
    at.selectbox[0].set_value("C004").run()
    assert not at.exception
    text = visible_main_text(at)
    assert "SYNTHETIC DEMO DATA" in text
    for removed in ("Parent Guidance", "Emotional Change Summary", "What Did We Notice?",
                    "Why Are We Suggesting This?", "Why Was This Change Detected?",
                    "What Pattern Did We Notice?", "Why Does the AI Think This?",
                    "What Does This Mean for You?"):
        assert removed not in text
    history, details = emotional_expanders(at)
    assert [history.label, details.label] == [HISTORY, AI_DETAILS]
    assert not history.proto.expanded and not details.proto.expanded
    assert len(history.get("plotly_chart")) == 1
    assert [item.value for item in history.caption] == [
        "The chart shows how the pattern varied across the recorded two-week periods.",
    ]
    assert not history.markdown
    assert [item.value for item in details.markdown] == [
        "3 out of 3 recent sessions were recorded as guarded. "
        "This was one detail that supported the AI’s match.",
    ]
    assert [item.value for item in details.caption] == [
        "These observations describe recorded interactions and cannot diagnose your child.",
    ]
    assert not at.metric and not at.dataframe and not at.table
    assert len(at.get("plotly_chart")) == 1
    for technical in ("Assigned cluster", "Nearest alternative cluster", "Squared-distance",
                      "contribution", "mean severity", "severity change"):
        assert technical.lower() not in text.lower()
    parent = wellbeing_card(at)
    assert not parent.metric and not parent.dataframe and not parent.get("plotly_chart")
    summary = " ".join(item.value for item in parent.markdown)
    assert "Calm and reflective" in summary and "Guarded responses" in summary
    assert "The overall trend showed more signs that your child may need support." in summary
    assert "This describes how your child responded in the latest period" in summary
    assert text.count("3 out of 3 recent sessions were recorded as guarded") == 1
    assert text.count("At the next quiet moment, keep the conversation low-pressure.") == 1
    assert text.count("What would you like me to understand about how that felt?") == 1
    assert "contribution" not in summary.lower()


def test_emotional_pattern_page_with_no_sessions(temp_db, monkeypatch):
    empty = services.demo_sessions().iloc[:0]
    monkeypatch.setattr(services, "sessions_for_child", lambda *_: empty)
    at = run_page()
    assert [e.label for e in emotional_expanders(at)] == [HISTORY, AI_DETAILS]
    assert not at.get("plotly_chart")
    text = visible_main_text(at)
    assert "Not enough information yet" in text
    assert "More recorded sessions are needed" in text
    assert "How have things felt online lately?" in text
    assert "The explanation will appear once there are enough recorded sessions." in text


def test_emotional_pattern_page_has_no_upload_label_or_contract_notice(child_with_sessions):
    at = run_page()
    main = html_of(at, at.main)
    assert "Emotional Pattern" not in main and "(Function 1)" not in main
    assert [e.label for e in emotional_expanders(at)] == [HISTORY, AI_DETAILS]
    assert not at.get("file_uploader")
    assert "pending a contract update" not in main
    assert "from the database" not in main                 # no child-count note
    assert at.selectbox[0].label == "Select Child"
    assert at.selectbox[0].value == "K1"                   # database children still loaded


def test_emotional_pattern_page_missing_model_shows_only_the_error(child_with_sessions,
                                                                  tmp_path, monkeypatch):
    monkeypatch.setenv("C4_F1_ARTIFACT_DIR", str(tmp_path))  # empty folder
    at = run_page()
    main, sidebar = html_of(at, at.main), html_of(at, at.sidebar)
    assert [notice.value for notice in at.info] == [F1_ERROR]
    assert "emotional_pattern_kmeans.pkl" not in visible_main_text(at)
    assert "Emotional Wellbeing" not in main
    assert not at.get("plotly_chart")                      # no chart
    assert not emotional_expanders(at)
    assert not at.selectbox                                # nothing to select
    assert "not loaded" in sidebar


def test_emotional_pattern_page_unloadable_model(child_with_sessions, tmp_path, monkeypatch):
    for name in services.F1_ARTIFACT_FILES:
        (tmp_path / name).write_text("garbage")
    monkeypatch.setenv("C4_F1_ARTIFACT_DIR", str(tmp_path))
    at = run_page()
    main = html_of(at, at.main)
    assert [notice.value for notice in at.info] == [F1_ERROR]
    assert "Emotional Wellbeing" not in main


# --- Function 2: Behavioral Anomaly page ----------------------------------------------

F2_ERROR = "Function 2 model could not be loaded. No anomaly prediction is shown."
F2_URGENT_TITLE = "Urgent Alerts (rule-based)"
F2_BANNED = ("threshold", "score", "Days shown", "Weekly Summary")   # never visible on the main page


def visible_main_text(at):
    """What a sighted viewer reads on the main page:
    HTML text (screen-reader-only text and tag attributes removed), Markdown and
    metric labels."""
    import re
    html = html_of(at, at.main)
    html = re.sub(r'<p class=\\?"c4-sr-only\\?">.*?</p>', " ", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text += " ".join(m.value for m in at.markdown)
    text += " ".join(m.label for m in at.metric)
    text += " ".join(c.value for c in at.caption)
    return text


def test_emotional_pattern_is_still_the_default_page(temp_db):
    at = run_page()
    main = html_of(at, at.main)
    assert at.selectbox[0].label == "Select Child"
    assert "Emotional Wellbeing" in main
    assert "Behavioral Anomaly Detection" not in main


def test_sidebar_shows_both_model_statuses(temp_db):
    sidebar = html_of(at := run_page(), at.sidebar)
    assert "Function 1 model" in sidebar and "Function 2 model" in sidebar
    assert "not loaded" not in sidebar


def test_behavioral_anomaly_page_with_real_model(temp_db):
    at = run_page("behavioral_anomaly")
    main = html_of(at, at.main)
    assert "Behavioral Anomaly Detection" in main
    assert "Behavioral Anomaly (Function 2)" not in main
    assert not any("Synthetic child data" in c.value for c in at.caption)
    assert at.selectbox(key="f2_child").options[0] == "C001"
    assert len(at.selectbox(key="f2_child").options) == 40
    assert "c4-flag" in main and "c4-badge" not in main
    assert len(at.get("plotly_chart")) == 1
    assert len(at.dataframe) == 2                         # top deviations and separate urgent alerts
    assert not at.date_input
    assert F2_URGENT_TITLE in main and "Weekly Summary" not in main
    assert [m.label for m in at.metric] == ["Flagged days · latest 30 days"]
    assert not at.expander
    assert not at.get("file_uploader")
    assert F2_ERROR not in main


def test_main_page_is_minimal_without_demo_subtitle_or_raw_scores(temp_db):
    at = run_page("behavioral_anomaly")
    text = visible_main_text(at).lower()
    # The helper sees real visible text (so the check below is not vacuous) ...
    assert "flagged days" in text and "behavioral anomaly detection" in text
    assert "synthetic child data" not in text
    # ... but not the screen-reader-only chart description.
    assert "amber diamonds" not in text
    for word in F2_BANNED:
        assert word.lower() not in text, word
    assert "Daily scores table" not in [e.label for e in at.expander]
    assert not any("week (Mon)" in d.value.columns for d in at.dataframe)
    deviations = next(d.value for d in at.dataframe if "Feature" in d.value.columns)
    assert len(deviations) == 3


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
    assert F1_ERROR not in main
    assert "**Calm and reflective**" in [item.value for item in wellbeing_card(at).markdown]
    assert "Function 2 model" in sidebar and "not loaded" in sidebar


@pytest.mark.parametrize("page", ["emotional_pattern", "behavioral_anomaly", "child_profile"])
def test_every_page_renders(temp_db, page):
    run_page(page)


def test_sidebar_has_no_data_and_admin_section(temp_db):
    at = run_page()
    sidebar = html_of(at, at.sidebar)
    assert "Data &amp; Admin" not in sidebar and "Data & Admin" not in sidebar
    links = [str(p.proto) for p in at.get("page_link")]
    for title in ("Dashboard", "Sessions", "Escalations", "Add record"):
        assert not any(f'label: "{title}"' in link for link in links)
    assert any('label: "Emotional Pattern"' in link for link in links)


def test_flagged_count_uses_latest_30_calendar_days_and_changes_with_child(temp_db):
    from datetime import timedelta
    at = run_page("behavioral_anomaly")
    for child in ["C001", "C009"]:
        at.selectbox(key="f2_child").set_value(child).run()
        assert not at.exception
        report = services.behavioral_report(services.behavioral_daily_source()["daily"], child)
        daily = report["daily"]
        end = daily.date.max()
        expected = int(daily.loc[daily.date.between(end-timedelta(days=29),end), "unusual"].sum())
        assert at.metric[0].value == str(expected)
        assert len(at.get("plotly_chart")) == 1


def test_explanations_remain_descriptive_without_technical_expanders(temp_db):
    at = run_page("behavioral_anomaly")
    assert not at.expander
    assert not at.get("file_uploader")
    assert "not exact model attributions" in " ".join(c.value for c in at.caption)


def test_stale_upload_state_does_not_change_demo_source(temp_db):
    at = AppTest.from_file(APP, default_timeout=120)
    # An upload left over from the old UI must not override the bundled source.
    at.session_state["f2_csv"] = "invalid stale CSV"
    at.run()
    at.switch_page("pages/behavioral_anomaly.py").run()
    assert not at.exception
    assert not at.error
    assert at.selectbox(key="f2_child").options == services.behavioral_daily_source()["child_ids"]
    assert "Demo" not in " ".join(c.value for c in at.caption)
    assert not at.expander and not at.get("file_uploader")

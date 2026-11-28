"""Render the real Streamlit pages (streamlit.testing AppTest) and check what
a viewer would see when the trained models load or fail to load."""
from datetime import date

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from parent_dashboard_test_helpers import follow_page_link
from db.child_profiles import PROFILE_DB_ENV
from ui import child_selection, services

APP = str(services.ROOT / "ui" / "app.py")
F1_ERROR = "The emotional wellbeing summary is temporarily unavailable."
HISTORY = "View Emotional History"
AI_DETAILS = "Why Did the AI Identify This Pattern?"


def run_legacy_comparison():
    entry = services.ROOT / "ui" / "legacy_pages" / "behavioral_comparison.py"
    at = AppTest.from_file(str(entry), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


SHAP_PREVIEW = "Research SHAP preview · synthetic data"


def html_of(at, area):
    return " ".join(str(h.proto) for h in area.get("html"))


@pytest.fixture(autouse=True)
def isolated_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv(PROFILE_DB_ENV, str(tmp_path / "profiles.db"))


def run_page(page="emotional_pattern", *, demo=True):
    at = AppTest.from_file(APP, default_timeout=120)
    if demo:
        at.session_state[child_selection.DEMO_WIDGET] = True
    at.run()
    if page:
        title = {"emotional_pattern": "Emotional Pattern", "behavioral_anomaly": "Behavioral Anomaly",
                 "child_profile": "Child Profiles"}[page]
        follow_page_link(at, at.sidebar, title)
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_research_shap_is_hidden_and_inactive_by_default(temp_db, monkeypatch):
    monkeypatch.delenv("C4_F2_RESEARCH_SHAP", raising=False)
    monkeypatch.setattr(services, "research_shap_context",
                        lambda *_: pytest.fail("Preview must be inactive"))
    at = run_legacy_comparison()
    assert not any(expander.label == SHAP_PREVIEW for expander in at.expander)
    assert not any(button.label == "Explain selected day" for button in at.button)


def test_research_shap_is_on_demand_and_clears_changed_selection(temp_db, monkeypatch):
    monkeypatch.setenv("C4_F2_RESEARCH_SHAP", "1")
    original = services.research_shap_report
    calls = []

    def record(*args):
        calls.append(args[1:])
        return original(*args)

    monkeypatch.setattr(services, "research_shap_report", record)
    at = run_legacy_comparison()
    assert any(expander.label == SHAP_PREVIEW for expander in at.expander)
    assert not calls
    at.selectbox(key="f2_child").select("C033").run()
    at.selectbox(key="f2_research_day_C033").select(date(2026, 9, 25)).run()
    assert not calls
    at.button(key="f2_research_explain").click().run()
    assert not at.exception and calls == [("C033", date(2026, 9, 25))]
    text = "\n".join(node.value for node in at.markdown)
    assert "`uncertain`" in text and "**Frozen anomaly score:**" in text
    assert ("The system identified an unusual pattern in the available recorded activity, "
            "but cannot reliably identify which recorded factors contributed most.") in text
    assert len(at.get("plotly_chart")) == 2
    assert any("Contribution" in frame.value.columns and len(frame.value) == 12 for frame in at.dataframe)
    at.selectbox(key="f2_research_day_C033").select(date(2026, 9, 24)).run()
    assert len(calls) == 1 and len(at.get("plotly_chart")) == 1
    assert not any("Reliability status" in node.value for node in at.markdown)
    at.selectbox(key="f2_research_day_C033").select(date(2026, 9, 25)).run()
    assert not any("Reliability status" in node.value for node in at.markdown)
    at.button(key="f2_research_explain").click().run()
    at.selectbox(key="f2_child").select("C008").run()
    assert not any("Reliability status" in node.value for node in at.markdown)
    at.selectbox(key="f2_research_day_C008").select(date(2026, 9, 15)).run()
    at.button(key="f2_research_explain").click().run()
    text = "\n".join(node.value for node in at.markdown)
    assert "`reliable`" in text
    assert "Compared with typical recorded days" not in text


def test_research_shap_failure_keeps_existing_cards(temp_db, monkeypatch):
    monkeypatch.setenv("C4_F2_RESEARCH_SHAP", "1")

    def unavailable(*args):
        raise RuntimeError("research-only failure")

    monkeypatch.setattr(services, "research_shap_report", unavailable)
    at = run_legacy_comparison()
    at.button(key="f2_research_explain").click().run()
    assert not at.exception
    assert any("Research SHAP preview could not run" in warning.value for warning in at.warning)
    assert len(at.get("plotly_chart")) == 1
    assert "Urgent Alerts (rule-based)" in html_of(at, at.main)


def parent_demo_address(monkeypatch, address):
    get_option = st.get_option
    monkeypatch.setattr(st, "get_option",
                        lambda name: address if name == "server.address" else get_option(name))


@pytest.mark.parametrize(("flag", "address"), [(None, "127.0.0.1"), ("1", "0.0.0.0")])
def test_parent_demo_hides_without_local_opt_in(temp_db, monkeypatch, flag, address):
    if flag is None:
        monkeypatch.delenv("C4_F2_PARENT_XAI_DEMO", raising=False)
    else:
        monkeypatch.setenv("C4_F2_PARENT_XAI_DEMO", flag)
    parent_demo_address(monkeypatch, address)
    monkeypatch.setattr(services, "parent_xai_demo_context",
                        lambda *_: pytest.fail("Hidden demo must remain inactive"))
    at = run_legacy_comparison()
    assert "Activity Update" not in html_of(at, at.main)
    assert not any(button.label == "Show activity update" for button in at.button)


def test_parent_demo_shows_plain_text_and_clears_stale_day(temp_db, monkeypatch):
    monkeypatch.setenv("C4_F2_PARENT_XAI_DEMO", "1")
    monkeypatch.delenv("C4_F2_RESEARCH_SHAP", raising=False)
    parent_demo_address(monkeypatch, "127.0.0.1")
    at = run_legacy_comparison()
    assert "Activity Update" in html_of(at, at.main)
    assert any("SYNTHETIC DEMO PREVIEW" in item.value for item in at.caption)
    at.selectbox(key="f2_child").select("C008").run()
    at.selectbox(key="f2_parent_activity_day_C008").select(date(2026, 9, 15)).run()
    original_html = html_of(at, at.main)
    assert not any("contributed most" in item.value for item in at.markdown)
    at.button(key="f2_parent_activity_show").click().run()
    assert not at.exception
    text = "\n".join(item.value for item in at.markdown)
    assert "Your child's recorded screen-related activity contributed most" in text
    assert all(marker not in text for marker in ("SHAP", "Saved threshold", "Reliability status", "Gate reasons"))
    assert len(at.get("plotly_chart")) == 1
    assert "Urgent Alerts (rule-based)" in original_html and "Urgent Alerts (rule-based)" in html_of(at, at.main)
    assert not any(expander.label == SHAP_PREVIEW for expander in at.expander)
    at.selectbox(key="f2_parent_activity_day_C008").select(date(2026, 9, 16)).run()
    assert not any("contributed most" in item.value for item in at.markdown)
    at.selectbox(key="f2_parent_activity_day_C008").select(date(2026, 9, 15)).run()
    assert not any("contributed most" in item.value for item in at.markdown)
    at.button(key="f2_parent_activity_show").click().run()
    at.selectbox(key="f2_child").select("C002").run()
    assert not any("contributed most" in item.value for item in at.markdown)


def wellbeing_card(at):
    return next(container for container in at.get("flex_container")
                if container.proto.id.endswith("-card-emotional-wellbeing"))


def emotional_expanders(at):
    # AppTest classifies expandable blocks with icons as status blocks.
    return [block for block in at.main if block.type in ("expander", "status")]


@pytest.fixture
def child_with_sessions(temp_db, monkeypatch):
    for i, emo in enumerate(["calm", "calm", "defensive", "calm", "calm"]):
        services.save_record({
            "child_id": "K1", "session_id": f"K1-{i}", "timestamp": f"2026-07-0{i + 1}T10:00:00Z",
            "risk_level": "low", "emotional_state": emo, "self_regulation_shown": False,
            "escalation_flag": False, "dialogue_turns": 2})
    sessions = services.sessions_for_child(services.child_session_source(), "K1")
    monkeypatch.setattr(services, "demo_sessions", lambda: sessions)


def test_emotional_pattern_page_with_real_model(child_with_sessions):
    at = run_page()
    assert len(at.get("plotly_chart")) == 1
    assert [e.label for e in emotional_expanders(at)] == [HISTORY, AI_DETAILS]
    assert not at.info
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
    at.selectbox(key=child_selection.F1_DEMO_CHILD).set_value("C004").run()
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
    assert at.main.selectbox[0].label == "Select Child"
    assert at.main.selectbox[0].value == "K1"              # fixture demo sessions


def test_emotional_pattern_page_missing_model_shows_only_the_error(child_with_sessions,
                                                                  tmp_path, monkeypatch):
    monkeypatch.setenv("C4_F1_ARTIFACT_DIR", str(tmp_path))  # empty folder
    at = run_page()
    main = html_of(at, at.main)
    assert [notice.value for notice in at.info] == [F1_ERROR]
    assert "emotional_pattern_kmeans.pkl" not in visible_main_text(at)
    assert "Emotional Wellbeing" not in main
    assert not at.get("plotly_chart")                      # no chart
    assert not emotional_expanders(at)
    assert not at.main.selectbox                           # no assessment selector
    assert "SYNTHETIC DEMO DATA" in visible_main_text(at)


def test_emotional_pattern_page_unloadable_model(child_with_sessions, tmp_path, monkeypatch):
    for name in services.F1_ARTIFACT_FILES:
        (tmp_path / name).write_text("garbage")
    monkeypatch.setenv("C4_F1_ARTIFACT_DIR", str(tmp_path))
    at = run_page()
    main = html_of(at, at.main)
    assert [notice.value for notice in at.info] == [F1_ERROR]
    assert "Emotional Wellbeing" not in main


# --- Function 2: Behavioral Anomaly page ----------------------------------------------

F2_ERROR = "Behavioral assessment is temporarily unavailable. Please try again later."
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


def test_overview_is_the_default_page(temp_db):
    at = run_page(None, demo=False)
    main = html_of(at, at.main)
    assert "Overview" in main
    assert not at.main.selectbox
    assert "Emotional Wellbeing" not in main


def test_sidebar_omits_both_routine_model_statuses(temp_db):
    sidebar = html_of(at := run_page(), at.sidebar)
    assert "Function 1 model" not in sidebar and "Function 2 model" not in sidebar
    assert "not loaded" not in sidebar


def test_behavioral_anomaly_page_with_real_model(temp_db):
    at = run_legacy_comparison()
    main = html_of(at, at.main)
    assert "Behavioral Anomaly Detection" in main
    assert "Behavioral Anomaly (Function 2)" not in main
    assert "Demo Mode · Fictional data for this assessment." in [c.value for c in at.caption]
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
    at = run_legacy_comparison()
    text = visible_main_text(at).lower()
    # The helper sees real visible text (so the check below is not vacuous) ...
    assert "flagged days" in text and "behavioral anomaly detection" in text
    assert "demo mode · fictional data for this assessment." in text
    assert "it does not guarantee that the child is safe" in text
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
    at = run_legacy_comparison()                    # C001: latest day not flagged
    main = html_of(at, at.main)
    assert re.search(r'aria-label=\\?"2026-09-28: Not flagged: the model found no unusual pattern', main)
    assert re.search(r'alt=\\?"2026-09-28: Not flagged', main)
    assert "This is not proof that the child is safe" in main          # part of the description
    assert "c4-sr-only" in main and "amber diamonds" in main            # chart description


def test_flagged_latest_day_uses_the_flagged_icon(temp_db):
    daily = services.behavioral_daily_source()["daily"]
    flagged_last = next(c for c in services.child_ids(daily)
                        if services.behavioral_report(daily, c)["latest"]["unusual"])
    at = run_legacy_comparison()
    at.selectbox(key="f2_child").set_value(flagged_last).run()
    assert not at.exception
    assert "Flagged: the model found an unusual pattern" in html_of(at, at.main)



def test_behavioral_anomaly_page_switches_child(temp_db):
    at = run_legacy_comparison()
    at.selectbox(key="f2_child").set_value("C009").run()
    assert not at.exception
    assert at.selectbox(key="f2_day_C009").value is not None   # C009 has flagged days


def test_behavioral_anomaly_page_missing_model_keeps_urgent_alerts(temp_db, tmp_path, monkeypatch):
    monkeypatch.setenv("C4_F2_ARTIFACT_DIR", str(tmp_path))   # empty folder
    at = run_legacy_comparison()
    main = html_of(at, at.main)
    assert [item.value for item in at.main.error] == [F2_ERROR]
    assert "behavioral_anomaly_pipeline.pkl" not in main and "scikit-learn==1.6.1" not in main
    assert str(tmp_path) not in main and str(tmp_path) not in at.main.error[0].value
    assert not at.get("page_link")  # The offline comparison has no dashboard navigation.
    assert "c4-flag" not in main
    assert not at.get("plotly_chart")
    assert at.selectbox(key="f2_child").value == "C001"
    assert F2_URGENT_TITLE in main
    assert len(at.dataframe) == 1 and "rule matched" in at.dataframe[0].value.columns
    assert not at.dataframe[0].value.empty


def test_behavioral_anomaly_page_missing_model_shows_only_the_error_and_safety_status(
        temp_db, tmp_path, monkeypatch):
    from component4_profiling_xai.src.urgent_alerts import URGENT_RULE_COLUMNS

    monkeypatch.setenv("C4_F2_ARTIFACT_DIR", str(tmp_path))
    source = services.behavioral_daily_source()
    child = source["child_ids"][0]
    daily = source["daily"].loc[source["daily"]["child_id"].eq(child)].tail(1).copy()
    daily[list(URGENT_RULE_COLUMNS)] = 0
    monkeypatch.setattr(services, "behavioral_daily_source",
                        lambda: {**source, "daily": daily, "child_ids": [child]})

    at = run_legacy_comparison()
    main = html_of(at, at.main)
    assert [item.value for item in at.main.error] == [F2_ERROR]
    assert "behavioral_anomaly_pipeline.pkl" not in main and "scikit-learn==1.6.1" not in main
    assert str(tmp_path) not in main
    assert "c4-flag" not in main
    assert not at.get("plotly_chart") and not at.dataframe
    assert at.selectbox(key="f2_child").value == child
    assert F2_URGENT_TITLE in main
    assert "No urgent rules matched the available records." in visible_main_text(at)


def test_behavioral_anomaly_page_shows_insufficient_data_instead_of_a_normal_status(temp_db, monkeypatch):
    source = services.behavioral_daily_source()
    daily = source["daily"].query("child_id == 'C001'").tail(1).copy()
    daily.loc[:, "screen_monitor_observed"] = 0.0
    daily.loc[:, "language_urgent_flags"] = 2
    monkeypatch.setattr(services, "behavioral_daily_source",
                        lambda: {"daily": daily, "child_ids": ["C001"], "source": "demo"})
    at = run_legacy_comparison()
    main = html_of(at, at.main)
    assert "Insufficient recorded information to assess behavior for this period." in visible_main_text(at)
    assert "c4-flag" not in main and not at.metric
    assert "Insufficient data" in str(at.get("plotly_chart")[0].proto)
    assert "urgent language flag (2)" in at.dataframe[0].value.iloc[0]["rule matched"]


def test_behavioral_parent_messages_distinguish_records_coverage_and_failure(temp_db, monkeypatch, caplog):
    source = services.behavioral_daily_source()
    monkeypatch.setattr(services, "behavioral_daily_source", lambda: dict(source, child_ids=[]))
    at = run_legacy_comparison()
    assert [item.value for item in at.main.info] == ["No fictional monitoring records are available."]
    assert not at.main.error

    daily = source["daily"].assign(screen_monitor_observed=0, language_monitor_observed=0)
    monkeypatch.setattr(services, "behavioral_daily_source", lambda: dict(source, daily=daily))
    at.run()
    assert "insufficient monitoring information" in at.main.info[0].value
    assert not at.main.error and not at.get("plotly_chart")

    def fail(*args):
        raise ValueError("private source detail at E:/private/child-record.csv")

    monkeypatch.setattr(services, "behavioral_daily_source", lambda: source)
    monkeypatch.setattr(services, "behavioral_report", fail)
    at.run()
    assert not at.exception and not at.get("plotly_chart")
    assert [item.value for item in at.main.error] == [F2_ERROR]
    assert "private" not in visible_main_text(at) and "private" not in caplog.text
    assert "assessment execution (ValueError)" in caplog.text


def test_function2_failure_leaves_function1_page_working(child_with_sessions, tmp_path, monkeypatch):
    monkeypatch.setenv("C4_F2_ARTIFACT_DIR", str(tmp_path))
    at = run_page()
    main = html_of(at, at.main)
    assert F1_ERROR not in main
    assert "**Calm and reflective**" in [item.value for item in wellbeing_card(at).markdown]
    assert F2_ERROR not in main


@pytest.mark.parametrize("page", ["emotional_pattern", "behavioral_anomaly", "child_profile"])
def test_every_page_renders(temp_db, page):
    run_page(page, demo=False)


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
    at = run_legacy_comparison()
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
    at = run_legacy_comparison()
    assert not at.expander
    assert not at.get("file_uploader")
    assert "not exact model attributions" in " ".join(c.value for c in at.caption)


def test_stale_upload_state_does_not_change_demo_source(temp_db):
    at = AppTest.from_file(APP, default_timeout=120)
    # An upload left over from the old UI must not override the bundled source.
    at.session_state["f2_csv"] = "invalid stale CSV"
    at.session_state[child_selection.DEMO_WIDGET] = True
    at.run()
    follow_page_link(at, at.sidebar, "Behavioral Anomaly")
    assert not at.exception
    assert not at.error
    assert at.selectbox(key="f2_child").options == services.behavioral_daily_source()["child_ids"]
    assert "Demo Mode · Fictional data" in " ".join(c.value for c in at.main.caption)
    assert at.selectbox(key="f2_activity_C001").label == "Activity date"
    assert not any(expander.label == SHAP_PREVIEW for expander in at.expander)
    assert not at.get("file_uploader")

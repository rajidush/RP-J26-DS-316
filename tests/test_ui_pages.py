"""Render the real Streamlit pages (streamlit.testing AppTest) and check what
a viewer would see when the trained models load or fail to load."""
from datetime import date

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from ui import services

APP = str(services.ROOT / "ui" / "app.py")
F1_ERROR = "Function 1 model could not be loaded. No emotional pattern is shown."
SHAP_PREVIEW = "Research SHAP preview · synthetic data"


def html_of(at, area):
    return " ".join(str(h.proto) for h in area.get("html"))


def run_page(page=None):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    if page:
        at.switch_page(f"pages/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_research_shap_is_hidden_and_inactive_by_default(temp_db, monkeypatch):
    monkeypatch.delenv("C4_F2_RESEARCH_SHAP", raising=False)
    monkeypatch.setattr(services, "research_shap_context",
                        lambda *_: pytest.fail("Preview must be inactive"))
    at = run_page("behavioral_anomaly")
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
    at = run_page("behavioral_anomaly")
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
    at = run_page("behavioral_anomaly")
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
    at = run_page("behavioral_anomaly")
    assert "Activity Update" not in html_of(at, at.main)
    assert not any(button.label == "Show activity update" for button in at.button)


def test_parent_demo_shows_plain_text_and_clears_stale_day(temp_db, monkeypatch):
    monkeypatch.setenv("C4_F2_PARENT_XAI_DEMO", "1")
    monkeypatch.delenv("C4_F2_RESEARCH_SHAP", raising=False)
    parent_demo_address(monkeypatch, "127.0.0.1")
    at = run_page("behavioral_anomaly")
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
    assert "Emotional Pattern" in main
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
    assert "Research demonstration using synthetic data" in [c.value for c in at.caption]
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
    assert "research demonstration using synthetic data" in text
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


def test_behavioral_anomaly_page_missing_model_keeps_urgent_alerts(temp_db, tmp_path, monkeypatch):
    monkeypatch.setenv("C4_F2_ARTIFACT_DIR", str(tmp_path))   # empty folder
    at = run_page("behavioral_anomaly")
    main = html_of(at, at.main)
    assert F2_ERROR in main
    assert "behavioral_anomaly_pipeline.pkl" in main and "scikit-learn==1.6.1" in main
    assert "c4-flag" not in main
    assert not at.get("plotly_chart")
    assert at.selectbox(key="f2_child").value == "C001"
    assert F2_URGENT_TITLE in main
    assert len(at.dataframe) == 1 and "rule matched" in at.dataframe[0].value.columns
    assert not at.dataframe[0].value.empty


def test_behavioral_anomaly_page_shows_insufficient_data_instead_of_a_normal_status(temp_db, monkeypatch):
    source = services.behavioral_daily_source()
    daily = source["daily"].query("child_id == 'C001'").tail(1).copy()
    daily.loc[:, "screen_monitor_observed"] = 0.0
    daily.loc[:, "language_urgent_flags"] = 2
    monkeypatch.setattr(services, "behavioral_daily_source",
                        lambda: {"daily": daily, "child_ids": ["C001"], "source": "demo"})
    at = run_page("behavioral_anomaly")
    main = html_of(at, at.main)
    assert "Insufficient recorded information to assess behavior for this period." in visible_main_text(at)
    assert "c4-flag" not in main and not at.metric
    assert "Insufficient data" in str(at.get("plotly_chart")[0].proto)
    assert "urgent language flag (2)" in at.dataframe[0].value.iloc[0]["rule matched"]


def test_function2_failure_leaves_function1_page_working(child_with_sessions, tmp_path, monkeypatch):
    monkeypatch.setenv("C4_F2_ARTIFACT_DIR", str(tmp_path))
    at = run_page()
    main, sidebar = html_of(at, at.main), html_of(at, at.sidebar)
    assert "c4-badge" in main and F1_ERROR not in main
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

import streamlit as st

from ui import services, theme

theme.page_header("Emotional Pattern",
                  "See your child's current emotional pattern and how it has changed over time.")


def show_model_error(status):
    """No badge, chart or features without the trained model -- only this card."""
    theme.error_card(
        "f1", "Function 1 model could not be loaded. No emotional pattern is shown.",
        status["reason"],
        ["Put `emotional_pattern_kmeans.pkl`, `emotional_pattern_scaler.pkl` and "
         "`emotional_pattern_metadata.json` in `component4_profiling_xai/models/`, "
         "or set `C4_F1_ARTIFACT_DIR` to the folder that has them.",
         "Install the scikit-learn version the model was trained with: "
         "`pip install scikit-learn==1.6.1`.",
         "Then restart the app."])
    st.stop()


f1_status = services.function1_status()
if not f1_status["available"]:
    show_model_error(f1_status)

source = services.child_session_source()
ids = source["child_ids"]

with theme.card("select-child", "Select Child"):
    child = st.selectbox("Select child", ids, index=ids.index("C004") if "C004" in ids else 0,
                         label_visibility="collapsed", width=440)
    if source["source"] == "demo":
        theme.note("DEMO DATA: synthetic children C001-C005. The database has no sessions with "
                   "child_id yet; run scripts/import_comp3_sessions.py to load Component 3's data.")

try:
    report = services.emotional_pattern_report(services.sessions_for_child(source, child), child)
except services.ModelUnavailableError as exc:
    show_model_error(exc.status)
except ValueError as exc:
    st.error(f"Function 1 could not run on this data: {exc}")
    st.stop()

n_sessions = theme.plural(report["sessions"], "session")
min_sessions = theme.plural(report["min_sessions"], "session")

with theme.card("current-pattern", "Current Emotional Pattern"):
    if not report["enough_data"]:
        theme.badge(theme.NOT_ENOUGH_DATA_STYLE,
                    subtitle=f"{n_sessions} so far, at least {report['min_sessions']} "
                             "are needed for a pattern.")
    else:
        since = (f"Latest 2-week window from {report['current_since']}"
                 if report["current_since"] else f"Across all {n_sessions}")
        theme.badge(theme.pattern_style(report["current_pattern"]), subtitle=since)
        if report["overall_pattern"] != report["current_pattern"]:
            theme.note(f"Overall pattern across all {n_sessions}: {report['overall_pattern']}.")

with theme.card("pattern-timeline", "Emotion Pattern Change Over Time"):
    if report["timeline"]:
        st.plotly_chart(theme.pattern_timeline_figure(report["timeline"]),
                        config=theme.PLOTLY_CONFIG, width="stretch")
        window_rule = f"Each point is a 2-week window with at least {min_sessions}."
        if report["severity_change"] is not None:
            theme.note(f"Trend: {report['trend']} (severity change "
                       f"{report['severity_change']:+}). {window_rule}")
        else:
            theme.note(window_rule)
    else:
        st.write(f"No trend yet: each 2-week window needs at least {min_sessions}.")

    # Only explain a pattern that exists.
    if report["enough_data"]:
        with st.expander("Why this pattern?"):
            st.write("The badge uses the latest 2-week window. The overall pattern uses all sessions.")
            st.dataframe(services.pattern_features_table(report), hide_index=True,
                         width="stretch")

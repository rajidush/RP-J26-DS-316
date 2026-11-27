import streamlit as st

from ui import activity_summary, child_selection, safety_summary, services, theme


def local_activity():
    child = child_selection.selected_profile_label(include_avatar=True)
    if child:
        theme.card("activity-child", child, "Monitoring data has not been connected for this child.")
    elif child_selection.has_active_profiles() is False:
        st.info("There are no active child profiles.")
        st.page_link("pages/child_profile.py", label="Go to Child Profiles", icon=":material/person:")
    elif child_selection.has_active_profiles() is None:
        st.info("Child profiles are temporarily unavailable.")
    else:
        st.info("Select a child profile in the sidebar to view its activity summary.")


def show_summary(summary, safety):
    counts = summary.assessment_counts(safety)
    with theme.card("activity-metrics", "Digital Activity Summary"):
        language, unusual, urgent = st.columns(3)
        with language:
            st.metric("Recorded language flags", summary.language_flags)
            if summary.average_flags is not None:
                st.caption(f"{summary.average_flags:.2f} per day with a usable count ({summary.language_days} days).")
        with unusual:
            st.metric("Days with unusual activity", counts.unusual_days)
            if counts.unusual_days is None:
                st.caption(safety.behavioral_message if safety.behavioral_message == safety_summary.BEHAVIOR_FAILED
                           else "No days could be assessed in this period.")
            else:
                st.caption(f"Across {counts.assessed_days} assessed days.")
        with urgent:
            st.metric("Days with urgent safety flags", counts.urgent_days)
            if counts.urgent_days is None:
                st.caption(safety_summary.URGENT_FAILED if safety.urgent_message == safety_summary.URGENT_FAILED
                           else "Urgent safety information is unavailable for this period.")
            else:
                st.caption("A day can contain several flags.")
            st.page_link("pages/safety_notifications.py", label="Open Safety Notifications", icon=":material/notifications:")

    with theme.card("activity-history", "Activity History"):
        if summary.language_flags is None:
            st.info("Daily language flag counts are unavailable for this period.")
        else:
            st.plotly_chart(activity_summary.language_history_figure(summary, theme.COLORS["blue"]),
                            config=theme.PLOTLY_CONFIG, width="stretch")
            st.caption("Bars show recorded language flags. A dot marks a recorded zero. Blank dates have no usable count.")

    with theme.card("activity-coverage", "Data Availability"):
        missing = summary.days - summary.records
        st.write(f"Daily records are available for {summary.records} of {summary.days} calendar days "
                 f"({missing} missing).")
        st.write(f"Usable language flag counts are available for {summary.language_days} of {summary.days} days.")
        st.write(f"Behavioral activity could be assessed on {counts.assessed_days} of {summary.days} days.")
        if counts.complete_urgent_days < summary.days and safety.urgent_message != safety_summary.URGENT_FAILED:
            st.caption(f"Complete urgent safety inputs were available on {counts.complete_urgent_days} "
                       f"of {summary.days} days.")
        st.caption("Language flags reflect recorded monitoring results. Total screen time and conversation totals are unavailable.")


def demo_activity():
    st.info("Demo Mode is on. All activity results on this page are fictional.")
    try:
        source = services.behavioral_daily_source()
    except (ImportError, OSError, ValueError, RuntimeError):
        st.error("Fictional activity records could not be loaded.")
        return
    if source["source"] != "demo" or not source["child_ids"]:
        st.info("No fictional activity records are available.")
        return

    child = child_selection.behavioral_demo_picker(
        "Function 02 demo child", source["child_ids"], "activity-demo-child",
        placeholder="Select a fictional child",
    )
    if child is None:
        st.info("Select a fictional child to view an activity summary.")
        return
    rows = safety_summary.records_for(source, child)
    if rows.empty:
        st.info("No fictional activity records are available for this child.")
        return
    dates = sorted(rows["date"].unique(), reverse=True)

    with theme.card("activity-period", "Selected Activity Period"):
        through = child_selection.behavioral_date_picker(
            "Assessment date", dates, f"activity-assessment-date-{child}",
        )
        periods = list(activity_summary.PERIODS)
        selected = st.radio("Activity period", periods, horizontal=True, index=periods.index(
            st.session_state.get(child_selection.ACTIVITY_PERIOD, activity_summary.DEFAULT_PERIOD)),
            key="activity-period-widget",
        )
        st.session_state[child_selection.ACTIVITY_PERIOD] = selected
        try:
            summary = child_selection.activity_demo_summary(source, child)
        except ValueError:
            st.error("The fictional activity records could not be summarized.")
            return
        st.caption(f"{summary.period} · Function 02 demo · Fictional child {child}")

    show_summary(summary, child_selection.safety_demo_summary(source, child, through))


theme.page_header("Activity Summary", "Review recorded digital activity over a selected period.")
if child_selection.demo_mode_active():
    demo_activity()
else:
    local_activity()

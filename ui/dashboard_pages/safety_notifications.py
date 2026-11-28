import streamlit as st

from ui import child_selection, safety_summary, services, theme


def local_safety():
    child = child_selection.selected_profile_label(include_avatar=True)
    if child:
        theme.card("safety-child", child, "Monitoring data has not been connected for this child.")
    elif child_selection.has_active_profiles() is False:
        st.info("There are no active child profiles.")
        st.page_link("dashboard_pages/child_profile.py", label="Go to Child Profiles", icon=":material/person:")
    elif child_selection.has_active_profiles() is None:
        st.info("Child profiles are temporarily unavailable.")
    else:
        st.info("Select a child profile in the sidebar to view its safety information.")


def notification_details(item):
    st.caption(f"{item.date} · {item.context} · Fictional record")
    st.write(item.description)
    if item.limitation:
        st.caption(item.limitation)


def show_notifications(summary):
    with theme.card("safety-overview", "Safety Overview"):
        if summary.period:
            st.caption(f"{summary.period} · Up to and including the assessment date")
        if summary.start is None:
            st.info(safety_summary.NO_RECORDS)
            return
        st.write(summary.urgent_message)
        st.write(summary.behavioral_message)
        if summary.coverage:
            st.warning(summary.coverage, icon=":material/info:")

    urgent = summary.history(safety_summary.URGENT)
    behavioral = summary.history(safety_summary.BEHAVIORAL)
    if urgent:
        with theme.card("safety-urgent", "Urgent Safety Concerns"):
            st.error("Urgent safety flags in the most recent matching daily record.", icon=":material/priority_high:")
            notification_details(urgent[0])
    with theme.card("safety-behavioral", "Unusual Behavioral Activity"):
        if behavioral:
            notification_details(behavioral[0])
        else:
            st.caption("There are no behavioral observations to show for this selection.")
        st.page_link("dashboard_pages/behavioral_anomaly.py", label="Open Behavioral Anomaly", icon=":material/monitoring:")

    with theme.card("safety-history", "Notification History"):
        category = st.selectbox("Notification category", (
            "All notifications", safety_summary.URGENT, safety_summary.BEHAVIORAL,
        ), key="safety-history-category")
        items = summary.history(None if category == "All notifications" else category)
        if not items:
            st.info("There are no supported notifications to show for this selection.")
        for item in items:
            icon = ":material/priority_high:" if item.category == safety_summary.URGENT else ":material/monitoring:"
            with st.expander(f"{item.date} · {item.category}", icon=icon):
                notification_details(item)

    if urgent or behavioral:
        with theme.card("safety-guidance", "Parent Guidance"):
            st.write("These daily records contain flags and activity observations. Specific next steps are unavailable from this assessment.")


def demo_safety():
    st.info("Demo Mode is on. All notifications on this page are fictional.")
    try:
        source = services.behavioral_daily_source()
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        services.log_assessment_failure("safety", "record loading", error)
        st.error("Fictional monitoring records could not be loaded.")
        return
    if source["source"] != "demo" or not source["child_ids"]:
        st.info(safety_summary.NO_RECORDS)
        return

    child = child_selection.behavioral_demo_picker(
        "Function 02 demo child", source["child_ids"], "safety-demo-child",
        placeholder="Select a fictional child",
    )
    if child is None:
        st.info("Select a fictional child to view available safety notifications.")
        return

    rows = safety_summary.records_for(source, child)
    if rows.empty:
        st.info(safety_summary.NO_RECORDS)
        return
    dates = sorted(rows["date"].unique(), reverse=True)
    through = child_selection.behavioral_date_picker("Assessment date", dates, f"safety-assessment-date-{child}")
    show_notifications(child_selection.safety_demo_summary(source, child, through))


theme.page_header("Safety Notifications", "Review recorded safety concerns and unusual activity.")
if child_selection.demo_mode_active():
    demo_safety()
else:
    local_safety()

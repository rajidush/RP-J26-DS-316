import streamlit as st

from ui import child_selection, overview_summary, services, theme


def linked_card(key, title, message, page, label, icon):
    with theme.card(key, title, message):
        st.page_link(f"pages/{page}.py", label=label, icon=f":material/{icon}:")


def demo_card(key, title, source_label, load_source, summarize, page, link, icon):
    results = st.session_state.setdefault(child_selection.OVERVIEW_DEMO_RESULTS, {})
    with theme.card(f"overview-{key}", title):
        st.caption(source_label)
        try:
            source = load_source()
        except (OSError, ValueError):
            source = None
        child = None
        summary = None
        if not source or source["source"] != "demo" or not source["child_ids"]:
            results.pop(key, None)
            st.info("Fictional records for this example are unavailable.")
        else:
            child = st.selectbox(
                f"{title} demo child", source["child_ids"], index=None,
                key=f"overview-demo-{key}", placeholder="Select a fictional child",
            )
            previous_child, summary = results.get(key, (None, None))
            if previous_child != child:
                results.pop(key, None)
                summary = None
            if st.button(f"Show {title.lower()} summary", key=f"overview-run-{key}", disabled=child is None):
                try:
                    summary = summarize(source, child)
                except (OSError, ValueError, services.ModelUnavailableError):
                    summary = overview_summary.Summary("This fictional assessment could not be generated.")
                results[key] = (child, summary)
            if summary:
                st.write(summary.message)
                if summary.period:
                    st.caption(summary.period)
                if summary.change:
                    st.write(summary.change)
        st.page_link(f"pages/{page}.py", label=link, icon=f":material/{icon}:")
    return child, summary


def local_overview():
    profile = child_selection.selected_profile()
    name = child_selection.selected_profile_label(include_avatar=True)
    if profile:
        with theme.card("overview-child", "Selected Child"):
            st.text(name)
            if profile.age_group:
                st.text(f"Age group: {profile.age_group}")
            st.caption("Monitoring data has not been connected for this child.")
    elif child_selection.has_active_profiles() is None:
        st.info("Child profiles are temporarily unavailable.")
    elif child_selection.has_active_profiles():
        st.info("Select a child profile in the sidebar.")
    else:
        st.info("There are no active child profiles.")
        st.page_link("pages/child_profile.py", label="Go to Child Profiles", icon=":material/person:")

    if profile is None:
        return
    emotional, behavioral = st.columns(2)
    with emotional:
        linked_card("overview-emotional", "Emotional Wellbeing", "Emotional assessment data is unavailable.",
                    "emotional_pattern", "Open Emotional Pattern", "bar_chart")
    with behavioral:
        linked_card("overview-behavioral", "Behavioral Activity", "Behavioral assessment data is unavailable.",
                    "behavioral_anomaly", "Open Behavioral Anomaly", "monitoring")
    supporting_cards(
        "Safety information is unavailable without connected monitoring data.",
        "Parent guidance is unavailable without a supported assessment.",
    )


def supporting_cards(safety, guidance, *, safety_caption=None, guidance_caption=None):
    left, right = st.columns(2)
    with left, theme.card("overview-safety", "Safety Alerts", safety):
        if safety_caption:
            st.caption(safety_caption)
        st.page_link("pages/safety_notifications.py", label="Open Safety Notifications", icon=":material/notifications:")
    with right:
        linked_card("overview-activity", "Activity Summary", "Activity summary data is unavailable.",
                    "activity_summary", "Open Activity Summary", "insights")
    with theme.card("overview-guidance", "Parent Guidance", guidance):
        if guidance_caption:
            st.caption(guidance_caption)


def demo_overview():
    st.info("Demo Mode is on. The two assessments use separate fictional records.")
    emotional, behavioral = st.columns(2)
    with emotional:
        emotional_child, emotional_result = demo_card(
            "emotional", "Emotional Wellbeing", "Fictional data · Emotional Pattern demo",
            lambda: services.child_session_source(demo_only=True), overview_summary.emotional_summary,
            "emotional_pattern", "Open Emotional Pattern", "bar_chart",
        )
    with behavioral:
        behavioral_child, behavioral_result = demo_card(
            "behavioral", "Behavioral Activity", "Fictional data · Behavioral Anomaly demo",
            services.behavioral_daily_source, overview_summary.behavioral_summary,
            "behavioral_anomaly", "Open Behavioral Anomaly", "monitoring",
        )
    safety = behavioral_result.related if behavioral_result else ""
    guidance = emotional_result.related if emotional_result else ""
    supporting_cards(
        safety or "Fictional safety information is unavailable until a behavioral example is evaluated.",
        guidance or "There is no supported guidance for the current fictional emotional example.",
        safety_caption=f"Behavioral Anomaly demo · Fictional child {behavioral_child}" if safety else None,
        guidance_caption=f"Emotional Pattern demo · Fictional child {emotional_child}" if guidance else None,
    )


theme.page_header("Overview", "A quick look at your child’s digital wellbeing.")
if child_selection.demo_mode_active():
    demo_overview()
    if child_selection.has_active_profiles() is False:
        st.page_link("pages/child_profile.py", label="Go to Child Profiles", icon=":material/person:")
else:
    local_overview()

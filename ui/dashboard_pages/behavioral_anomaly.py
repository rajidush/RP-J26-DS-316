import os

import streamlit as st
from pandas.errors import EmptyDataError

from ui import child_selection, services, theme
from ui.legacy_pages import _behavioral_upgrade


def show_unavailable(message):
    st.error(message)
    st.page_link("dashboard_pages/safety_notifications.py", label="Open Safety Notifications",
                 icon=":material/notifications:")


def render_external_assessment(filename):
    st.caption("Local CSV assessment · Separate from Demo Mode and child profiles.")
    try:
        source = services.behavioral_external_source(filename)
    except EmptyDataError:
        source = None
        st.info("The configured CSV contains no assessment records.")
    except OSError as error:
        source = None
        services.log_assessment_failure("behavioral", "local CSV access", error)
        st.error("The configured CSV could not be opened.")
    except (ValueError, RuntimeError) as error:
        source = None
        services.log_assessment_failure("behavioral", "local CSV validation", error)
        st.error("The configured CSV contains invalid assessment records.")
    if not source:
        child_selection.clear_csv_assessment()
        return
    if not source["child_ids"]:
        child_selection.clear_csv_assessment()
        st.info("The configured CSV contains no assessment records.")
        return
    with theme.card("select-child", "Select CSV Record"):
        key = f"{child_selection.CSV_WIDGET_PREFIX}child_{source['identity']}"
        if st.session_state.get(key) not in source["child_ids"]:
            st.session_state[key] = source["child_ids"][0]
        child = st.selectbox("CSV child identifier", source["child_ids"], index=None, key=key, width=440)
    _behavioral_upgrade.render(source, child=child)


if not child_selection.demo_mode_active():
    local_csv = os.getenv("C4_F2_DAILY_CSV")
    if local_csv:
        theme.page_header("Behavioral Anomaly Detection")
        if st.toggle("Assess configured local CSV", key=child_selection.CSV_ASSESSMENT_WIDGET,
                     on_change=child_selection.clear_csv_assessment):
            render_external_assessment(local_csv)
            st.stop()
    child_selection.clear_csv_assessment()
    child_selection.show_assessment_context("Behavioral Anomaly Detection", show_title=not local_csv)

theme.page_header("Behavioral Anomaly Detection")
try:
    source = services.behavioral_daily_source()
    if source["source"] != "demo":
        raise ValueError("Demo Mode requires the bundled dataset")
except (ImportError, OSError, ValueError, RuntimeError) as error:
    services.log_assessment_failure("behavioral", "record loading", error)
    show_unavailable("Fictional monitoring records could not be loaded. Please try again later.")
    st.stop()
st.caption("Demo Mode · Fictional data for this assessment.")
ids = source["child_ids"]
if not ids:
    st.info("No fictional monitoring records are available.")
    st.stop()

with theme.card("select-child", "Select Child"):
    child = child_selection.behavioral_demo_picker(
        "Select child", ids, child_selection.F2_DEMO_CHILD, index=0,
        label_visibility="collapsed", width=440,
    )

_behavioral_upgrade.render(source, child=child)

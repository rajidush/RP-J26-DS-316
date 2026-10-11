import streamlit as st

from ui import services, theme
from ui.pages._shared import session_details, sessions_dataframe

theme.page_header("Escalations",
                  "Sessions Component 3 flagged for parent review, newest first.")

sessions = services.escalated_sessions()
if not sessions:
    with theme.card("none", "Nothing to review", "No sessions are escalated."):
        pass
    st.stop()

needs = "needs" if len(sessions) == 1 else "need"
with theme.card("escalations-table", f"{theme.plural(len(sessions), 'session')} {needs} review",
                "Click a row to see its details."):
    selected = sessions_dataframe(sessions, key="escalations_table")
if selected:
    session_details(selected)

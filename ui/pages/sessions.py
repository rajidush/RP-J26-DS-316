import streamlit as st

from ui import services, theme
from ui.pages._shared import session_details, sessions_dataframe

theme.page_header("Sessions", "All Component 3 sessions in the database, newest first.")

with theme.card("filters", "Filters"):
    c1, c2, c3, c4 = st.columns(4)
    risk = c1.multiselect("Risk level", services.RISK_LEVELS, default=list(services.RISK_LEVELS))
    emotion = c2.multiselect("Emotional state", services.EMOTIONAL_STATES,
                             default=list(services.EMOTIONAL_STATES))
    escalation = c3.selectbox("Escalated", ["Any", "Yes", "No"])
    dates = c4.date_input("Date range (UTC)", value=(), help="Leave empty for all dates")

start_date = dates[0] if len(dates) >= 1 else None
end_date = dates[1] if len(dates) == 2 else start_date

sessions = services.list_sessions(
    risk_levels=risk, emotional_states=emotion,
    escalation_flag={"Any": None, "Yes": True, "No": False}[escalation],
    start_date=start_date, end_date=end_date,
)

with theme.card("sessions-table", theme.plural(len(sessions), "session"),
                "Click a row to see its details."):
    selected = sessions_dataframe(sessions, key="sessions_table")
if selected:
    session_details(selected)

"""Small display helpers shared by the pages (no DB or model logic here)."""
import streamlit as st

from ui import services, theme


def session_details(session):
    with theme.card("session-details", f"Session {session['session_id']}"):
        _session_details_body(session)


def _session_details_body(session):
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk level", session["risk_level"])
    c2.metric("Emotional state", session["emotional_state"])
    c3.metric("Dialogue turns", session["dialogue_turns"])
    st.write(
        f"**Child:** {session.get('child_id') or 'not recorded'}  \n"
        f"**Timestamp:** {session['timestamp']}  \n"
        f"**Self-regulation shown:** {'Yes' if session['self_regulation_shown'] else 'No'}  \n"
        f"**Escalated for parent review:** {'Yes' if session['escalation_flag'] else 'No'}  \n"
        f"**Received:** {session['received_at']}"
    )
    if session.get("dialogue_summary"):
        st.write(f"**Parent-safe summary:** {session['dialogue_summary']}")


def sessions_dataframe(sessions, key):
    """Render the sessions table; returns the selected session dict or None."""
    table = services.sessions_table(sessions)
    event = st.dataframe(table, hide_index=True, width="stretch", key=key,
                         on_select="rerun", selection_mode="single-row")
    rows = event.selection.rows
    return sessions[rows[0]] if rows else None

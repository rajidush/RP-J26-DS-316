from datetime import datetime, timezone

import streamlit as st

from ui import services, theme

theme.page_header("Add record",
                  "For testing: every record is validated against the Component 3 contract "
                  "before it is saved. Only contract fields are stored.")


def show_result(result, label="Record"):
    if result["ok"]:
        st.success(f"{label} saved (row id {result['id']}).")
    else:
        st.error(f"{label} rejected:\n\n" + "\n".join(f"- {e}" for e in result["errors"]))


form_tab, json_tab = st.container(key="card-add").tabs(["Form", "JSON paste / upload"])

with form_tab:
    with st.form("add_record", clear_on_submit=False, border=False):
        c1, c2 = st.columns(2)
        session_id = c1.text_input("session_id")
        child_id = c2.text_input(
            "child_id (optional)",
            help="Not part of the Component 3 contract; groups sessions per child for "
                 "Function 1. Leave empty if unknown.")
        c1, c2 = st.columns(2)
        day = c1.date_input("Date (UTC)", value=datetime.now(timezone.utc).date())
        at = c2.time_input("Time (UTC)", value=datetime.now(timezone.utc).time().replace(microsecond=0))
        c1, c2, c3 = st.columns(3)
        risk = c1.selectbox("risk_level", services.RISK_LEVELS)
        emotion = c2.selectbox("emotional_state", services.EMOTIONAL_STATES)
        turns = c3.number_input("dialogue_turns", min_value=0, value=1, step=1,
                                help="Must be at least 1")
        c1, c2 = st.columns(2)
        selfreg = c1.checkbox("self_regulation_shown")
        escalated = c2.checkbox("escalation_flag")
        summary = st.text_area("dialogue_summary (optional)",
                               help="Short, redacted, parent-safe summary. Never raw conversation content.")
        submitted = st.form_submit_button("Save")

    if submitted:
        record = {
            "session_id": session_id,
            "timestamp": datetime.combine(day, at, timezone.utc).isoformat(),
            "risk_level": risk,
            "emotional_state": emotion,
            "self_regulation_shown": selfreg,
            "escalation_flag": escalated,
            "dialogue_turns": int(turns),
        }
        if summary.strip():
            record["dialogue_summary"] = summary.strip()
        if child_id.strip():
            record["child_id"] = child_id.strip()
        show_result(services.save_record(record))

with json_tab:
    st.write("Paste one record (a JSON object) or several (a JSON list), or upload a `.json` file.")
    pasted = st.text_area("JSON", height=220, placeholder='{"session_id": "S-1", "timestamp": "2026-07-20T16:19:00Z", ...}')
    uploaded = st.file_uploader("…or upload a JSON file", type=["json"])

    if st.button("Validate and save"):
        text = uploaded.getvalue().decode("utf-8") if uploaded else pasted
        if not text.strip():
            st.warning("Nothing to import.")
        else:
            try:
                records = services.parse_json_records(text)
            except ValueError as exc:
                st.error(str(exc))
            else:
                results = services.import_records(records)
                saved = sum(r["ok"] for r in results)
                st.write(f"**{saved} of {len(results)} record(s) saved.**")
                for r in results:
                    show_result(r, label=f"Record #{r['index'] + 1} ({r['session_id'] or 'no session_id'})")

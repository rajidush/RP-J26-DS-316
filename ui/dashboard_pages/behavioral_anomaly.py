from datetime import timedelta

import streamlit as st

from ui import child_selection, services, theme

child_selection.show_assessment_context("Behavioral Anomaly Detection")
theme.page_header("Behavioral Anomaly Detection")
st.caption("Demo Mode · Fictional data for this assessment.")


def show_unavailable(message):
    st.error(message)
    st.page_link("dashboard_pages/safety_notifications.py", label="Open Safety Notifications",
                 icon=":material/notifications:")
    st.stop()


def show_model_error(status):
    stage = "required assessment files missing" if status.get("missing") else "assessment files could not load"
    services.log_assessment_failure("behavioral", stage)
    show_unavailable("Behavioral assessment is temporarily unavailable. Please try again later.")


try:
    source = services.behavioral_daily_source()
except (ImportError, OSError, ValueError, RuntimeError) as error:
    services.log_assessment_failure("behavioral", "record loading", error)
    show_unavailable("Fictional monitoring records could not be loaded. Please try again later.")
if not source["child_ids"]:
    st.info("No fictional monitoring records are available.")
    st.stop()

f2_status = services.function2_status()
if not f2_status["available"]:
    show_model_error(f2_status)

ids = source["child_ids"]

with theme.card("select-child", "Select Child"):
    child = child_selection.behavioral_demo_picker(
        "Select child", ids, child_selection.F2_DEMO_CHILD, index=0,
        label_visibility="collapsed", width=440,
    )

rows = source["daily"]
rows = rows[rows["child_id"] == child]
if rows.empty:
    st.info("No fictional monitoring records are available for this selection.")
    st.stop()
try:
    if not services.behavioral_observed_days(rows).any():
        st.info("There is insufficient monitoring information to assess unusual activity.")
        st.stop()
    report = services.behavioral_report(rows, child)
except services.ModelUnavailableError as exc:
    show_model_error(exc.status)
except (ImportError, OSError, ValueError, RuntimeError) as error:
    services.log_assessment_failure("behavioral", "assessment execution", error)
    show_unavailable("Behavioral activity could not be assessed. Please try again later.")

table = report["daily"]
latest = table.iloc[-1]
last = latest["date"]
window_start = last - timedelta(days=29)
recent = table[table["date"].between(window_start, last)]
with theme.card("latest-day", "Latest Prediction"):
    theme.flag_status(bool(latest["unusual"]), latest["date"])
    st.metric("Flagged days · latest 30 days", int(recent["unusual"].sum()))
    st.caption(f"{window_start} to {last} · ending on this child's latest recorded day")
    if latest["imputed"]:
        theme.note("Some values were missing and were filled in.")

with theme.card("anomaly-history", "Daily Behavioral Anomaly History"):
    st.plotly_chart(theme.anomaly_history_figure(table),
                    config=theme.PLOTLY_CONFIG, width="stretch")
    theme.screen_reader_text(
        f"Daily behavioral anomaly history for {child} from {table['date'].min()} to {last}. "
        f"{int(table['unusual'].sum())} of {len(table)} recorded days are flagged. "
        "Flagged days are amber diamonds; days not flagged are grey circles. "
        "Each chart point includes its date and status.")

with theme.card("flagged-days", "Top 3 Observed Feature Deviations"):
    st.caption("Observed differences from typical training values; these are not exact model attributions.")
    flagged_dates = list(table.loc[table["unusual"], "date"])
    if not flagged_dates:
        st.write("No flagged days for this child.")
    else:
        day = st.selectbox("Flagged day", flagged_dates[::-1], key=f"f2_day_{child}")
        st.dataframe(services.explanation_table(report["explanations"][day]).head(3),
                     hide_index=True, width="stretch")

with theme.card("urgent-alerts", "Urgent Alerts (rule-based)"):
    st.caption("Independent rules for urgent language, grooming, self-harm or self-harm imagery; separate from behavioral predictions.")
    urgent = report["urgent"]
    if urgent.empty:
        st.write("No urgent rule-based alerts for this child.")
    else:
        st.dataframe(urgent.rename(columns={"reasons": "rule matched"}), hide_index=True,
                     width="stretch")

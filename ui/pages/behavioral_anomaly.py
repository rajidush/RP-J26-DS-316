from datetime import timedelta

import streamlit as st

from ui import services, theme

theme.page_header("Behavioral Anomaly Detection")


def show_model_error(status):
    theme.error_card(
        "f2", "Function 2 model could not be loaded. No anomaly prediction is shown.",
        status["reason"],
        ["Put `behavioral_anomaly_pipeline.pkl` and `behavioral_anomaly_metadata.json` "
         "(the saved Step 3 model) in `component4_profiling_xai/models/`, "
         "or set `C4_F2_ARTIFACT_DIR` to the folder that has them.",
         "Install the scikit-learn version the model was trained with: "
         "`pip install scikit-learn==1.6.1`.",
         "Then restart the app."])


def show_urgent(report):
    with theme.card("urgent-alerts", "Urgent Alerts (rule-based)"):
        st.caption("These rules check recorded safety signals separately from the anomaly model. A rule match does not confirm harm.")
        urgent = report["urgent"]
        if urgent.empty:
            st.write("No urgent rules matched the available records.")
        else:
            st.dataframe(urgent.rename(columns={"reasons": "rule matched"}), hide_index=True,
                         width="stretch")
        if report["urgent_incomplete"]:
            st.caption("Some urgent signals were unavailable or missing from the records.")


try:
    source = services.behavioral_daily_source()
except (OSError, ValueError) as exc:
    st.error(f"Behavioral records could not be loaded: {exc}")
    st.stop()
if source["source"] == "demo":
    st.caption("Research demonstration using synthetic data")
st.caption("Not flagged means the model did not detect an unusual pattern in the available records. "
           "It does not guarantee that the child is safe.")
ids = source["child_ids"]
if not ids:
    st.info("Insufficient recorded information to assess behavior for this period.")
    st.stop()

with theme.card("select-child", "Select Child"):
    child = st.selectbox("Select child", ids, label_visibility="collapsed", width=440,
                         key="f2_child")

report = None
try:
    report = services.behavioral_report(source["daily"], child)
except services.ModelUnavailableError as exc:
    show_model_error(exc.status)
except ValueError as exc:
    st.error(f"Function 2 could not run on this data: {exc}")
if report is None:
    try:
        show_urgent(services.behavioral_urgent_report(source["daily"], child))
    except ValueError as alert_error:
        st.error(f"Urgent records could not be checked: {alert_error}")
    st.stop()

table = report["daily"]
latest = table.iloc[-1]
last = latest["date"]
window_start = last - timedelta(days=29)
recent = table[table["date"].between(window_start, last)]
with theme.card("latest-day", "Latest Prediction"):
    if latest["label"] == "Insufficient data":
        st.write("**Insufficient data**")
        st.write("Insufficient recorded information to assess behavior for this period.")
    else:
        theme.flag_status(bool(latest["unusual"]), latest["date"])
    assessed = int(recent["unusual"].count())
    if assessed:
        st.metric("Flagged days · latest 30 days", int(recent["unusual"].sum()))
    st.caption(f"{window_start} to {last} · ending on this child's latest recorded day")
    if assessed < len(recent):
        st.caption(f"{len(recent) - assessed} recorded days in this period could not be assessed.")
    if latest["imputed"]:
        theme.note("Some values were missing and were filled in.")

with theme.card("anomaly-history", "Daily Behavioral Anomaly History"):
    st.plotly_chart(theme.anomaly_history_figure(table),
                    config=theme.PLOTLY_CONFIG, width="stretch")
    theme.screen_reader_text(
        f"Daily behavioral anomaly history for {child} from {table['date'].min()} to {last}. "
        f"{int(table['unusual'].sum())} of {table['unusual'].count()} assessed days are flagged. "
        "Flagged days are amber diamonds; days not flagged are grey circles. "
        "Days with insufficient data are open squares. "
        "Each chart point includes its date and status.")

with theme.card("flagged-days", "Top 3 Observed Feature Deviations"):
    st.caption("Observed differences from typical training values; these are not exact model attributions.")
    flagged_dates = list(table.loc[table["unusual"], "date"])
    if not table["unusual"].notna().any():
        st.write("No days had enough recorded information for a prediction.")
    elif not flagged_dates:
        st.write("No flagged days for this child.")
    else:
        day = st.selectbox("Flagged day", flagged_dates[::-1], key=f"f2_day_{child}")
        st.dataframe(services.explanation_table(report["explanations"][day]).head(3),
                     hide_index=True, width="stretch")

show_urgent(report)

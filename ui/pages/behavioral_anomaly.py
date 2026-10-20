import streamlit as st

from ui import services, theme

theme.page_header("Behavioral Anomaly (Function 2)",
                  "Days on which a child's screen and language risk events look unusual "
                  "compared with normal days.")

NORMAL_IS_NOT_SAFE = ("Normal means the model found no unusual pattern in these 27 daily "
                      "features. It does not prove that the child is safe, and the model is known "
                      "to miss some concerning patterns (for example, multi-day bullying runs).")


def show_model_error(status):
    """No prediction, chart or explanation without the saved model -- only this card."""
    theme.error_card(
        "f2", "Function 2 model could not be loaded. No anomaly prediction is shown.",
        status["reason"],
        ["Put `behavioral_anomaly_pipeline.pkl` and `behavioral_anomaly_metadata.json` "
         "(the saved Step 3 model) in `component4_profiling_xai/models/`, "
         "or set `C4_F2_ARTIFACT_DIR` to the folder that has them.",
         "Install the scikit-learn version the model was trained with: "
         "`pip install scikit-learn==1.6.1`.",
         "Then restart the app."])
    st.stop()


f2_status = services.function2_status()
if not f2_status["available"]:
    show_model_error(f2_status)

uploaded = st.session_state.get("f2_csv")
try:
    source = services.behavioral_daily_source(uploaded)
except ValueError as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    source = services.behavioral_daily_source()

demo = source["source"] == "demo"
if demo:
    theme.note("SYNTHETIC DATA – DEMO PREDICTIONS. Children C001–C040 and all their events are "
               "generated (behavioral_daily.csv); no real child is involved. These IDs are not "
               "the Function 1 demo children with the same names.")
ids = source["child_ids"]

with theme.card("select-child", "Select Child"):
    child = st.selectbox("Select child", ids, label_visibility="collapsed", width=440,
                         key="f2_child")
    if demo:
        theme.note("Bundled synthetic daily dataset. Components 1 and 2 have no database storage "
                   "yet, so no live data is available.")
    else:
        theme.note(f"Data from the uploaded CSV ({theme.plural(len(source['daily']), 'row')}, "
                   f"{theme.plural(len(ids), 'child', 'children')}).")

try:
    report = services.behavioral_report(source["daily"], child)
except services.ModelUnavailableError as exc:
    show_model_error(exc.status)
except ValueError as exc:
    st.error(f"Function 2 could not run on this data: {exc}")
    st.stop()

table = report["daily"]
first, last = table["date"].min(), table["date"].max()
picked = st.date_input("Days shown (local dates, Asia/Colombo)", value=(first, last),
                       min_value=first, max_value=last, key=f"f2_range_{child}")
# While a range is being picked, Streamlit returns only its first date.
start, end = (tuple(picked) + (last,))[:2] if isinstance(picked, (tuple, list)) else (picked, last)
shown = table[(table["date"] >= start) & (table["date"] <= end)]
if shown.empty:
    st.warning("No days in the selected range.")
    st.stop()

latest = shown.iloc[-1]
with theme.card("latest-day", "Latest Day"):
    theme.status_badge(latest["label"],
                       subtitle=f"{latest['date']} · score {latest['anomaly_score']:.3f} vs "
                                f"threshold {report['threshold']:.3f} "
                                f"(margin {latest['margin']:+.3f})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Days shown", len(shown))
    c2.metric("Unusual days", int(shown["unusual"].sum()))
    c3.metric("Saved threshold", f"{report['threshold']:.3f}")
    theme.note(NORMAL_IS_NOT_SAFE)
    if latest["imputed"]:
        theme.note("This day had missing values; the model filled them with training medians.")

with theme.card("anomaly-history",
                "Daily Anomaly Score" + (" (synthetic demo predictions)" if demo else "")):
    st.plotly_chart(theme.anomaly_history_figure(shown, report["threshold"]),
                    config=theme.PLOTLY_CONFIG, width="stretch")
    theme.note("A day is Unusual when its score is above the dashed threshold (the 97% quantile "
               "of normal training days). Margin = score − threshold.")
    with st.expander("Daily predictions table"):
        st.dataframe(shown[["date", "label", "anomaly_score", "threshold", "margin", "imputed"]]
                     .rename(columns={"label": "prediction", "anomaly_score": "score"})
                     .round(4), hide_index=True, width="stretch")

with theme.card("weekly-summary", "Weekly Summary"):
    weekly = services.weekly_summary(shown)
    st.dataframe(weekly.round(3).rename(columns={
        "week_start": "week (Mon)", "unusual_days": "unusual days",
        "urgent_days": "urgent-rule days", "highest_score": "highest score",
        "mean_score": "mean score", "partial_week": "partial week"}),
        hide_index=True, width="stretch")
    theme.note("Urgent-rule days are counted from the separate rule below, not from the model.")

with theme.card("why-unusual", "Why Was This Day Unusual?"):
    unusual_dates = [d for d in shown.loc[shown["unusual"], "date"]]
    if not unusual_dates:
        st.write("No unusual days in the selected range.")
    else:
        day = st.selectbox("Unusual day", unusual_dates[::-1], key=f"f2_day_{child}")
        st.dataframe(services.explanation_table(report["explanations"][day]),
                     hide_index=True, width="stretch")
        theme.note("These are the observed values furthest from an average normal training day, "
                   "in standard deviations. They describe the day; they are not an exact cause. "
                   "The model combines all 27 features non-linearly.")

with theme.card("urgent-alerts", "Urgent Safety Alerts (rule-based, not the anomaly model)"):
    urgent = report["urgent"]
    urgent = urgent[(urgent["date"] >= start) & (urgent["date"] <= end)]
    if urgent.empty:
        st.write("No urgent-rule alerts in the selected range.")
    else:
        st.dataframe(urgent.rename(columns={"reasons": "rule matched"}), hide_index=True,
                     width="stretch")
    theme.note("A day is urgent when it has any urgent, grooming or self-harm language flag, or "
               "self-harm imagery on screen. This fixed rule is independent of the anomaly score. "
               "Because grooming and self-harm counts are also model inputs, an urgent day is "
               "often Unusual too, but the two outputs are never merged.")

with st.expander("About this model"):
    meta = report["status"]["metadata"]
    for key, value in services.model_card(meta).items():
        st.write(f"**{key}:** {value}")
    theme.note(meta["disclaimer"])
    theme.note("Known limits (Step 4A review): detections are driven mainly by rare categories "
               "(grooming, self-harm, urgent, night violence); multi-day bullying runs and volume "
               "spikes are often missed; very quiet days can score close to the threshold. "
               "An experimental Step 4 model exists for research only and is not used here.")

with st.expander("Use your own data (daily-feature CSV)"):
    st.file_uploader("Daily-feature CSV with child_id, date (YYYY-MM-DD) and the 27 model "
                     "features, e.g. component4_profiling_xai/data/behavioral_daily.csv. "
                     "Label columns and any other extra columns are ignored; nothing is stored.",
                     type=["csv"], key="f2_csv")

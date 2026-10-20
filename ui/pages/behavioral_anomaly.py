from datetime import timedelta

import streamlit as st

from ui import services, theme

theme.page_header("Behavioral Anomaly (Function 2)")


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
ids = source["child_ids"]
if demo:
    st.caption("Demo · Synthetic child data")

with theme.card("select-child", "Select Child"):
    child = st.selectbox("Select child", ids, label_visibility="collapsed", width=440,
                         key="f2_child")

try:
    report = services.behavioral_report(source["daily"], child)
except services.ModelUnavailableError as exc:
    show_model_error(exc.status)
except ValueError as exc:
    st.error(f"Function 2 could not run on this data: {exc}")
    st.stop()

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

with st.expander("About the data and model"):
    meta = report["status"]["metadata"]
    if demo:
        st.markdown("**Data source: synthetic demo data.** Children C001–C040 and all their "
                    "events are generated (`behavioral_daily.csv`); no real child is involved, and "
                    "these results are demo predictions, not findings about real children. These "
                    "IDs are not the Function 1 demo children with the same names. Components 1 "
                    "and 2 have no database storage yet, so no live data is available.")
    else:
        st.markdown(f"**Data source: uploaded CSV** ({theme.plural(len(source['daily']), 'row')}, "
                    f"{theme.plural(len(ids), 'child', 'children')}). The model itself was "
                    "trained on synthetic data only.")
    st.markdown(
        f"**Reading the page.** An amber diamond with \"!\" marks a day the model flagged as "
        f"unusual; a grey circle with \"–\" marks a day it did not flag. A day is flagged when "
        f"its anomaly score is above the saved decision threshold "
        f"({report['threshold']:.4f}, the {meta['threshold_quantile']:.0%} quantile of normal "
        "training days). **Not flagged is not proof that a child is safe**: the model is known "
        "to miss some concerning patterns, such as multi-day bullying runs.")
    st.markdown("**Flagged day details** list the observed values furthest from an average "
                "normal training day, in standard deviations. They describe the day; they are not "
                "an exact cause, because the model combines all 27 features non-linearly.")
    st.markdown("**Urgent alerts** come from a fixed rule (any urgent, grooming or self-harm "
                "language flag, or self-harm imagery on screen), independent of the anomaly score. "
                "Grooming and self-harm counts are also model inputs, so an urgent day is often "
                "flagged too, but the two outputs are never merged.")
    st.write(f"**Latest raw anomaly score:** {latest['anomaly_score']:.4f}")
    st.write("**Calculation:** anomaly score = −pipeline.score_samples(features); "
             "flagged when score > saved threshold. The history chart shows this decision only.")
    for key, value in services.model_card(meta).items():
        st.write(f"**{key}:** {value}")
    st.markdown(f"_{meta['disclaimer']}_")
    st.markdown("Known limits (Step 4A review): detections are driven mainly by rare categories "
                "(grooming, self-harm, urgent, night violence); multi-day bullying runs and volume "
                "spikes are often missed; very quiet days can score close to the threshold. An "
                "experimental Step 4 model exists for research only and is not used here.")

with st.expander("Use your own data (daily-feature CSV)"):
    st.file_uploader("Daily-feature CSV with child_id, date (YYYY-MM-DD) and the 27 model "
                     "features, e.g. component4_profiling_xai/data/behavioral_daily.csv. "
                     "Label columns and any other extra columns are ignored; nothing is stored.",
                     type=["csv"], key="f2_csv")

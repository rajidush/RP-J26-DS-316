from datetime import timedelta
import os
from math import isclose

import plotly.graph_objects as go
import streamlit as st

from ui import child_selection, services, theme
from ui.legacy_pages import _behavioral_upgrade

child_selection.show_assessment_context("Behavioral Anomaly Detection")
theme.page_header("Behavioral Anomaly Detection")
RESEARCH_RESULT = "f2_research_shap_result"
PARENT_ACTIVITY_RESULT = "f2_parent_activity_result"
def show_unavailable(message):
    st.error(message)
    st.page_link("dashboard_pages/safety_notifications.py", label="Open Safety Notifications",
                 icon=":material/notifications:")


def show_model_error(status):
    stage = "required assessment files missing" if status.get("missing") else "assessment files could not load"
    services.log_assessment_failure("behavioral", stage)
    show_unavailable("Behavioral assessment is temporarily unavailable. Please try again later.")


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


def show_shap_result(result, child, day):
    st.write(f"**Synthetic child:** {child} · **Day:** {day}")
    st.write(f"**Saved threshold:** {result['threshold']}")
    if result["score"] is None:
        st.write("No anomaly score or flag because monitoring coverage is insufficient.")
    else:
        st.write(f"**Frozen anomaly score:** {result['score']:.12g}")
        st.write(f"**Anomaly flag:** {'Flagged' if result['flag'] else 'Not flagged'}")
    st.write(f"**Reliability status:** `{result['status']}`")
    st.write("**Gate reasons:** " + (", ".join(result["reasons"]) or "All reliability checks passed."))
    if result["neutral_text"]:
        st.write(result["neutral_text"])
    if not result["references"]:
        return

    st.subheader("Exact group-aware SHAP")
    st.caption("Research diagnostics only. Positive contributions raise the model score "
               "relative to the stated background. Negative contributions lower it.")
    table = [
        {"Group": group, "Contribution": value, "Background size": reference["size"],
         "Seed": reference["seed"]}
        for reference in result["references"] for group, value in reference["groups"].items()
    ]
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"Contribution": st.column_config.NumberColumn(format="%+.6f")})
    first = result["references"][0]
    groups = first["groups"]
    figure = go.Figure(go.Bar(
        x=list(groups.values()), y=list(groups), orientation="h",
        marker_color=["#94601c" if value >= 0 else "#3d6b98" for value in groups.values()],
    ))
    figure.update_layout(
        title=f"Primary background · {first['size']} normal training days · seed {first['seed']}",
        xaxis_title="Signed contribution to anomaly score",
        yaxis={"autorange": "reversed", "automargin": True},
    )
    figure.add_vline(x=0, line_color="#666666")
    st.plotly_chart(figure, config=theme.PLOTLY_CONFIG, width="stretch")


def show_research_shap(source, child, table):
    if not services.research_shap_enabled():
        st.session_state.pop(RESEARCH_RESULT, None)
        return
    with st.expander("Research SHAP preview · synthetic data"):
        st.caption("Local research use only. This explains a model score on synthetic records. "
                   "An anomaly flag does not confirm harm.")
        try:
            context = services.research_shap_context(source)
            day = st.selectbox("Research day", table["date"].tolist()[::-1],
                               key=f"f2_research_day_{child}")
            key = (context, child, day)
            stored = st.session_state.get(RESEARCH_RESULT)
            if stored is not None and stored[0] != key:
                st.session_state.pop(RESEARCH_RESULT)
                stored = None
            if st.button("Explain selected day", key="f2_research_explain"):
                with st.spinner("Checking group attributions and reliability…"):
                    result = services.research_shap_report(source, child, day)
                expected = table.loc[table["date"].eq(day)].iloc[0]
                same_prediction = (
                    result["score"] is None and result["flag"] is None
                    if expected["label"] == "Insufficient data" else
                    result["score"] is not None
                    and isclose(result["score"], expected["anomaly_score"], rel_tol=0, abs_tol=1e-8)
                    and result["flag"] == bool(expected["unusual"])
                )
                if not same_prediction or result["threshold"] != expected["threshold"]:
                    raise ValueError("The verified research result disagrees with the displayed prediction.")
                stored = (key, result)
                st.session_state[RESEARCH_RESULT] = stored
            if stored is not None:
                show_shap_result(stored[1], child, day)
        except (OSError, ValueError) as error:
            st.session_state.pop(RESEARCH_RESULT, None)
            services.log_assessment_failure("behavioral", "research SHAP", error)
            st.warning("Research SHAP preview is unavailable for the selected day.")
        except Exception as error:
            st.session_state.pop(RESEARCH_RESULT, None)
            services.log_assessment_failure("behavioral", "research SHAP", error)
            st.warning("Research SHAP preview could not run for the selected day.")


def show_parent_activity_demo(source, child, table):
    address = st.get_option("server.address")
    if not services.parent_xai_demo_enabled(address):
        st.session_state.pop(PARENT_ACTIVITY_RESULT, None)
        return
    with theme.card("parent-activity-demo", "Your Child's Activity Update"):
        st.caption("SYNTHETIC DEMO PREVIEW · Uses simulated activity for a local demonstration.")
        st.caption("An unusual pattern does not confirm harm.")
        try:
            context = services.parent_xai_demo_context(source, address)
            day = st.selectbox("Activity date", table["date"].tolist()[::-1],
                               key=f"f2_parent_activity_day_{child}")
            key = (context, child, day)
            stored = st.session_state.get(PARENT_ACTIVITY_RESULT)
            if stored is not None and stored[0] != key:
                st.session_state.pop(PARENT_ACTIVITY_RESULT)
                stored = None
            if st.button("Show activity update", key="f2_parent_activity_show"):
                st.session_state.pop(PARENT_ACTIVITY_RESULT, None)
                with st.spinner("Preparing the activity update…"):
                    message = services.parent_activity_update(source, child, day, address)
                stored = (key, message)
                st.session_state[PARENT_ACTIVITY_RESULT] = stored
            if stored is not None:
                if stored[1]:
                    st.write(stored[1])
                else:
                    st.info("An activity update is unavailable for this day.")
        except Exception as error:
            st.session_state.pop(PARENT_ACTIVITY_RESULT, None)
            services.log_assessment_failure("behavioral", "parent activity preview", error)
            st.info("An activity update is unavailable for this day.")


try:
    local_csv = os.getenv("C4_F2_DAILY_CSV") if os.getenv("C4_F2_LEGACY_VIEW") == "1" else None
    source = services.behavioral_daily_source(local_csv) if local_csv else services.behavioral_daily_source()
except (ImportError, OSError, ValueError, RuntimeError) as error:
    st.session_state.pop(RESEARCH_RESULT, None)
    st.session_state.pop(PARENT_ACTIVITY_RESULT, None)
    services.log_assessment_failure("behavioral", "record loading", error)
    show_unavailable("Fictional monitoring records could not be loaded. Please try again later.")
    st.stop()
if source["source"] == "demo":
    st.caption("Demo Mode · Fictional data for this assessment.")
ids = source["child_ids"]
if not ids:
    st.session_state.pop(RESEARCH_RESULT, None)
    st.session_state.pop(PARENT_ACTIVITY_RESULT, None)
    st.info("No fictional monitoring records are available.")
    st.stop()

with theme.card("select-child", "Select Child"):
    child = child_selection.behavioral_demo_picker(
        "Select child", ids, child_selection.F2_DEMO_CHILD, index=0,
        label_visibility="collapsed", width=440,
    )

if os.getenv("C4_F2_LEGACY_VIEW") != "1":
    try:
        upgrade = _behavioral_upgrade.selected_bundle()
    except (OSError, ValueError, KeyError) as error:
        services.log_assessment_failure("behavioral", "deployed model loading", error)
        upgrade = None
    _behavioral_upgrade.render(source, upgrade, child=child)
    st.stop()

st.caption("Not flagged means the model did not detect an unusual pattern in the available records. "
           "It does not guarantee that the child is safe.")
rows = source["daily"]
rows = rows[rows["child_id"] == child]
if rows.empty:
    st.info("No fictional monitoring records are available for this selection.")
    st.stop()
if not services.behavioral_observed_days(rows).any():
    st.info("There is insufficient monitoring information to assess unusual activity.")
    try:
        show_urgent(services.behavioral_urgent_report(source["daily"], child))
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        services.log_assessment_failure("behavioral", "urgent records", error)
        st.error("Urgent records could not be checked.")
    st.stop()
report = None
try:
    report = services.behavioral_report(rows, child)
except services.ModelUnavailableError as exc:
    show_model_error(exc.status)
    report = None
except (ImportError, OSError, ValueError, RuntimeError) as error:
    services.log_assessment_failure("behavioral", "assessment execution", error)
    show_unavailable("Behavioral assessment is temporarily unavailable. Please try again later.")
if report is None:
    st.session_state.pop(RESEARCH_RESULT, None)
    st.session_state.pop(PARENT_ACTIVITY_RESULT, None)
    try:
        show_urgent(services.behavioral_urgent_report(source["daily"], child))
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        services.log_assessment_failure("behavioral", "urgent records", error)
        st.error("Urgent records could not be checked.")
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

show_parent_activity_demo(source, child, table)
show_urgent(report)
if os.getenv("C4_F2_LEGACY_VIEW") == "1":
    show_research_shap(source, child, table)

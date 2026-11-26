"""Qualitative Function 02 view backed by the deployed local detector."""
import logging
from datetime import timedelta

import joblib
import pandas as pd
import sklearn
import streamlit as st

from component4_profiling_xai.src.behavioral_anomaly import assessable_days
from component4_profiling_xai.src.behavioral_explanation_reliability import (
    assess_daily, assess_temporal, daily_evidence, temporal_evidence,
)
from component4_profiling_xai.src.behavioral_profile import (
    ANCHORS, BUNDLE, ProfileConfig, profiles, score_history,
)
from ui import services, theme

logger = logging.getLogger(__name__)

GROUPS = {
    "Screen activity": "the screen activity records",
    "Language activity": "the language monitoring records",
    "Calendar context": "whether it was a weekday or weekend",
}
OBSERVED_LABELS = {
    "screen_incidents": "screen-related incidents",
    "screen_night_incidents": "nighttime screen incidents",
    "screen_escalated_incidents": "screen incidents marked for review",
    "language_flags": "language-related concerns",
    "language_night_flags": "nighttime language concerns",
    "language_bullying_flags": "language signals related to bullying",
}
SAFETY_LABELS = {
    "urgent language flag": "Language that needs urgent review",
    "grooming language flag": "Language that may indicate grooming",
    "self-harm language flag": "Language about self-harm",
    "self-harm imagery on screen": "Images that may show self-harm",
}


@st.cache_resource
def load_bundle(path, modified):
    bundle = joblib.load(path)
    if bundle["sklearn_version"] != sklearn.__version__:
        raise ValueError("The detector needs the scikit-learn version recorded during training.")
    return bundle


def selected_bundle():
    if not BUNDLE.is_file():
        return None
    return load_bundle(BUNDLE, BUNDLE.stat().st_mtime_ns)


def recent_behavior(history, day, minimum):
    """Interpret only records available by the selected date.

    A latched statistic alone can reflect accumulated evidence from earlier
    months. Require a recent onset in this run or repeated recent model flags
    within the uninterrupted run before describing it as a current change.
    """
    through = history.loc[history["date"].le(day)].sort_values("date")
    start = day - timedelta(days=29)
    recent = through.loc[through["date"].ge(start)]
    supported = False
    earlier = None
    for method in ("ewma", "cusum"):
        active = through[f"{method}_active"].fillna(False)
        onsets = through.loc[through[f"{method}_alert"].fillna(False), "date"]
        if not onsets.empty:
            earlier = onsets.max() if earlier is None else max(earlier, onsets.max())
        if active.empty or not active.iloc[-1]:
            continue
        # The detector resets on gaps; also check adjacency of the recorded dates
        # so consecutive true rows cannot join separate active episodes.
        breaks = ~active | pd.to_datetime(through["date"]).diff().dt.days.ne(1)
        run = through.iloc[breaks.to_numpy().nonzero()[0][-1]:]
        run = run.loc[run["date"].ge(start) & run[f"{method}_active"].eq(True)]
        supported |= bool(run[f"{method}_alert"].any() or run["flag"].sum() >= 2)

    flags = recent["flag"].fillna(False).sum()
    if supported:
        summary = "A change in recorded activity continued across recent days."
    elif flags >= 2:
        summary = "Unusual activity was detected on more than one recent day."
    elif flags:
        summary = "An unusual pattern was detected on one recent day."
    elif recent["flag"].count() < minimum:
        summary = "There are too few recent records to describe a pattern."
    else:
        summary = "No unusual patterns were detected in the recent records that could be checked."
    return summary, supported, earlier


def observed_summary(row, details):
    """Describe recorded differences without interpreting statistical significance."""
    baseline = details["median"].reindex(ANCHORS)
    current = row.reindex(ANCHORS)
    comparable = current.notna() & baseline.notna()
    if not comparable.any():
        return "There isn’t enough earlier activity to make a comparison."
    differences = (current - baseline).where(comparable)
    changed = differences[differences.notna() & differences.ne(0)]
    if changed.empty:
        return "The activity we could compare was similar to what was usually recorded."
    largest = changed.abs().idxmax()
    direction = "More" if changed[largest] > 0 else "Fewer"
    return f"{direction} {OBSERVED_LABELS[largest]} were recorded than usual."


def temporal_text(result, active, review=None):
    if result is None or result.status == "unsupported":
        return "An AI explanation of changes across days is unavailable."
    if result.status == "insufficient":
        return "There are too few usable records to explain changes across days."
    if review is not None and review.status == "invalid":
        return "An AI explanation of changes across days is unavailable."
    if review is not None and review.status != "supported":
        return ("A recent change was detected, but the AI could not identify a clear contributing factor."
                if active else "The AI could not identify a clear factor that contributed more strongly across days.")
    if result.status == "stable":
        return ("The recent change alert has no clear accompanying change in the AI’s activity factors."
                if active else "The factors the AI used to assess activity were similar across recent and earlier records.")
    if result.status == "unclear":
        return "The AI could not identify a clear factor that contributed more strongly across days."
    activity = GROUPS[result.group]
    if result.status == "isolated":
        return f"On one recent recorded day, {activity} contributed more strongly to the AI’s unusual-activity assessment."
    message = f"Across several consecutive days, {activity} contributed more strongly to the AI’s unusual-activity assessment."
    if result.detectors:
        message += " A change alert was also active on the selected day."
    return message


def explanation_text(result, flagged, review=None):
    if review is not None:
        if review.status in {"invalid", "insufficient_data"}:
            return "An AI explanation is unavailable for this day."
        if review.status != "supported":
            return ("The AI identified an unusual pattern, but the available information "
                    "does not clearly explain which activity contributed most."
                    if flagged else "The AI could not identify a clear factor behind this day’s assessment.")
        direction = "more unusual" if flagged else "less unusual"
        text = f"The AI assessment found that {GROUPS[review.group]} made this day look {direction}."
        return text + " A second explanation also pointed in that direction." if review.corroborated else text
    if result is None:
        return "An AI explanation is unavailable for this day."
    sign = 1 if flagged else -1
    shap = result["shap"]
    lime = result["lime"]
    supporting = [group for group, value in shap.items() if sign * value > 1e-8]
    if not supporting:
        return "The explanation did not identify a clear factor behind this day’s result."
    strongest = max(supporting, key=lambda group: abs(shap[group]))
    direction = "more unusual" if flagged else "less unusual"
    text = f"The AI assessment found that {GROUPS[strongest]} made this day look {direction}."
    if sign * lime.get(strongest, 0) > 1e-8:
        return text + " A second explanation also pointed in that direction."
    return text


def guidance(row, attribution, sustained):
    positive = {} if not attribution else {key: value for key, value in attribution["shap"].items()
                                           if value > 0}
    focus = max(positive, key=positive.get) if positive else None

    def recorded(field):
        return pd.notna(row[field]) and row[field] > 0

    if focus == "Screen activity" and recorded("screen_night_incidents"):
        return ("Find a calm time to talk about your child’s online evenings.",
                "How have your online evenings been feeling lately?")
    if focus == "Language activity" and recorded("language_flags"):
        return ("Invite your child to talk about recent online conversations.",
                "Was there anything someone said online that you’d like to talk about?")
    if sustained:
        return ("Make time for regular, supportive check-ins.",
                "Has anything changed about how it feels to spend time online lately?")
    if focus == "Screen activity" and recorded("screen_incidents"):
        return ("Offer to talk through any uncomfortable online experiences together.",
                "Did anything you saw online surprise you or make you uncomfortable?")
    return ("Keep an open conversation about your child’s digital experiences.",
            "What did you enjoy online, and was there anything you’d like help with?")


def safety_text(reasons):
    return ", ".join(SAFETY_LABELS[reason.rsplit(" (", 1)[0]]
                     for reason in reasons.split("; "))


def render(source, bundle):
    config = ProfileConfig(**bundle["profile"]) if bundle else ProfileConfig()
    with theme.card("f2-selection"):
        children, dates = st.columns(2)
        with children:
            child = st.selectbox("Child", source["child_ids"], key="f2_child")
        daily = services._behavioral_child_rows(source["daily"], child)
        with dates:
            day = st.selectbox("Activity date", daily["date"].tolist()[::-1], key=f"f2_activity_{child}")
    latest = daily["date"].iloc[-1]
    urgent_report = services.behavioral_urgent_report(daily, child)
    alerts = urgent_report["urgent"]
    for alert in alerts[alerts["date"].isin({day, latest})].itertuples(index=False):
        label = "Latest recorded day" if alert.date == latest else "Selected day"
        st.warning(f"**Urgent safety review · {label} ({alert.date})**  \n"
                   f"{safety_text(alert.reasons)}. Review the record promptly and seek appropriate human support.")

    index = daily.index[daily["date"].eq(day)][0]
    matrix, details = profiles(daily, config)
    if bundle:
        try:
            history = score_history(daily, matrix, bundle)
        except (ValueError, KeyError, FloatingPointError):
            logger.exception("Could not assess behavioral activity")
            bundle = None
    if bundle is None:
        history = daily[["date"]].assign(flag=pd.NA, ewma_active=False, cusum_active=False,
                                         ewma_alert=False, cusum_alert=False)
    flag = history.at[index, "flag"]
    assessed = pd.notna(flag)
    flagged = assessed and bool(flag)
    summary, active, earlier = recent_behavior(history, day, config.minimum)
    temporal_result = temporal_review = None
    if bundle:
        temporal_result, variants = temporal_evidence(history, matrix, bundle, child, day)
        temporal_review = assess_temporal(temporal_result, variants)
    result = None
    daily_review = None
    if assessed:
        score = history.at[index, "score"]
        result, variants = daily_evidence(matrix.loc[[index], bundle["features"]], score, bundle)
        daily_review = assess_daily(result, score, flagged, bundle, variants)

    with theme.card("f2-overview", "Behavior Overview"):
        if flagged:
            st.markdown("**An unusual activity pattern was detected on this day.**")
        elif assessed:
            st.markdown("**No unusual pattern was detected on this day.**")
        else:
            st.markdown("**Not enough information to assess activity on this day.**")
        if active:
            st.caption("The recent records also show a change that continued across days.")

    first, second = st.columns(2)
    with first:
        with theme.card("f2-observed", "Has Anything Changed?"):
            st.write(observed_summary(matrix.loc[index], details.loc[index]))
            st.caption("This compares the recorded activity with your child’s earlier records.")
        with theme.card("f2-recent", "Recent Behavioral Pattern"):
            st.write(summary)
            st.caption(temporal_text(temporal_result, active, temporal_review))
            if earlier is not None and not active:
                st.caption(f"An earlier change was detected on {earlier}. "
                           "Recent records do not establish that it is still continuing.")
    with second:
        with theme.card("f2-xai", "Why Did the AI Say This?"):
            st.write(explanation_text(result, flagged, daily_review))
            st.caption("This explains the AI’s assessment for the selected day.")
        with theme.card("f2-guidance", "How Can You Help?"):
            advice, question = guidance(matrix.loc[index], result, active)
            st.write(advice)
            st.markdown(f"**Try Asking** · “{question}”")

    with theme.card("f2-safety", "Safety Alerts"):
        st.write("Safety alerts are checked separately from the AI activity assessment.")
        if not alerts["date"].eq(day).any():
            st.write("No urgent safety alert was recorded for the selected day.")
        older = alerts.loc[alerts["date"].lt(day) & alerts["date"].ne(latest)]
        if not older.empty:
            with st.expander("Earlier safety alerts"):
                for alert in older.sort_values("date", ascending=False).itertuples(index=False):
                    st.write(f"**{alert.date}** · {safety_text(alert.reasons)}")
    with theme.card("f2-availability", "Data Availability"):
        baseline_counts = details.loc[index, "n"].reindex(ANCHORS)
        st.write("Enough historical records are available." if baseline_counts.ge(config.minimum).all()
                 else "Historical records are limited.")
        start = max(daily["date"].iloc[0], day - timedelta(days=config.days))
        prior_dates = daily["date"].between(start, day)
        has_gaps = int(prior_dates.sum()) < (day - start).days + 1
        available = assessable_days(daily.loc[prior_dates], matrix.loc[prior_dates]).all()
        if has_gaps or not available:
            st.write("Some activity records are missing.")
        st.caption("An unusual pattern does not necessarily indicate harm. "
                   "No detected anomaly does not guarantee safety.")

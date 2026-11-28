"""Qualitative Function 02 view backed by the deployed local detector."""
from datetime import timedelta

import pandas as pd
import streamlit as st

from component4_profiling_xai.src.behavioral_anomaly import assessable_days
from component4_profiling_xai.src.behavioral_adaptive_profile import adaptive_profiles, admission_evidence
from component4_profiling_xai.src.behavioral_explanation_reliability import (
    EXPLANATION_ERRORS, assess_temporal,
)
from component4_profiling_xai.src.behavioral_profile import ANCHORS, ProfileConfig
from component4_profiling_xai.src.behavioral_xai_runtime import ExplanationRequest
from ui import child_selection, safety_summary, services, theme

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
        supported |= bool(run[f"{method}_alert"].any() or run["flag"].eq(True).sum() >= 2)

    if supported:
        summary = "A change in recorded activity continued across recent days."
    elif recent["flag"].count() < minimum:
        summary = "There are too few recent records to tell whether a change continued."
    else:
        earlier_flags = recent.loc[recent["date"].lt(day), "flag"].eq(True).sum()
        if earlier_flags:
            earlier_days = "One earlier day stood out" if earlier_flags == 1 else "Several earlier days stood out"
            summary = f"{earlier_days} to the AI. No continuing change was detected."
        else:
            summary = "No continuing change was detected in the recent records."
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


def adaptive_summary(profile):
    if not profile["available"]:
        return "There’s too little activity recorded for this day to compare it with earlier days."
    if not adaptive_history_ready(profile):
        return "There are too few usable earlier records for a reliable comparison."
    if profile["abrupt"]:
        deviations = dict(zip(ANCHORS, profile["robust_z"], strict=True))
        largest = max(deviations, key=lambda key: abs(deviations[key]))
        direction = "More" if deviations[largest] > 0 else "Fewer"
        return f"{direction} {OBSERVED_LABELS[largest]} were recorded than usual for your child."
    if profile["shift"]:
        return "Your child’s usual recorded pattern shifted over recent days."
    return "This day’s recorded activity was similar to your child’s recent usual pattern."


def adaptive_history_ready(profile):
    return all(pd.notna(value) for value in profile["median"])


def adaptive_update_caption(profile):
    if profile["recent_exclusion"]:
        return "Records that raised a concern were set aside when updating the usual pattern."
    if profile["decision"] == "confirmed_update" and profile["admitted_indices"]:
        return "Recent suitable records were included in the usual activity profile."
    if profile["decision"] == "probation":
        return "This record was set aside while the system checks for a similar pattern on later days."
    return None


def historical_comparison(daily, matrix, details, flags, index, day, config):
    through = daily.loc[daily["date"].le(day)]
    try:
        comparison = adaptive_profiles(through, matrix.loc[through.index], flags)
        profile = comparison.loc[index]
        return adaptive_summary(profile), adaptive_update_caption(profile), adaptive_history_ready(profile)
    except (KeyError, ValueError, FloatingPointError) as exc:
        services.log_assessment_failure("behavioral", "adaptive activity comparison", exc)
    window = through.loc[through["date"].ge(day - timedelta(days=config.days))]
    try:
        available, urgent, concern = admission_evidence(window, matrix.loc[window.index])
        if (available.all() and not urgent.any() and not concern.any(axis=None)
                and flags.reindex(window.index).eq(False).fillna(False).all()):
            return observed_summary(matrix.loc[index], details.loc[index]), None, None
    except (KeyError, ValueError, FloatingPointError) as exc:
        services.log_assessment_failure("behavioral", "fixed activity comparison", exc)
    return "A usual activity comparison is unavailable for this day.", None, None


def temporal_text(result, active, review=None):
    if result is None or result.status == "unsupported":
        return "An AI explanation across days is unavailable."
    if result.status == "insufficient":
        return "There are too few usable records to explain the AI’s assessments across days."
    if review is not None and review.status == "invalid":
        return "An AI explanation across days is unavailable."
    if review is not None and review.status != "supported":
        return "The AI couldn’t identify a clear factor in its recent assessments."
    if result.status == "stable":
        return "The factors in the AI’s assessments were similar across recent and earlier days."
    if result.status == "unclear":
        return "The AI couldn’t identify a clear factor in its recent assessments."
    activity = GROUPS[result.group]
    if result.status == "isolated":
        return f"On one recent day, {activity} made the activity look more unusual to the AI than before."
    message = f"Across several days, {activity} made the activity look more unusual to the AI than before."
    if result.detectors and not active:
        message += " A change alert was also active on the selected day."
    return message


def explanation_text(_result, flagged, review=None):
    if review is None or review.status in {"invalid", "insufficient_data"}:
        return "An AI explanation is unavailable for this day."
    if review.status != "supported":
        return "The AI couldn’t identify a clear factor in this day’s assessment."
    direction = "more unusual" if flagged else "less unusual"
    text = f"In the AI’s assessment, {GROUPS[review.group]} made the recorded activity look {direction}."
    return text + " Both explanation checks pointed to that factor." if review.corroborated else text


def guidance(row, review, sustained, urgent=False):
    def recorded(field):
        value = row.get(field)
        return pd.notna(value) and value > 0

    if urgent:
        return ("Review the safety alert above and seek appropriate human support.",
                None)
    focus = review.group if review is not None and review.status == "supported" else None
    if focus == "Screen activity" and recorded("screen_night_incidents"):
        return ("The AI explanation highlighted screen activity. Ask about online evenings at a calm moment.",
                "What have your evenings online been like?")
    if focus == "Language activity" and recorded("language_flags"):
        return ("The AI explanation highlighted language activity. Ask about recent online conversations.",
                "How have your conversations online been going?")
    if focus == "Screen activity" and recorded("screen_incidents"):
        return ("The AI explanation highlighted screen activity. Offer to talk through online experiences together.",
                "Did anything you saw online surprise you or make you uncomfortable?")
    if recorded("language_bullying_flags"):
        return ("The records include language flagged for possible bullying. Invite your child to talk about it.",
                "How have your online conversations been feeling lately?")
    if recorded("screen_escalated_incidents"):
        return ("A screen activity record was marked for review. Offer to look at it together.",
                "Is there anything you saw online that you’d like to talk about?")
    if sustained:
        return ("Set aside time for a check-in about the recent change.",
                "Has anything felt different online lately?")
    return ("Pick a calm moment to check in about time spent online.",
            "What’s been good or difficult about being online lately?")


def safety_text(reasons):
    return ", ".join(SAFETY_LABELS[reason.rsplit(" (", 1)[0]]
                     for reason in reasons.split("; "))


def render(source, child=None):
    with theme.card("f2-selection"):
        children, dates = st.columns(2)
        with children:
            if child is None:
                child = child_selection.behavioral_demo_picker(
                    "Child", source["child_ids"], child_selection.F2_DEMO_CHILD, index=0,
                )
            else:
                st.write(f"Child: {child}")
        daily = services._behavioral_child_rows(source["daily"], child)
        with dates:
            day = child_selection.behavioral_date_picker(
                "Activity date", daily["date"].tolist()[::-1], f"f2_activity_{child}",
            )
    assessment = child_selection.enhanced_demo_assessment(source, child, day)
    daily = assessment.rows
    index = daily.index[-1]
    selected_rows = daily.loc[[index]]
    review = assessment.urgent_review
    selected_urgent = False
    if review is not None:
        alerts, urgent_available, urgent_complete = review
        alerts = alerts.loc[alerts["is_urgent"]]
        selected_alerts = alerts.loc[alerts["date"].eq(day)]
        selected_urgent = not selected_alerts.empty
        for alert in selected_alerts.itertuples(index=False):
            st.warning(f"**Urgent safety review · Selected day ({alert.date})**  \n"
                       f"{safety_text(alert.reasons)}. Review the record promptly and seek appropriate human support.")

    matrix, details, bundle = assessment.matrix, assessment.details, assessment.bundle
    config = ProfileConfig(**bundle["profile"]) if bundle is not None else ProfileConfig()
    history = assessment.history
    if history is None:
        history = daily[["date"]].assign(flag=pd.NA, ewma_active=False, cusum_active=False,
                                         ewma_alert=False, cusum_alert=False)
    flag = history.at[index, "flag"]
    assessed = pd.notna(flag)
    flagged = assessed and bool(flag)
    summary, active, earlier = recent_behavior(history, day, config.minimum)
    temporal_result = temporal_review = None
    result = daily_review = None
    request = None
    # Temporal explanation requires a scored endpoint in the selected window.
    if assessed:
        try:
            request = ExplanationRequest(bundle)
        except (RuntimeError, *EXPLANATION_ERRORS) as exc:
            services.log_assessment_failure("behavioral", "activity explanation setup", exc)

    if request is not None:
        try:
            temporal_candidate, variants = request.temporal(history, matrix, child, day)
            temporal_review = assess_temporal(temporal_candidate, variants)
            temporal_result = temporal_candidate
        except (RuntimeError, *EXPLANATION_ERRORS) as exc:
            services.log_assessment_failure("behavioral", "activity explanation across days", exc)

        score = history.at[index, "score"]
        row = matrix.loc[[index], bundle["features"]]
        try:
            daily_candidate, variants = request.daily(row, score)
            daily_review = request.review_daily(daily_candidate, score, flagged, variants)
            result = daily_candidate
        except (RuntimeError, *EXPLANATION_ERRORS) as exc:
            services.log_assessment_failure("behavioral", "daily activity explanation", exc)

    with theme.card("f2-overview", "Behavior Overview"):
        if flagged:
            st.markdown("**The AI detected an unusual pattern in this day’s records.**")
        elif assessed:
            st.markdown("**The AI detected no unusual pattern in this day’s records.**")
        elif assessment.status == "failed":
            st.write(safety_summary.BEHAVIOR_FAILED)
        elif assessment.status == "model_unavailable":
            st.write(safety_summary.MODEL_UNAVAILABLE)
        else:
            st.markdown("**The AI couldn’t assess this day with the records available.**")

    first, second = st.columns(2)
    with first:
        with theme.card("f2-observed", "Has Anything Changed?"):
            comparison_text, update_caption, adaptive_ready = (
                historical_comparison(daily, matrix, details, history["flag"], index, day, config)
                if matrix is not None else ("A usual activity comparison is unavailable for this day.", None, None)
            )
            st.write(comparison_text)
            if update_caption:
                st.caption(update_caption)
            st.caption("The comparison uses your child’s own earlier records.")
        with theme.card("f2-recent", "Recent Behavioral Pattern"):
            st.write(summary)
            if bundle is not None and not assessed:
                st.caption("There are too few usable records to explain the AI’s assessments across days.")
            else:
                st.caption(temporal_text(temporal_result, active, temporal_review))
            if earlier is not None and not active:
                st.caption(f"An earlier change was detected on {earlier}. "
                           "Recent records do not establish that it is still continuing.")
    with second:
        with theme.card("f2-xai", "Why Did the AI Say This?"):
            st.write(explanation_text(result, flagged, daily_review))
        with theme.card("f2-guidance", "How Can You Help?"):
            row = matrix.loc[index] if matrix is not None else pd.Series(dtype=float)
            advice, question = guidance(row, daily_review, active, selected_urgent)
            st.write(advice)
            if question:
                st.markdown(f"**Try Asking** · “{question}”")

    with theme.card("f2-safety", "Safety Alerts"):
        st.write("Safety checks use the available records and run separately from the AI assessment.")
        if review is None:
            st.warning(safety_summary.URGENT_FAILED)
        else:
            if not urgent_complete.at[index]:
                st.warning("Safety monitoring was incomplete for the selected day.")
                if not selected_urgent:
                    st.write("The available records contained no urgent alert for this day."
                             if urgent_available.at[index] else safety_summary.URGENT_INSUFFICIENT)
            elif not selected_urgent:
                st.write("No urgent alert was recorded for this day.")
            if urgent_complete.at[index] and not urgent_complete.all():
                st.warning("Safety monitoring was incomplete on other recorded days.")
            older = alerts.loc[alerts["date"].lt(day)]
            if not older.empty:
                with st.expander("Earlier safety alerts"):
                    for alert in older.sort_values("date", ascending=False).itertuples(index=False):
                        st.write(f"**{alert.date}** · {safety_text(alert.reasons)}")
    with theme.card("f2-availability", "Data Availability"):
        if matrix is None:
            st.write("Activity data availability could not be checked.")
            return
        baseline_counts = details.loc[index, "n"].reindex(ANCHORS)
        st.write("Earlier activity records are available." if baseline_counts.ge(config.minimum).all()
                 else "There are few earlier activity records.")
        selected_assessable = assessable_days(selected_rows, matrix.loc[[index]]).at[index]
        st.write("The activity records needed for an AI check are available."
                 if selected_assessable else "This day is missing records needed for an AI check.")
        if adaptive_ready is None:
            st.write("The usual-pattern history could not be checked.")
        elif adaptive_ready:
            st.write("Enough earlier records could be used to establish the usual pattern.")
        else:
            st.write("Too few earlier records could be used to establish the usual pattern.")
        start = max(daily["date"].iloc[0], day - timedelta(days=config.days))
        prior_dates = daily["date"].between(start, day)
        has_gaps = int(prior_dates.sum()) < (day - start).days + 1
        earlier_dates = prior_dates & daily["date"].lt(day)
        available = assessable_days(daily.loc[earlier_dates], matrix.loc[earlier_dates]).all()
        if has_gaps or not available:
            st.write("There are gaps in the recent activity records.")
        st.caption("The AI’s activity assessment cannot tell you whether your child is safe.")

"""Parent-facing wording for the existing Function 1 guidance results."""
from datetime import date
from math import isclose, isfinite

from ui import services


PARENT_PATTERN_LABELS = {
    "Calm / reflective": "Calm and reflective",
    "Defensive": "Guarded responses",
    "Distressed": "Signs of distress",
    "Withdrawn / unclear": "Harder to read",
}
_PATTERN_MEANINGS = {
    "Calm / reflective": "The responses in the {scope} sessions fit a calmer response pattern.",
    "Defensive": "The responses in the {scope} sessions fit a pattern where replies can come across as less open.",
    "Withdrawn / unclear": "The responses in the {scope} sessions fit a pattern where the emotional tone was harder to read.",
    "Distressed": "The responses in the {scope} sessions fit a pattern associated with signs of distress.",
}
_SESSION_DESCRIPTIONS = {
    "p_calm": "were recorded as calm",
    "p_defensive": "were recorded as guarded",
    "p_distressed": "were recorded as showing distress",
    "p_unclear": "were recorded as unclear",
    "selfreg_rate": "included words that can suggest thinking about what to do differently",
}
_PATTERN_QUESTIONS = {
    "Calm / reflective": "What felt good or manageable online recently?",
    "Defensive": "What would you like me to understand about what happened?",
    "Distressed": "Is there anything you’d like me to listen to or help with?",
    "Withdrawn / unclear": "Would you rather talk, write it down, or come back to it later?",
}
_TREND_WORDING = {
    "stable": ("About the same", "At a regular check-in", "",
               "Things looked much the same overall, so a regular check-in makes sense."),
    "worsening": ("More signs your child may need support", "At the next quiet moment",
                  "I’m here to listen.",
                  "Recent sessions suggest checking in sooner may be helpful."),
    "improving": ("Fewer signs your child may need support", "At your next check-in",
                  "I’d like to hear what’s felt easier lately.",
                  "Recent recorded patterns showed fewer signs that your child may need support."),
}
_CHANGE_TREND_WORDING = {
    "stable": "The recorded patterns showed no clear overall shift across the periods compared.",
    "improving": "The overall trend showed fewer signs that your child may need support.",
    "worsening": "The overall trend showed more signs that your child may need support.",
}


def _observation(feature, value, count, scope):
    if feature == "mean_turns":
        return f"Conversations averaged {value:.1f} turns."
    description = _SESSION_DESCRIPTIONS[feature]
    if value == 0:
        return f"None of the {scope} sessions {description}."
    if count and isclose(value * count, round(value * count), abs_tol=1e-9):
        matched = round(value * count)
        verb = description
        if matched == 1:
            verb = description.replace("were ", "was ", 1)
        return f"{matched} out of {count} {scope} sessions {verb}."
    return f"About {value:.0%} of the {scope} sessions {description}."


def parent_guidance_view(report):
    """Present the selected guidance without exposing the distance calculation."""
    guidance = services.parent_guidance(report)
    pattern = report.get("current_pattern")
    label = PARENT_PATTERN_LABELS.get(pattern, "Not enough information yet")
    trend, timing, opening, trend_reason = _TREND_WORDING.get(
        report.get("trend"), ("Not enough history yet", "", "", ""))
    explanation = (report.get("xai") or {}).get("pattern") or {}
    scope = "recent" if report.get("current_since") else "recorded"
    observations = [
        _observation(row["feature"], row["observed_value"], explanation.get("session_count"), scope)
        for row in guidance["supporting_features"][:2]
    ]
    if guidance["personalized"]:
        recommendation = guidance["guidance"].split(". ", 1)[0].rstrip(".") + "."
        question = guidance["question"] or _PATTERN_QUESTIONS[pattern]
        if opening:
            question = f"{opening} {question}"
        first = observations[0].rstrip(".")
        reason = (f"One observation that supported the “{label}” pattern was that "
                  f"{first[0].lower() + first[1:]}. {trend_reason}")
    else:
        recommendation = "Choose a quiet moment and listen at your child’s pace."
        question = "How have things felt online lately?"
        timing = ""
        reason = ("The recorded sessions give us limited information. "
                  "An open question gives your child room to tell you how things have felt.")
    return {"pattern": label, "trend": trend, "observations": observations,
            "recommendation": recommendation, "question": question, "timing": timing,
            "reason": reason}


def parent_xai_view(report):
    """Explain the assigned pattern using the strongest usable session observation."""
    pattern = report.get("current_pattern") if report.get("enough_data") else None
    label = PARENT_PATTERN_LABELS.get(pattern)
    if not label:
        return {
            "pattern": "Not enough information yet",
            "meaning": "There are too few recorded sessions to describe a pattern yet.",
            "observation": None,
            "explanation": "The explanation will appear once there are enough recorded sessions.",
            "takeaway": "The available records give only a limited view of your child’s experience.",
        }

    current_since = report.get("current_since")
    scope = "recent" if current_since else "recorded"
    view = {
        "pattern": label,
        "meaning": _PATTERN_MEANINGS[pattern].format(scope=scope),
        "observation": None,
        "explanation": "There isn’t enough clear supporting information to explain this pattern.",
        "takeaway": "Use this pattern as one piece of context. Your child’s own account can help you understand how they felt.",
    }
    xai = report.get("xai") or {}
    explanation = xai.get("pattern") or {}
    count = explanation.get("session_count")
    period = explanation.get("analysis_period") or {}
    expected_source = "latest_reliable_window" if current_since else "all_sessions"
    if (xai.get("status") != "available"
            or explanation.get("pattern") != pattern
            or explanation.get("equal_distance")
            or type(count) is not int or count <= 0):
        return view
    if period and (period.get("source") != expected_source
                   or (current_since and period.get("start") != current_since)):
        return view

    supporting = []
    for row in explanation.get("feature_contributions") or ():
        feature = row.get("feature")
        value, contribution = row.get("observed_value"), row.get("contribution")
        if (feature in _SESSION_DESCRIPTIONS and isinstance(value, (int, float))
                and 0 <= value <= 1
                and isclose(value * count, round(value * count), rel_tol=0, abs_tol=1e-9)
                and isinstance(contribution, (int, float))
                and isfinite(contribution) and contribution > 1e-12):
            supporting.append(row)
    if not supporting:
        return view

    strongest = max(supporting, key=lambda row: row["contribution"])
    observation = _observation(strongest["feature"], strongest["observed_value"], count, scope)
    view["observation"] = observation
    view["explanation"] = f"{observation} This was one detail that supported the AI’s match."
    return view


def _reliable_windows(report):
    minimum = report.get("min_sessions", services.MIN_SESSIONS_FOR_PATTERN)
    for window in report.get("timeline") or ():
        if (window.get("pattern") not in PARENT_PATTERN_LABELS
                or not isinstance(window.get("sessions"), int)
                or window["sessions"] < minimum):
            continue
        try:
            start = date.fromisoformat(window.get("date"))
        except (ValueError, TypeError):
            continue
        yield start, window


def _change_observations(report, current, earlier):
    xai = report.get("xai") or {}
    explanation = xai.get("pattern") or {}
    period = explanation.get("analysis_period") or {}
    if (xai.get("status") != "available" or explanation.get("pattern") != current["pattern"]
            or explanation.get("equal_distance")
            or explanation.get("session_count") != current["sessions"]
            or (period and (period.get("source") != "latest_reliable_window"
                            or period.get("start") != current["date"]))):
        return []

    earlier_features = report.get("earlier_window_features") or {}
    observations = []
    for row in explanation.get("top_supporting_features") or ():
        feature, value = row.get("feature"), row.get("observed_value")
        contribution = row.get("contribution")
        if (feature not in _SESSION_DESCRIPTIONS
                or not isinstance(value, (int, float)) or not 0 <= value <= 1
                or not isinstance(contribution, (int, float))
                or not isfinite(contribution) or contribution <= 0):
            continue
        recent_text = _observation(feature, value, current["sessions"], "recent")
        previous = earlier_features.get(feature)
        if (earlier and isinstance(previous, (int, float)) and 0 <= previous <= 1
                and not isclose(previous, value, abs_tol=1e-9)):
            recent_text += " " + _observation(feature, previous, earlier["sessions"], "earlier")
        observations.append(recent_text)
        if len(observations) == 2:
            break
    return observations


def emotional_change_view(report, guidance):
    """Compare the first and latest reliable windows and retain the model's trend."""
    windows = sorted(_reliable_windows(report), key=lambda pair: pair[0])
    current = windows[-1][1] if windows else None
    if current and (current["date"] != report.get("current_since")
                    or current["pattern"] != report.get("current_pattern")):
        current = None
    earlier = windows[0][1] if current and windows[0][0] < windows[-1][0] else None
    earlier_label = PARENT_PATTERN_LABELS[earlier["pattern"]] if earlier else "Unavailable"
    current_label = PARENT_PATTERN_LABELS[current["pattern"]] if current else "Unavailable"

    trend = report.get("trend")
    has_trend = trend in _TREND_WORDING
    if not earlier:
        change = "More recorded sessions are needed to tell whether the pattern has changed."
    elif earlier["pattern"] != current["pattern"]:
        change = (f"The first period matched “{earlier_label}”. "
                  f"The latest period matched “{current_label}”.")
    elif all(window["pattern"] == current["pattern"] for _, window in windows):
        change = "The same pattern appeared in all the recorded periods with enough sessions."
    else:
        change = "The first and latest periods matched the same pattern. Some periods in between showed other patterns."

    if len(windows) == 1:
        history = "The chart shows the one recorded period with enough sessions to describe a pattern."
    elif len({window["pattern"] for _, window in windows}) > 1:
        history = "The chart shows how the pattern varied across the recorded two-week periods."
    else:
        history = "The same pattern appears in each recorded two-week period shown."

    return {
        "earlier": earlier_label,
        "current": current_label,
        "status": trend.capitalize() if has_trend else "Insufficient Data",
        "change": change,
        "trend_summary": _CHANGE_TREND_WORDING.get(
            trend, "More recorded periods are needed to describe the overall trend."
        ) if earlier else "",
        "history_summary": history,
        "observations": _change_observations(report, current, earlier if has_trend else None)
        if current else [],
        "recommendation": guidance["recommendation"],
        "question": guidance["question"],
        "timing": guidance["timing"],
    }

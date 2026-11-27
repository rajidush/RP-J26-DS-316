"""Display-ready safety observations from Function 02 daily records."""

from dataclasses import dataclass
from datetime import date
from math import isfinite

from ui import services


URGENT = "Urgent safety concern"
BEHAVIORAL = "Behavioral observation"
NO_RECORDS = "No fictional monitoring records are available for this selection."
BEHAVIOR_FAILED = "The behavioral assessment could not be generated."
URGENT_FAILED = "The urgent safety review could not be completed."

# Present observed deviations using everyday terms. Omit scores and imputed values.
_OBSERVATIONS = {
    "screen_incidents": "screen activity flags",
    "screen_night_incidents": "screen activity flags at night",
    "language_flags": "language flags",
    "language_night_flags": "language flags at night",
    "total_risk_events": "screen and language flags",
    "night_risk_events": "screen and language flags at night",
}


@dataclass(frozen=True)
class Notification:
    date: date
    category: str
    context: str
    description: str
    limitation: str = ""


@dataclass(frozen=True)
class SafetySummary:
    start: date | None = None
    end: date | None = None
    notifications: tuple[Notification, ...] = ()
    urgent_message: str = NO_RECORDS
    behavioral_message: str = NO_RECORDS
    coverage: str = ""
    latest_unusual: bool | None = None
    assessed_dates: frozenset[date] = frozenset()
    urgent_review_dates: frozenset[date] = frozenset()
    complete_urgent_dates: frozenset[date] = frozenset()

    @property
    def period(self):
        if self.start is None:
            return ""
        if self.start == self.end:
            return f"Recorded date: {self.end}"
        return f"Recorded dates: {self.start} to {self.end}"

    def history(self, category=None):
        return tuple(item for item in self.notifications if category is None or item.category == category)


def _observed_difference(explanation):
    if explanation is not None:
        for row in explanation.itertuples():
            label = _OBSERVATIONS.get(row.feature)
            if label and isfinite(row.observed) and isfinite(row.typical) and row.observed != row.typical:
                direction = "more" if row.direction == "higher than typical" else "fewer"
                return f"The records show {direction} {label} than on typical comparison days."
    return ""


def _urgent_notifications(rows):
    try:
        alerts, available, complete = services.behavioral_urgent_review(rows)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        services.log_assessment_failure("safety", "urgent rule review", error)
        return [], URGENT_FAILED, "", frozenset(), frozenset()

    items = [
        Notification(day.date, URGENT, "Function 02 · Urgent safety rules",
                     f"The daily record contained the following flags: {day.reasons}.",
                     "Some information used in the urgent safety review was missing that day."
                     if not complete.at[day.Index] else "")
        for day in alerts.loc[alerts["is_urgent"]].itertuples()
    ]
    if items:
        latest_date = max(item.date for item in items)
        message = f"Most recent urgent safety concern: {latest_date}."
    elif available.any():
        message = "No urgent safety concerns were identified in the available records."
    else:
        message = "There is insufficient monitoring information to check for urgent safety concerns."
    coverage = "" if complete.all() else "Some days are missing information used in the urgent safety review."
    return (items, message, coverage, frozenset(rows.loc[available, "date"]),
            frozenset(rows.loc[complete, "date"]))


def _behavioral_notifications(rows):
    try:
        observed = services.behavioral_observed_days(rows)
        if not observed.any():
            return [], "There is insufficient monitoring information to assess unusual activity.", None, frozenset()
        report = services.behavioral_report(rows, rows.iloc[0]["child_id"])
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        services.log_assessment_failure("behavioral", "safety assessment", error)
        return [], BEHAVIOR_FAILED, None, frozenset()

    assessed_dates = rows.loc[observed, "date"]
    predictions = report["daily"]
    assessed = predictions[predictions["date"].isin(assessed_dates)]
    items = []
    for day in assessed.loc[assessed["unusual"]].itertuples():
        description = "Activity on this day differed from typical patterns."
        difference = _observed_difference(report["explanations"].get(day.date))
        if difference:
            description = f"{description} {difference}"
        items.append(Notification(
            day.date, BEHAVIORAL, "Function 02 · Behavioral Anomaly assessment", description,
            "Some activity measurements were missing that day." if day.imputed else "",
        ))
    latest = assessed[assessed["date"] == rows.iloc[-1]["date"]]
    message = ("The available assessment identified days with unusual activity." if items
               else "No unusual activity was identified on the assessed days.")
    if not observed.all():
        message += " Some recorded days had too little information to assess."
    if assessed["imputed"].any():
        message += " Some assessed days had missing activity measurements."
    latest_unusual = bool(latest.iloc[0]["unusual"]) if not latest.empty else None
    return items, message, latest_unusual, frozenset(assessed["date"])


def summarize(rows):
    """Summarize one child's records for the selected assessment period."""
    if rows.empty:
        return SafetySummary()
    if rows["child_id"].nunique() != 1:
        raise ValueError("Safety notifications require records for one child")
    rows = rows.sort_values("date").reset_index(drop=True)
    urgent, urgent_message, urgent_coverage, reviewed, complete = _urgent_notifications(rows)
    behavioral, behavioral_message, latest, assessed = _behavioral_notifications(rows)
    start, end = rows["date"].iloc[[0, -1]]
    coverage = []
    if rows["date"].nunique() < (end - start).days + 1:
        coverage.append("Some dates in this range have no monitoring record.")
    if urgent_coverage:
        coverage.append(urgent_coverage)
    notifications = tuple(sorted(dict.fromkeys(urgent + behavioral), key=lambda item: item.date, reverse=True))
    return SafetySummary(start, end, notifications, urgent_message, behavioral_message,
                         " ".join(coverage), latest, assessed, reviewed, complete)


def records_for(source, child):
    if source["source"] != "demo" or child not in source["child_ids"]:
        raise ValueError("Choose a fictional Function 02 child")
    daily = source["daily"]
    return daily[daily["child_id"] == child]

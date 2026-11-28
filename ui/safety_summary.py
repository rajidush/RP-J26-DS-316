"""Display-ready safety observations from Function 02 daily records."""

from dataclasses import dataclass
from datetime import date


URGENT = "Urgent safety concern"
BEHAVIORAL = "Behavioral observation"
NO_RECORDS = "No fictional monitoring records are available for this selection."
BEHAVIOR_FAILED = "The behavioral assessment could not be generated."
MODEL_UNAVAILABLE = "The behavioral assessment is temporarily unavailable."
URGENT_FAILED = "The urgent safety review could not be completed."
URGENT_INSUFFICIENT = "There is insufficient monitoring information to check for urgent safety concerns."
LATEST_UNASSESSED = "The latest recorded day had too little information to assess unusual activity."


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


def _urgent_notifications(rows, review):
    if review is None:
        return [], URGENT_FAILED, "", frozenset(), frozenset()
    alerts, available, complete = review

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
        if not complete.all():
            message += " The review was incomplete."
    else:
        message = URGENT_INSUFFICIENT
    coverage = "" if complete.all() else "Some days are missing information used in the urgent safety review."
    return (items, message, coverage, frozenset(rows.loc[available, "date"]),
            frozenset(rows.loc[complete, "date"]))


def _behavioral_notifications(assessment):
    if assessment.status == "model_unavailable":
        return [], MODEL_UNAVAILABLE, None, frozenset()
    if assessment.status == "failed":
        return [], BEHAVIOR_FAILED, None, frozenset()

    history = assessment.history
    observed = history["flag"].notna()
    assessed = history.loc[observed]
    if assessed.empty:
        return [], "There is insufficient monitoring information to assess unusual activity.", None, frozenset()
    imputed = assessment.matrix.loc[:, assessment.bundle["features"]].isna().any(axis=1)
    items = [
        Notification(
            day.date, BEHAVIORAL, "Function 02 · Behavioral Anomaly assessment",
            "Activity on this day differed from typical patterns.",
            "Some activity measurements were missing that day." if imputed.at[day.Index] else "",
        )
        for day in assessed.loc[assessed["flag"].eq(True)].itertuples()
    ]
    latest = history["flag"].iloc[-1]
    message = ("The available assessment identified days with unusual activity." if items
               else "No unusual activity was identified on the assessed days.")
    if not observed.iloc[-1]:
        message = f"{LATEST_UNASSESSED} {message}"
    if not observed.iloc[:-1].all():
        message += " Some recorded days had too little information to assess."
    if imputed.loc[assessed.index].any():
        message += " Some assessed days had missing activity measurements."
    latest_unusual = bool(latest) if observed.iloc[-1] else None
    return items, message, latest_unusual, frozenset(assessed["date"])


def summarize(assessment):
    """Present the enhanced assessment and its independent urgent review."""
    rows = assessment.rows
    if rows.empty:
        return SafetySummary()
    urgent, urgent_message, urgent_coverage, reviewed, complete = _urgent_notifications(rows, assessment.urgent_review)
    behavioral, behavioral_message, latest, assessed = _behavioral_notifications(assessment)
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

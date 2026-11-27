"""Calendar-window summaries of the validated Function 02 demo records."""

from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite
from numbers import Real

import plotly.graph_objects as go

from ui import safety_summary


PERIODS = {"Last 7 days": 7, "Last 30 days": 30}
DEFAULT_PERIOD = "Last 7 days"


@dataclass(frozen=True)
class ActivitySummary:
    start: date
    end: date
    records: int
    language_history: tuple[tuple[date, int | None], ...]
    language_days: int
    language_flags: int | None

    @property
    def period(self):
        return f"{self.start} to {self.end}"

    @property
    def days(self):
        return len(self.language_history)

    @property
    def average_flags(self):
        return self.language_flags / self.language_days if self.language_days else None

    def assessment_counts(self, safety):
        def in_window(dates):
            return {day for day in dates if self.start <= day <= self.end}

        assessed = in_window(safety.assessed_dates)
        reviewed = in_window(safety.urgent_review_dates)
        complete = in_window(safety.complete_urgent_dates)
        behavioral = assessed.intersection(item.date for item in safety.history(safety_summary.BEHAVIORAL))
        urgent = in_window(item.date for item in safety.history(safety_summary.URGENT))
        return AssessmentCounts(len(behavioral) if assessed else None,
                                len(urgent) if urgent or reviewed else None,
                                len(assessed), len(complete))


@dataclass(frozen=True)
class AssessmentCounts:
    unusual_days: int | None
    urgent_days: int | None
    assessed_days: int
    complete_urgent_days: int


def summarize(rows, end, days):
    """Build a full calendar window with None for unobserved flag counts."""
    start = end - timedelta(days=days - 1)
    window = rows.loc[rows["date"].between(start, end)]
    if window["child_id"].nunique() > 1 or window["date"].duplicated().any():
        raise ValueError("Activity records must contain one row per day for one child")
    flags_by_date = {}
    for row in window.reindex(columns=["date", "language_monitor_observed", "language_flags"]).itertuples():
        count = row.language_flags
        if (isinstance(row.language_monitor_observed, Real) and row.language_monitor_observed == 1
                and isinstance(count, Real) and isfinite(count)
                and count >= 0 and count == int(count)):
            flags_by_date[row.date] = int(count)
    dates = (start + timedelta(days=offset) for offset in range(days))
    history = tuple((day, flags_by_date.get(day)) for day in dates)
    total = sum(flags_by_date.values()) if flags_by_date else None
    return ActivitySummary(start, end, len(window), history, len(flags_by_date), total)


def language_history_figure(summary, color):
    labels = [str(day) for day, _ in summary.language_history]
    counts = [count for _, count in summary.language_history]
    figure = go.Figure(go.Bar(x=labels, y=counts, marker_color=color,
                              hovertemplate="%{x}<br>%{y} flags<extra></extra>"))
    zeros = [label for label, count in zip(labels, counts) if count == 0]
    if zeros:
        figure.add_scatter(x=zeros, y=[0] * len(zeros), mode="markers", marker_color=color,
                           hovertemplate="%{x}<br>0 flags<extra></extra>")
    figure.update_layout(showlegend=False, height=290, margin=dict(l=0, r=0, t=12, b=0),
                         xaxis=dict(type="category", tickangle=-45, nticks=10),
                         yaxis=dict(title="Language flags", rangemode="tozero", dtick=1))
    if summary.language_flags == 0:
        figure.update_yaxes(range=[-0.2, 1])
    return figure

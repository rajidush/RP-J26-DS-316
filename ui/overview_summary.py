"""Parent-facing summaries of the existing fictional assessment reports."""

from dataclasses import dataclass

from ui import services
from ui.parent_guidance_view import emotional_change_view, parent_guidance_view


@dataclass(frozen=True)
class Summary:
    message: str
    period: str = ""
    change: str = ""
    related: str = ""


def emotional_summary(source, child):
    report = services.emotional_pattern_report(services.sessions_for_child(source, child), child)
    if not report["enough_data"]:
        return Summary("There are too few fictional sessions to describe an emotional pattern.")

    guidance = parent_guidance_view(report)
    change = emotional_change_view(report, guidance)
    pattern = (report.get("xai") or {}).get("pattern") or {}
    period = pattern.get("analysis_period") or {}
    start = period.get("start") or report.get("current_since")
    end = period.get("end")
    recorded_period = f"Assessment period: {start} to {end}" if start and end and start != end else ""
    if not recorded_period and start:
        recorded_period = f"{'Period started' if not end else 'Assessment date'}: {start}"
    return Summary(
        guidance["pattern"],
        recorded_period,
        change["trend_summary"] if change["status"] != "Insufficient Data" else "",
        guidance["recommendation"] if guidance["observations"] else "",
    )


def behavioral_summary(source, child):
    report = services.behavioral_report(source["daily"], child)
    latest = report["latest"]
    urgent = report["urgent"]
    if urgent.empty:
        safety = "No urgent rules matched the available fictional records."
    else:
        safety = f"Most recent urgent rule match: {urgent['date'].max()}."
    return Summary(
        "The latest recorded day was flagged for unusual activity."
        if latest["unusual"] else "The latest recorded day was not flagged for unusual activity.",
        f"Assessment date: {latest['date']}",
        related=safety,
    )

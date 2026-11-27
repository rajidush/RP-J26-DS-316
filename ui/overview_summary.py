"""Parent-facing summaries of the existing fictional assessment reports."""

from dataclasses import dataclass

from ui import child_selection, services
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
    safety = child_selection.safety_demo_summary(source, child)
    latest = safety.latest_unusual
    message = safety.behavioral_message
    if latest is not None:
        message = ("The latest recorded day was flagged for unusual activity." if latest else
                   "The latest recorded day was not flagged for unusual activity.")
    return Summary(
        message,
        f"Assessment date: {safety.end}" if safety.end else "",
        related=" ".join(filter(None, (safety.urgent_message, safety.period, safety.coverage))),
    )

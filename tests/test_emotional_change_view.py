from copy import deepcopy

import pytest

from ui import services
from ui.parent_guidance_view import emotional_change_view, parent_guidance_view


CALM = "Calm / reflective"
GUARDED = "Defensive"


def report_with_windows(patterns, trend):
    timeline = [{"date": day, "pattern": pattern, "sessions": 4}
                for day, pattern in zip(("2026-07-01", "2026-07-15", "2026-07-29"), patterns)]
    latest = timeline[-1]
    return {
        "enough_data": True,
        "min_sessions": 3,
        "timeline": timeline,
        "current_pattern": latest["pattern"],
        "current_since": latest["date"],
        "trend": trend,
        "earlier_window_features": {"p_defensive": 0, "p_unclear": 0.25},
        "xai": {"status": "available", "pattern": {
            "pattern": latest["pattern"],
            "session_count": 4,
            "top_supporting_features": [
                {"feature": "p_defensive", "observed_value": 0.75, "contribution": 2.5},
                {"feature": "p_unclear", "observed_value": 0, "contribution": 1.5},
                {"feature": "selfreg_rate", "observed_value": 0.25, "contribution": 0.5},
            ],
        }},
    }


@pytest.mark.parametrize("patterns,trend,earlier,current,change_phrase", [
    ((CALM, CALM, GUARDED), "worsening", "Calm and reflective", "Guarded responses", "latest period"),
    ((GUARDED, GUARDED, CALM), "improving", "Guarded responses", "Calm and reflective", "latest period"),
    ((CALM, CALM, CALM), "stable", "Calm and reflective", "Calm and reflective", "same pattern appeared"),
    ((CALM, CALM, GUARDED), "stable", "Calm and reflective", "Guarded responses", "latest period"),
    ((CALM, GUARDED, CALM), "stable", "Calm and reflective", "Calm and reflective", "in between"),
])
def test_summary_keeps_window_patterns_and_trend_separate(patterns, trend, earlier, current,
                                                         change_phrase):
    report = report_with_windows(patterns, trend)
    guidance = parent_guidance_view(report)
    summary = emotional_change_view(report, guidance)
    assert summary["earlier"] == earlier
    assert summary["current"] == current
    assert change_phrase in summary["change"]
    assert summary["status"] == trend.capitalize()
    assert "worsen" not in summary["change"].lower()
    assert "improv" not in summary["change"].lower()
    expected_trend = {
        "stable": "The recorded patterns showed no clear overall shift across the periods compared.",
        "improving": "The overall trend showed fewer signs that your child may need support.",
        "worsening": "The overall trend showed more signs that your child may need support.",
    }
    assert summary["trend_summary"] == expected_trend[trend]
    for field in ("recommendation", "question", "timing"):
        assert summary[field] == guidance[field]


def test_summary_uses_session_counts_and_preserves_recorded_zero():
    report = report_with_windows((CALM, CALM, GUARDED), "worsening")
    summary = emotional_change_view(report, parent_guidance_view(report))
    assert summary["observations"] == [
        "3 out of 4 recent sessions were recorded as guarded. "
        "None of the earlier sessions were recorded as guarded.",
        "None of the recent sessions were recorded as unclear. "
        "1 out of 4 earlier sessions was recorded as unclear.",
    ]


def test_summary_skips_invalid_feature_values_and_keeps_valid_recent_only_observation():
    report = report_with_windows((CALM, CALM, GUARDED), "stable")
    report["earlier_window_features"] = {"p_unclear": float("nan")}
    rows = report["xai"]["pattern"]["top_supporting_features"]
    rows[0].pop("observed_value")
    rows[2]["observed_value"] = 1.5
    summary = emotional_change_view(report, parent_guidance_view(report))
    assert summary["observations"] == ["None of the recent sessions were recorded as unclear."]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_summary_with_too_few_reliable_windows(count):
    report = report_with_windows((CALM, GUARDED, GUARDED), "not enough data")
    report["timeline"] = report["timeline"][-count:] if count else []
    if not count:
        report.update(enough_data=False, current_pattern=None, current_since=None, xai=None)
    summary = emotional_change_view(report, parent_guidance_view(report))
    assert summary["status"] == "Insufficient Data"
    if count == 2:
        assert "same pattern appeared" in summary["change"]
        assert summary["trend_summary"] == "More recorded periods are needed to describe the overall trend."
    else:
        assert summary["change"].startswith("More recorded sessions are needed")
        assert summary["trend_summary"] == ""
    assert (summary["current"] == "Guarded responses") == bool(count)
    assert (summary["earlier"] == "Guarded responses") == (count == 2)
    assert summary["question"] == "How have things felt online lately?"


def test_summary_reports_pattern_change_even_when_overall_trend_is_unavailable():
    report = report_with_windows((CALM, GUARDED), "not enough data")
    summary = emotional_change_view(report, parent_guidance_view(report))
    assert summary["change"] == (
        "The first period matched “Calm and reflective”. "
        "The latest period matched “Guarded responses”."
    )
    assert summary["trend_summary"] == "More recorded periods are needed to describe the overall trend."


def test_summary_with_missing_xai_still_reports_supported_pattern_comparison():
    report = report_with_windows((CALM, CALM, GUARDED), "stable")
    report["xai"] = None
    summary = emotional_change_view(report, parent_guidance_view(report))
    assert summary["status"] == "Stable"
    assert summary["earlier"] == "Calm and reflective"
    assert summary["current"] == "Guarded responses"
    assert summary["observations"] == []


def test_synthetic_c004_summary_preserves_predictions_and_xai():
    report = services.emotional_pattern_report(services.demo_sessions(), "C004")
    original = deepcopy(report)
    summary = emotional_change_view(report, parent_guidance_view(report))
    assert report == original
    assert summary["earlier"] == "Calm and reflective"
    assert summary["current"] == "Guarded responses"
    assert summary["status"] == "Worsening"
    assert summary["observations"][0] == (
        "3 out of 3 recent sessions were recorded as guarded. "
        "None of the earlier sessions were recorded as guarded.")

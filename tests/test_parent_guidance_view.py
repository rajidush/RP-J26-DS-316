import pytest

from ui.parent_guidance_view import parent_guidance_view


@pytest.mark.parametrize("pattern,feature,label,recommendation,question", [
    ("Calm / reflective", "p_calm", "Calm and reflective", "manageable or enjoyable",
     "What felt manageable or good"),
    ("Defensive", "p_defensive", "Guarded responses", "low-pressure",
     "What would you like me to understand"),
    ("Distressed", "p_distressed", "Signs of distress", "quiet moment to listen",
     "Was there a hard moment"),
    ("Withdrawn / unclear", "p_unclear", "Harder to read", "when and how to talk",
     "Would it be easier to talk"),
])
@pytest.mark.parametrize("trend,trend_label,starter,reason", [
    ("stable", "About the same", "", "a regular check-in"),
    ("worsening", "More signs your child may need support", "I’m here to listen.", "checking in sooner"),
    ("improving", "Fewer signs your child may need support",
     "I’d like to hear what’s felt easier lately.", "fewer signs"),
    ("not enough data", "Not enough history yet", "", "limited information"),
])
def test_guidance_view_renders_patterns_and_trends(
    pattern, feature, label, recommendation, question, trend, trend_label, starter, reason,
):
    report = {
        "enough_data": True, "current_pattern": pattern, "current_since": "2026-07-01",
        "trend": trend, "xai": {"status": "available", "pattern": {
            "pattern": pattern, "session_count": 3,
            "top_supporting_features": [
                {"feature": feature, "observed_value": 2 / 3, "contribution": 3.5},
                {"feature": "selfreg_rate", "observed_value": 0, "contribution": 0.75},
                {"feature": "mean_turns", "observed_value": 2.0, "contribution": 0.5},
            ],
        }},
    }
    view = parent_guidance_view(report)
    assert view["pattern"] == label
    assert view["trend"] == trend_label
    assert len(view["observations"]) == 2
    assert view["observations"][0].startswith("2 out of 3 recent sessions")
    assert view["observations"][1].startswith("None of the recent sessions")
    assert reason in view["reason"]
    assert view["reason"].count(".") == 2
    if trend == "not enough data":
        assert view["question"] == "How have things felt online lately?"
        assert view["timing"] == ""
    else:
        assert recommendation in view["recommendation"]
        assert question in view["question"]
        assert view["question"].startswith(starter or question)
        assert label in view["reason"]
        assert "2 out of 3 recent sessions" in view["reason"]


@pytest.mark.parametrize("report", [
    {"enough_data": False},
    {"enough_data": True, "current_pattern": "Defensive", "trend": "stable"},
    {"enough_data": True, "current_pattern": "Defensive", "trend": "stable",
     "xai": {"status": "available", "pattern": {"pattern": "Defensive",
              "top_supporting_features": [{"feature": "p_defensive", "contribution": 1}]}}},
])
def test_guidance_view_is_cautious_when_observations_are_missing(report):
    view = parent_guidance_view(report)
    assert view["observations"] == []
    assert view["question"] == "How have things felt online lately?"
    assert "limited information" in view["reason"]

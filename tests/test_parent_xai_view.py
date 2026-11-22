from copy import deepcopy

import pytest

from ui.parent_guidance_view import parent_xai_view


def xai_report(pattern, rows):
    return {
        "enough_data": True,
        "current_pattern": pattern,
        "current_since": "2026-07-01",
        "xai": {"status": "available", "pattern": {
            "pattern": pattern,
            "session_count": 4,
            "feature_contributions": [
                {"feature": name, "observed_value": value, "contribution": term}
                for name, value, term in rows
            ],
        }},
    }


@pytest.mark.parametrize("pattern,label,meaning", [
    ("Calm / reflective", "Calm and reflective", "calmer response pattern"),
    ("Defensive", "Guarded responses", "come across as less open"),
    ("Withdrawn / unclear", "Harder to read", "emotional tone was harder to read"),
    ("Distressed", "Signs of distress", "associated with signs of distress"),
])
def test_xai_uses_assigned_pattern_and_strongest_valid_support(pattern, label, meaning):
    report = xai_report(pattern, [
        ("p_calm", 0.25, 0.125),
        ("p_defensive", 0.75, -8.0),
        ("selfreg_rate", 0.25, 0.5),
        ("p_distressed", None, 7.0),
    ])
    before = deepcopy(report)
    view = parent_xai_view(report)
    assert view["pattern"] == label
    assert "recent sessions" in view["meaning"]
    assert meaning in view["meaning"]
    assert "one piece of context" in view["takeaway"]
    assert (
        "1 out of 4 recent sessions included words that can suggest thinking about what to do differently"
        in view["explanation"]
    )
    assert "3 out of 4" not in view["explanation"]
    assert "75%" not in view["explanation"]
    assert report == before


def test_positive_contribution_can_support_a_zero_observed_proportion():
    report = xai_report("Calm / reflective", [("p_unclear", 0.0, 0.5)])
    view = parent_xai_view(report)
    assert view["observation"] == "None of the recent sessions were recorded as unclear."
    assert "supported the AI’s match" in view["explanation"]
    assert view["explanation"].startswith(view["observation"])


def test_same_pattern_uses_each_childs_strongest_explainable_observation():
    first = xai_report("Calm / reflective", [
        ("p_calm", 0.75, 0.5), ("selfreg_rate", 0.25, 2.0), ("mean_turns", 3, 4.0),
    ])
    second = xai_report("Calm / reflective", [
        ("p_calm", 0.75, 2.0), ("selfreg_rate", 0.25, 0.5), ("mean_turns", 3, 4.0),
    ])

    first_view = parent_xai_view(first)
    second_view = parent_xai_view(second)

    assert first_view["pattern"] == second_view["pattern"] == "Calm and reflective"
    assert first_view["observation"] == (
        "1 out of 4 recent sessions included words that can suggest thinking about what to do differently."
    )
    assert second_view["observation"] == "3 out of 4 recent sessions were recorded as calm."


@pytest.mark.parametrize("rows", [
    [],
    [("p_calm", 0.75, -0.5), ("p_unclear", 0, 0.0)],
    [("p_calm", None, 1.0), ("p_defensive", 1.5, 4.0)],
    [("p_calm", float("nan"), 1.0), ("p_defensive", 0.5, float("inf"))],
    [("mean_turns", 2, 1.0)],
    [("p_calm", 0.75, 1e-15)],
    [("p_calm", 0.3, 1.0)],
])
def test_xai_falls_back_without_meaningful_positive_observations(rows):
    assert parent_xai_view(xai_report("Calm / reflective", rows))["observation"] is None


def test_xai_does_not_explain_mismatched_or_tied_assignments():
    report = xai_report("Calm / reflective", [("p_calm", 0.75, 1.0)])
    report["current_pattern"] = "Defensive"
    assert parent_xai_view(report)["observation"] is None
    report["current_pattern"] = "Calm / reflective"
    report["xai"]["pattern"]["equal_distance"] = True
    assert parent_xai_view(report)["observation"] is None


def test_xai_does_not_use_evidence_from_a_different_window_with_the_same_pattern():
    report = xai_report("Calm / reflective", [("p_calm", 0.75, 1.0)])
    report["xai"]["pattern"]["analysis_period"] = {
        "source": "latest_reliable_window", "start": "2026-06-17",
    }
    assert parent_xai_view(report)["observation"] is None


def test_xai_uses_recorded_sessions_when_no_recent_window_qualified():
    report = xai_report("Defensive", [("p_defensive", 0.5, 1.0)])
    report["current_since"] = None
    view = parent_xai_view(report)
    assert "recorded sessions" in view["meaning"]
    assert view["observation"] == "2 out of 4 recorded sessions were recorded as guarded."


def test_xai_does_not_describe_a_pattern_with_insufficient_data():
    report = xai_report("Defensive", [("p_defensive", 0.5, 1.0)])
    report["enough_data"] = False
    view = parent_xai_view(report)
    assert view["pattern"] == "Not enough information yet"
    assert view["observation"] is None
    assert "too few recorded sessions" in view["meaning"]

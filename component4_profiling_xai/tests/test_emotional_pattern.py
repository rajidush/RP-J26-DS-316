
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.emotional_pattern import (
    run_emotional_pattern,
    run_dashboard_outputs,
)


def make_mock():
    return pd.DataFrame({
        "child_id": ["M001"] * 4 + ["M002"] * 4,
        "session_id": [f"M-{i:06d}" for i in range(1, 9)],
        "timestamp": pd.date_range(
            "2026-10-01", periods=8, freq="D"
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "risk_level": ["low"] * 8,
        "emotional_state": [
            "calm", "calm", "calm", "defensive",
            "unclear", "unclear", "unclear", "distressed"
        ],
        "self_regulation_shown": [
            "True", "True", "False", "False"
        ] + ["False"] * 4,
        "escalation_flag": ["False"] * 8,
        "dialogue_turns": [2, 2, 2, 2, 4, 4, 3, 2],
        "dialogue_summary": ["mock"] * 8,
    })


def test_mock_patterns():
    result = run_emotional_pattern(make_mock())
    assert result.loc["M001", "pattern"] == "Calm / reflective"
    assert result.loc["M002", "pattern"] == "Withdrawn / unclear"


def test_unknown_label_rejected():
    bad = make_mock()
    bad.loc[0, "emotional_state"] = "happy"

    with pytest.raises(ValueError):
        run_emotional_pattern(bad)


def test_missing_column_rejected():
    bad = make_mock().drop(columns=["dialogue_turns"])

    with pytest.raises(ValueError):
        run_emotional_pattern(bad)


def test_trend_output_shape():
    patterns, trend = run_dashboard_outputs(make_mock())

    assert set(trend) == {"M001", "M002"}

    for child in trend.values():
        assert child["trend"] in {
            "stable",
            "improving",
            "worsening",
            "not enough data",
        }
        assert "overall_pattern" in child


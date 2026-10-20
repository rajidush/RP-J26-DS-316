import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.urgent_alerts import URGENT_RULE_COLUMNS, urgent_alerts

DATA = Path(__file__).resolve().parents[1] / "data" / "behavioral_daily.csv"


def rows(n=1, **counts):
    df = pd.DataFrame({"child_id": ["K1"] * n, "date": [f"2026-07-{i + 1:02d}" for i in range(n)],
                       **{c: [0] * n for c in URGENT_RULE_COLUMNS}})
    for col, value in counts.items():
        df[col] = value
    return df


def test_quiet_day_is_not_urgent():
    out = urgent_alerts(rows())
    assert list(out.columns) == ["child_id", "date", "is_urgent", "reasons"]
    assert not out["is_urgent"].any() and out["reasons"].iloc[0] == ""


@pytest.mark.parametrize("column", list(URGENT_RULE_COLUMNS))
def test_each_rule_condition_alone_triggers(column):
    out = urgent_alerts(rows(**{column: 2}))
    assert out["is_urgent"].iloc[0]
    assert out["reasons"].iloc[0] == f"{URGENT_RULE_COLUMNS[column]} (2)"


def test_several_reasons_are_joined_in_rule_order():
    out = urgent_alerts(rows(language_self_harm_flags=1, language_urgent_flags=3))
    assert out["reasons"].iloc[0] == "urgent language flag (3); self-harm language flag (1)"


def test_nan_means_not_available_and_does_not_trigger():
    out = urgent_alerts(rows(screen_self_harm_imagery_incidents=np.nan))
    assert not out["is_urgent"].iloc[0]


def test_missing_rule_column_is_an_error():
    with pytest.raises(ValueError, match="missing urgent-rule columns"):
        urgent_alerts(rows().drop(columns="language_grooming_flags"))


def test_rule_ignores_labels_and_scores():
    base = rows(3, language_grooming_flags=[0, 1, 0])
    noisy = base.assign(is_anomaly=[1, 0, 1], anomaly_type="bullying_run", anomaly_score=99.0)
    pd.testing.assert_frame_equal(urgent_alerts(base), urgent_alerts(noisy))


@pytest.mark.skipif(not DATA.is_file(), reason="dataset not present")
def test_matches_the_step4a_rule_on_the_bundled_dataset():
    d = pd.read_csv(DATA)
    step4a = ((d.language_urgent_flags > 0) | (d.language_grooming_flags > 0)
              | (d.language_self_harm_flags > 0) | (d.screen_self_harm_imagery_incidents > 0))
    out = urgent_alerts(d)
    assert (out["is_urgent"].to_numpy() == step4a.to_numpy()).all()
    assert out["is_urgent"].sum() > 0

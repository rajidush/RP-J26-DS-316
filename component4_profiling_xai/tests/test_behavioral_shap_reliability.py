from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import behavioral_shap_reliability as reliability
from src.behavioral_anomaly import FEATURE_SET_A, feature_matrix, predict_anomalies
from src.behavioral_shap import BehavioralShap, mask_violations
from src.urgent_alerts import urgent_alerts

SCREEN, LANGUAGE, CALENDAR = reliability.GROUPS


@pytest.fixture(scope="module")
def daily():
    return pd.read_csv(ROOT / "data" / "behavioral_daily.csv")


@pytest.fixture(scope="module")
def gate(daily):
    return reliability.ReliabilityGate(BehavioralShap(), daily)


def record(daily, child, date):
    rows = daily.loc[daily.child_id.eq(child) & daily.date.eq(date)]
    assert len(rows) == 1
    return rows


def attributions(screen, language, calendar=0.0):
    return dict(zip((SCREEN, LANGUAGE, CALENDAR), (screen, language, calendar), strict=True))


def test_clear_leader_separates_from_competitors_and_matches_deletion():
    vectors = [attributions(0.9, -0.13), attributions(0.8, -0.2), attributions(1.2, -0.3)]
    leader, reasons, metrics = reliability.attribution_checks(vectors, [attributions(0.8, -0.2)])
    assert leader == SCREEN
    assert not reasons
    assert metrics["separation"] == pytest.approx(0.5)
    assert all(metrics["direction_agreement"].values())


@pytest.mark.parametrize(("vectors", "deletion", "reason"), [
    ([attributions(0.8, 0.4), attributions(0.3, 0.9)], attributions(1, 0.2), "leading_group_changed"),
    ([attributions(0.8, 0.7), attributions(0.5, 0.4)], attributions(1, 0.2), "attribution_envelopes_overlap"),
    ([attributions(0.8, 0.4), attributions(-0.8, 0.4)], attributions(1, 0.2), "leading_direction_unstable"),
    ([attributions(0.8, 0.4)] * 2, attributions(0.3, 0.9), "donor_deletion_disagrees"),
])
def test_unsteady_or_weak_evidence(vectors, deletion, reason):
    _, reasons, _ = reliability.attribution_checks(vectors, [deletion])
    assert reason in reasons


def test_exact_support_requires_distinct_training_children(daily):
    row = daily.loc[daily.screen_incidents.gt(0) & daily.language_flags.gt(0)].head(1)
    features = feature_matrix(row, FEATURE_SET_A)
    assert not mask_violations(features).any(axis=None)
    repeated = pd.concat([row, row], ignore_index=True)
    rule = reliability.EmpiricalSupport(repeated)
    assert not any(flags[0] for flags in rule.check(features).values())
    repeated["child_id"] = ["child_a", "child_b"]
    rule = reliability.EmpiricalSupport(repeated)
    assert all(flags[0] for flags in rule.check(features).values())
    other_calendar = features.assign(is_weekend=1 - features["is_weekend"])
    assert not any(flags[0] for flags in rule.check(other_calendar).values())


def test_near_threshold_and_neutral_fallback(gate):
    independent = pd.read_csv(ROOT / "data" / "step4a_heldout_seed20261020" / "behavioral_daily.csv")
    row = record(independent, "C007", "2026-09-13")
    result = gate.review(row)
    assert result.is_flagged and result.status == reliability.Reliability.UNCERTAIN
    assert "cutoff_margin_within_background_span" in result.reasons
    assert "leading_group_changed" in result.reasons
    assert result.text == reliability.FALLBACK
    assert not any(group in result.text for group in reliability.GROUPS)


def test_sparse_but_structurally_valid_observation_uses_fallback(daily, gate):
    row = record(daily, "C033", "2026-09-25").assign(screen_incidents=1000)
    row["total_risk_events"] = row["screen_incidents"] + row["language_flags"]
    assert not mask_violations(feature_matrix(row, FEATURE_SET_A)).any(axis=None)
    result = gate.review(row)
    assert result.is_flagged and result.status == reliability.Reliability.UNCERTAIN
    assert "target_context_sparse" in result.reasons
    assert "explanation_donors_sparse" in result.reasons
    assert result.text == reliability.FALLBACK


def test_missing_coverage_and_unflagged_skip_shap(daily, gate, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("suppressed parent explanations must skip SHAP")

    monkeypatch.setattr(gate.explainer, "explain", unexpected)
    unflagged = record(daily, "C002", "2026-09-18")
    result = gate.review(unflagged)
    assert result.status == reliability.Reliability.NOT_FLAGGED
    assert result.is_flagged is False and result.text is None
    insufficient = gate.review(unflagged.assign(language_monitor_observed=False))
    assert insufficient.status == reliability.Reliability.INSUFFICIENT_DATA
    assert insufficient.score is None and insufficient.is_flagged is None and insufficient.text is None


def test_urgent_rule_is_independent_and_model_prediction_is_unchanged(daily, gate):
    row = record(daily, "C033", "2026-09-25")
    changed = row.assign(screen_self_harm_imagery_incidents=1)
    assert not urgent_alerts(row).iloc[0].is_urgent
    assert urgent_alerts(changed).iloc[0].is_urgent
    first, second = gate.review(row), gate.review(changed)
    assert first == second
    expected = predict_anomalies(changed).iloc[0]
    assert second.score == expected.anomaly_score
    assert second.is_flagged == expected.is_flagged
    assert "imagery" not in second.text


def test_supported_stable_case_names_only_its_group(daily, gate):
    row = record(daily, "C008", "2026-09-15")
    decision = gate.review(row)
    assert decision.status == reliability.Reliability.RELIABLE
    assert decision.is_flagged and not decision.reasons
    assert decision.diagnostics["stability"]["leader"] == SCREEN
    assert f"{SCREEN} increased" in decision.text
    assert LANGUAGE not in decision.text and CALENDAR not in decision.text
    assert len(decision.diagnostics["support_restricted_groups"]) == 4
    assert all(item["retained_children"] >= 2 for item in decision.diagnostics["reference_support"])

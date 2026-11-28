import hashlib
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import behavioral_anomaly as anomaly
from src import behavioral_shap as xai


@pytest.fixture(scope="module")
def daily():
    return pd.read_csv(ROOT / "data" / "behavioral_daily.csv")


@pytest.fixture(scope="module")
def explainer():
    return xai.BehavioralShap()


@pytest.fixture(scope="module")
def examples(daily, explainer):
    test = daily.loc[daily["split"].eq("test")]
    scores = anomaly.predict_anomalies(test)["anomaly_score"]
    return test.loc[[scores.idxmax()]], test.loc[[scores.idxmin()]]


def test_background_is_reproducible_and_only_eligible_train(daily):
    first = xai.select_background(daily, 32, 17)
    replay = xai.select_background(daily.sample(frac=1, random_state=10), 32, 17)
    assert first.identifiers == replay.identifiers
    pd.testing.assert_frame_equal(first.values, replay.values)
    assert len(set(first.identifiers)) == 32
    assert len({child for child, _ in first.identifiers}) == 32
    second = xai.select_background(daily, 64, 29)
    assert first.identifiers != second.identifiers and len(second.values) == 64
    pool = xai.normal_training_rows(daily)
    assert len(pool) == 1894
    keys = set(pool[["child_id", "date"]].itertuples(index=False, name=None))
    assert set(second.identifiers) <= keys
    with pytest.raises(ValueError, match="background size"):
        xai.select_background(daily, 0, 17)


@pytest.mark.parametrize("method", ["kernel", "group_aware"])
def test_explains_saved_score_and_preserves_threshold(daily, explainer, examples, method):
    if method == "kernel":
        pytest.importorskip("shap")
    ref = xai.select_background(daily, 32, 17)
    for row in examples:
        expected = anomaly.predict_anomalies(row).iloc[0]
        result = explainer.explain(row[row.columns[::-1]], ref, method=method, nsamples=128)
        assert result.anomaly_score == expected.anomaly_score
        assert result.threshold == -12.515804087687126
        assert result.is_anomaly == expected.is_flagged
        assert result.decision_margin == result.anomaly_score - result.threshold
        assert result.reconstructs
        assert result.expected_score == pytest.approx(
            anomaly.anomaly_scores(explainer.pipeline, ref.values).mean()
        )
        if method == "kernel":
            assert list(result.feature_values) == explainer.metadata["features"]
            assert sum(result.feature_values.values()) == pytest.approx(
                result.anomaly_score - result.expected_score
            )
            assert result.invalid_rows > 0
        else:
            assert result.invalid_rows == 0
            assert result.model_rows == 1 + 8 * len(ref.values)


def test_kernel_direction_with_controlled_scoring_function(monkeypatch, daily, explainer, examples):
    pytest.importorskip("shap")
    row = examples[0].assign(screen_incidents=100, language_flags=100)
    background = xai.select_background(daily, 32, 17)

    def linear_score(pipeline, frame):
        return 2 * frame["screen_incidents"].to_numpy() - 3 * frame["language_flags"].to_numpy()

    monkeypatch.setattr(xai, "anomaly_scores", linear_score)
    result = explainer.explain(row, background, nsamples=128)
    for feature, weight in (("screen_incidents", 2), ("language_flags", -3)):
        difference = row[feature].iloc[0] - background.values[feature].mean()
        assert difference > 0
        assert result.feature_values[feature] == pytest.approx(weight * difference)


def test_group_assignment_and_masking(daily, examples):
    assigned = [feature for group in xai.AGGREGATE_GROUPS.values() for feature in group]
    assert sorted(assigned) == sorted(anomaly.FEATURE_SET_A)
    row = anomaly.feature_matrix(examples[0], anomaly.FEATURE_SET_A)
    ref = xai.select_background(daily, 32, 17)
    group = next(iter(xai.GROUPS))
    masked = xai.BehavioralShap._replace_groups(row, ref.values, [group])
    assert not xai.mask_violations(masked).any(axis=None)
    broken = masked.assign(total_risk_events=-1)
    assert xai.mask_violations(broken)["total_risk_events"].all()


def test_invalid_input_and_coverage_gate(daily, explainer, examples):
    row = examples[0]
    ref = xai.select_background(daily, 32, 17)
    with pytest.raises(ValueError, match="one daily record"):
        explainer.explain(pd.concat([row, row]), ref)
    with pytest.raises(ValueError, match="missing feature"):
        explainer.explain(row.drop(columns="is_weekend"), ref)
    with pytest.raises(ValueError, match="invalid background"):
        explainer.explain(row, replace(ref, values=ref.values[ref.values.columns[::-1]]))
    with pytest.raises(ValueError, match="consistent daily"):
        explainer.explain(row.assign(total_risk_events=99999), ref, method="group_aware")
    with pytest.raises(ValueError, match="finite values"):
        explainer.explain(row.assign(screen_flagged_seconds=np.inf), ref)
    for invalid in (row.assign(language_monitor_observed=0), row.assign(screen_flagged_seconds=np.nan)):
        assert explainer.explain(invalid, ref) is None
        assert pd.isna(anomaly.predict_anomalies(invalid).iloc[0].is_flagged)
    assert xai.format_parent_explanation([None], {}) is None


def test_stability_and_parent_grounding(daily, explainer, examples):
    row = examples[0]
    first, second = (explainer.explain(row, xai.select_background(daily, 32, seed),
                                       method="group_aware") for seed in (17, 29))
    groups = list(xai.GROUPS)
    # Isolate formatter validation with controlled contribution vectors that reconstruct.
    values = dict(zip(groups, (1.0, -0.5, 0.001)))
    first, second = (replace(e, group_values=values) for e in (first, second))
    assert xai.top_overlap(values, values, 2) == 1
    assert xai.top_overlap({"a": 1, "b": 0}, {"a": 0, "b": 1}, 1) == 0
    text = xai.format_parent_explanation([first, second], values)
    assert f"{groups[0]} increased" in text
    assert f"{groups[1]} decreased" in text
    assert groups[2] not in text
    assert xai.format_parent_explanation([first], values) == xai.FALLBACK
    assert xai.format_parent_explanation([first, first], values) == xai.FALLBACK
    assert xai.format_parent_explanation([first, second], {}) == xai.FALLBACK
    assert xai.format_parent_explanation(
        [first, replace(second, reconstruction_error=0.01)], values
    ) == xai.FALLBACK
    assert xai.format_parent_explanation(
        [first, replace(second, input_digest="different")], values
    ) == xai.FALLBACK
    assert xai.format_parent_explanation([first, second], {key: -v for key, v in values.items()}) == xai.FALLBACK
    assert xai.format_parent_explanation([replace(first, is_anomaly=False)], values) is None


def test_saved_artifacts_unchanged():
    hashes = {
        anomaly.PIPELINE_FILE: "25634fbafabd133495bc66b9ced5921f27ddaac222902d72f38a5fe1515774b7",
        anomaly.METADATA_FILE: "3f5f0fa72eb5ac1cb19644c36d9a96e063a4aa2855515028c97293b406f375a4",
    }
    for name, expected in hashes.items():
        assert hashlib.sha256((anomaly.ARTIFACT_DIR / name).read_bytes()).hexdigest() == expected

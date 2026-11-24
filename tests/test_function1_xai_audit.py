"""Check Function 1 against the artifacts without using its feature or trend helpers."""
from collections import Counter
from itertools import product
import json
from math import fsum

import joblib
import pandas as pd
import pytest

from ui import services
from ui.parent_guidance_view import emotional_change_view, parent_guidance_view, parent_xai_view


COMPONENT = services.ROOT / "component4_profiling_xai"
EMOTIONS = ("calm", "defensive", "distressed", "unclear")
FEATURES = ("p_calm", "p_defensive", "p_distressed", "p_unclear", "selfreg_rate", "mean_turns")
TREND_LABELS = {
    "improving": "Improving", "worsening": "Worsening",
    "stable": "Stable", "not enough data": "Insufficient Data",
}
PATTERN_LABELS = {
    "Calm / reflective": "Calm and reflective", "Defensive": "Guarded responses",
    "Withdrawn / unclear": "Harder to read", "Distressed": "Signs of distress",
}


@pytest.fixture(scope="module")
def artifacts():
    path = COMPONENT / "models"
    metadata = json.loads((path / "emotional_pattern_metadata.json").read_text())
    model = joblib.load(path / "emotional_pattern_kmeans.pkl")
    scaler = joblib.load(path / "emotional_pattern_scaler.pkl")
    assert tuple(metadata["features"]) == tuple(scaler.feature_names_in_) == FEATURES
    return model, scaler, metadata


def reference_prediction(records, artifacts):
    model, scaler, metadata = artifacts
    counts = Counter(row.emotional_state for row in records)
    count = len(records)
    raw = [counts[emotion] / count for emotion in EMOTIONS] + [
        sum(row.self_regulation_shown for row in records) / count,
        fsum(row.dialogue_turns for row in records) / count,
    ]
    standardized = [(value - mean) / scale for value, mean, scale in
                    zip(raw, scaler.mean_, scaler.scale_)]
    distances = [fsum((value - center) ** 2 for value, center in zip(standardized, centroid))
                 for centroid in model.cluster_centers_]
    assigned, alternative = sorted(range(len(distances)), key=distances.__getitem__)[:2]
    assert model.predict([standardized])[0] == assigned
    # Use the difference-of-squares identity as an independent calculation.
    terms = [(a - b) * (2 * value - a - b) for value, a, b in zip(
        standardized, model.cluster_centers_[assigned], model.cluster_centers_[alternative])]
    return metadata["cluster_names"][str(assigned)], assigned, alternative, raw, distances, terms


def reference_windows(records, artifacts):
    settings = artifacts[2]["trend_settings"]
    days = settings["window_days"]
    first = min(pd.Timestamp(row.timestamp).date() for row in records)
    bins = {}
    for row in records:
        index = (pd.Timestamp(row.timestamp).date() - first).days // days
        bins.setdefault(index, []).append(row)
    return [
        ((first + pd.Timedelta(days=index * days)).isoformat(), group,
         reference_prediction(group, artifacts)[0])
        for index, group in sorted(bins.items()) if len(group) >= settings["min_sessions"]
    ]


def assert_result(frame, child, artifacts):
    records = list(frame.loc[frame.child_id == child].itertuples(index=False))
    windows = reference_windows(records, artifacts)
    selected = windows[-1][1] if windows else records
    pattern, assigned, alternative, raw, distances, terms = reference_prediction(selected, artifacts)
    report = services.emotional_pattern_report(frame, child)
    assert report["current_pattern"] == pattern
    assert report["timeline"] == [{"date": date, "sessions": len(group), "pattern": label}
                                  for date, group, label in windows]
    xai = report["xai"]["pattern"]
    assert xai["assigned_cluster"] == assigned
    assert xai["alternative_cluster"] == alternative
    assert xai["session_count"] == len(selected)
    assert xai["assigned_squared_distance"] == pytest.approx(distances[assigned], abs=1e-12)
    assert xai["alternative_squared_distance"] == pytest.approx(distances[alternative], abs=1e-12)
    rows = xai["feature_contributions"]
    assert [row["feature"] for row in rows] == list(FEATURES)
    assert [row["observed_value"] for row in rows] == pytest.approx(raw, abs=1e-12)
    assert [row["contribution"] for row in rows] == pytest.approx(terms, abs=1e-12)
    assert fsum(row["contribution"] for row in rows) == pytest.approx(xai["distance_margin"], abs=1e-12)
    assert xai["distance_margin"] == pytest.approx(distances[alternative] - distances[assigned], abs=1e-12)
    supporting = xai["top_supporting_features"]
    assert all(row in rows and row["contribution"] > 0 for row in supporting)
    expected_support = sorted((term for term in terms if term > 0), reverse=True)[:3]
    assert [row["contribution"] for row in supporting] == pytest.approx(expected_support, abs=1e-12)

    metadata = artifacts[2]
    severity = [metadata["severity_order"][label] for _, _, label in windows]
    trend = report["xai"]["trend"]
    assert trend["reliable_window_count"] == len(windows)
    if len(severity) < metadata["trend_settings"]["min_valid_windows"]:
        status = "not enough data"
        assert trend["earlier_mean"] is trend["recent_mean"] is trend["severity_change"] is None
    else:
        earlier, recent = fsum(severity[:2]) / 2, fsum(severity[-2:]) / 2
        delta = recent - earlier
        threshold = metadata["trend_settings"]["change_threshold"]
        status = "improving" if delta <= -threshold else "worsening" if delta >= threshold else "stable"
        assert (trend["earlier_mean"], trend["recent_mean"], trend["severity_change"]) == (
            earlier, recent, delta)
        assert [row["window_start"] for row in trend["earlier_windows"]] == [w[0] for w in windows[:2]]
        assert [row["window_start"] for row in trend["recent_windows"]] == [w[0] for w in windows[-2:]]
    assert report["trend"] == trend["status"] == status
    guidance = parent_guidance_view(report)
    assert guidance["pattern"] == PATTERN_LABELS[pattern]
    change = emotional_change_view(report, guidance)
    assert change["status"] == TREND_LABELS[status]
    if len(windows) > 1 and windows[0][2] != windows[-1][2]:
        assert PATTERN_LABELS[windows[0][2]] in change["change"]
        assert PATTERN_LABELS[windows[-1][2]] in change["change"]
    view = parent_xai_view(report)
    assert view["pattern"] == PATTERN_LABELS[pattern]
    eligible = [i for i, term in enumerate(terms[:5]) if term > 1e-12]
    if eligible and not xai["equal_distance"]:
        index = max(eligible, key=terms.__getitem__)
        count = round(raw[index] * len(selected))
        scope = "recent" if windows else "recorded"
        assert view["observation"].startswith(
            f"{count} out of {len(selected)} {scope} sessions" if count else f"None of the {scope} sessions")
        assert view["observation"].endswith((
            "recorded as calm.", "recorded as guarded.", "recorded as showing distress.",
            "recorded as unclear.", "included words that can suggest thinking about what to do differently.",
        )[index])
        assert view["explanation"].startswith(view["observation"])
        assert "supported the AI’s match" in view["explanation"]
    else:
        assert view["observation"] is None
    return report


@pytest.fixture(scope="module")
def educator_data():
    return pd.read_csv(COMPONENT / "data" / "educator_sessions.csv")


@pytest.mark.parametrize("child", ["C001", "C004", "C007", "C031"])
def test_named_educator_children(child, educator_data, artifacts):
    original = assert_result(educator_data, child, artifacts)
    shuffled = educator_data.sample(frac=1, random_state=13)
    repeated = services.emotional_pattern_report(shuffled, child)
    assert repeated["xai"] == original["xai"]


@pytest.mark.parametrize("child", ["C001", "C004"])
def test_named_dashboard_demo_children(child, artifacts):
    assert_result(services.demo_sessions(), child, artifacts)


def make_sessions(groups):
    origin = pd.Timestamp("2026-01-01T23:59:59Z")
    return pd.DataFrame([
        {"child_id": "audit", "session_id": f"{window}-{i}",
         "timestamp": origin + pd.Timedelta(days=14 * window, seconds=i),
         "emotional_state": emotion, "self_regulation_shown": flag, "dialogue_turns": turns}
        for window, group in enumerate(groups) for i, (emotion, flag, turns) in enumerate(group)
    ])


@pytest.fixture(scope="module")
def pattern_examples(artifacts):
    examples = {}
    for emotion, flag, turns in product(EMOTIONS, (False, True), range(1, 5)):
        group = [(emotion, flag, turns)] * 3
        frame = make_sessions([group])
        pattern = reference_prediction(list(frame.itertuples(index=False)), artifacts)[0]
        examples.setdefault(pattern, group)
    assert examples.keys() == PATTERN_LABELS.keys()
    return examples


@pytest.mark.parametrize("pattern", list(PATTERN_LABELS))
@pytest.mark.parametrize("status", list(TREND_LABELS))
def test_saved_model_across_all_pattern_trend_combinations(pattern, status, pattern_examples, artifacts):
    calm, distressed = "Calm / reflective", "Distressed"
    sequences = {
        "improving": (distressed, distressed, calm, pattern),
        "worsening": (calm, calm, distressed, pattern),
        "stable": (pattern, pattern, pattern, pattern),
        "not enough data": (calm, pattern),
    }
    # Include a one-record period after the latest reliable window, and an empty period.
    groups = [pattern_examples[label] for label in sequences[status]] + [[], pattern_examples[calm][:1]]
    frame = make_sessions(groups)
    report = assert_result(frame, "audit", artifacts)
    assert (report["current_pattern"], report["trend"]) == (pattern, status)


def test_saved_model_majority_can_differ_from_assignment(artifacts):
    frame = make_sessions([[("calm", False, 5), ("calm", False, 5), ("defensive", False, 5)]])
    report = assert_result(frame, "audit", artifacts)
    assert frame.emotional_state.mode().item() == "calm"
    assert report["current_pattern"] != "Calm / reflective"


def test_subthreshold_change_has_stable_trend(pattern_examples, artifacts):
    calm = "Calm / reflective"
    withdrawn = "Withdrawn / unclear"
    frame = make_sessions([pattern_examples[calm]] * 3 + [pattern_examples[withdrawn]])
    report = assert_result(frame, "audit", artifacts)
    assert (report["trend"], report["severity_change"]) == ("stable", 0.5)


def test_all_sessions_fallback_when_each_window_is_sparse(artifacts):
    frame = make_sessions([[("calm", False, 2)]] * 3)
    report = assert_result(frame, "audit", artifacts)
    assert report["current_since"] is None
    assert report["xai"]["pattern"]["analysis_period"]["source"] == "all_sessions"


@pytest.mark.parametrize("column,value", [
    ("timestamp", None), ("timestamp", "bad date"), ("dialogue_turns", None),
    ("dialogue_turns", float("inf")), ("dialogue_turns", 1.5),
    ("emotional_state", None), ("self_regulation_shown", None),
])
def test_invalid_record_is_rejected_before_explanation(column, value):
    frame = make_sessions([[("calm", False, 2)] * 3]).astype({column: object})
    if value == "bad date":
        frame[column] = frame[column].astype(str)
    frame.loc[2, column] = value
    with pytest.raises(ValueError, match=column if column != "timestamp" else "timestamp|date"):
        services.emotional_pattern_report(frame, "audit")


def test_missing_feature_and_duplicate_only_history():
    frame = make_sessions([[("calm", False, 2)] * 3])
    with pytest.raises(ValueError, match="dialogue_turns"):
        services.emotional_pattern_report(frame.drop(columns="dialogue_turns"), "audit")
    duplicate_rows = frame.iloc[[0, 0, 1]]
    report = services.emotional_pattern_report(duplicate_rows, "audit")
    assert not report["enough_data"]
    assert report["sessions"] == 2
    assert report["xai"]["status"] == "insufficient_data"


def test_nullable_turn_count_is_rejected_before_aggregation():
    frame = make_sessions([[("calm", False, 2)] * 3])
    frame["dialogue_turns"] = pd.array([2, pd.NA, 2], dtype="Int64")
    with pytest.raises(ValueError, match="dialogue_turns"):
        services.emotional_pattern_report(frame, "audit")


def test_explainer_rejects_mixed_children():
    frame = make_sessions([[("calm", False, 2)] * 3])
    frame.loc[0, "child_id"] = "other"
    with pytest.raises(ValueError, match="exactly one child"):
        services._f1_module().explain_pattern(frame)

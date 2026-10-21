import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # repo root: Component 2

from src import behavioral_synthetic as S
from src.behavioral_dataset import (
    BASELINE_COLUMNS,
    BASELINE_METRICS,
    DAILY_COLUMNS,
    DATASET_BASELINE_COLUMNS,
    DESCRIPTIVE_NAMES,
    FEATURE_CATALOG,
    LABEL_COLUMNS,
    MODEL_FEATURES,
    add_baseline_features,
    c1_events,
    c2_events,
    read_jsonl,
)
from component2_hate_speech_detection.src.engine.decide import decide

DATA = Path(__file__).resolve().parents[1] / "data"
MOCK = Path(__file__).resolve().parents[1] / "mock_inputs"
COUNT_COLUMNS = [c for c in MODEL_FEATURES if c not in DATASET_BASELINE_COLUMNS]
DATASET_DAILY_COLUMNS = [DESCRIPTIVE_NAMES[c] for c in DAILY_COLUMNS]
TO_BUILDER = {d: b for b, d in DESCRIPTIVE_NAMES.items()}


@pytest.fixture(scope="module")
def result():
    return S.build_dataset(S.SEED)


@pytest.fixture(scope="module")
def daily(result):
    return result.daily


@pytest.fixture(scope="module")
def written(result, tmp_path_factory):
    return S.write_outputs(result, tmp_path_factory.mktemp("behavioral"))


# ------------------------------------------------------------------ format

def test_records_have_the_sample_shapes(result):
    c1_sample_keys = set(read_jsonl(MOCK / "c1_sample_incidents.jsonl")[0])
    assert all(set(r) == c1_sample_keys | {"child_id"} for r in result.c1_records)
    c2_allowed = {"id", "ts", "persona", "top_category", "p", "theta", "rung", "urgent",
                  "abstained", "recommended_action", "source", "app", "child_id"}
    assert all(set(r) == c2_allowed for r in result.c2_records)
    assert {r["category"] for r in result.c1_records} == {"violence"}
    # the Step 1 builder accepts every generated record
    assert len(c1_events(result.c1_records)) == len(result.c1_records)
    assert len(c2_events(result.c2_records)) == len(result.c2_records)


def test_builder_columns_and_order(daily):
    builder = [c for c in daily.columns if c in DATASET_DAILY_COLUMNS]
    assert builder == DATASET_DAILY_COLUMNS
    assert set(daily["child_id_source"]) == {"record"} and not daily["demo_only"].any()
    assert len(daily.columns) == 69


def test_every_c2_record_matches_c2_decide(result):
    for r in result.c2_records:
        d = decide(r["top_category"], r["p"], r["persona"], abstained=r["abstained"])
        assert (d.rung, d.urgent, d.theta, d.recommended_action) == (
            r["rung"], r["urgent"], r["theta"], r["recommended_action"])
        assert d.rung != "L0"                       # C2 writes alerts only


# --------------------------------------------------------------- rows / keys

def test_rows_children_days_and_keys(daily):
    assert len(daily) == 3600
    assert sorted(daily["child_id"].unique()) == [f"C{i:03d}" for i in range(1, 41)]
    assert not daily.duplicated(["child_id", "date"]).any()
    expected = [(S.START_DATE + timedelta(days=i)).isoformat() for i in range(90)]
    for _, g in daily.groupby("child_id"):
        assert list(g["date"]) == expected
        assert list(g["day_index"]) == list(range(1, 91))


def test_full_coverage_no_missing_counts(daily):
    assert daily["screen_monitor_observed"].all() and daily["language_monitor_observed"].all()
    assert not daily[COUNT_COLUMNS].isna().any().any()
    rolling = [c for c in DATASET_BASELINE_COLUMNS if c != "baseline_ready"]
    ready = daily["baseline_ready"]
    assert not daily.loc[ready, rolling].isna().any().any()
    assert daily.loc[~ready, rolling].isna().all().all()
    assert (daily.loc[~ready, "day_index"] <= 7).all() and (daily.loc[ready, "day_index"] >= 8).all()


# ------------------------------------------------------------------- time

def test_sorted_and_splits_chronological(daily):
    assert daily[["child_id", "date"]].equals(
        daily[["child_id", "date"]].sort_values(["child_id", "date"]).reset_index(drop=True))
    ranges = {s: (g["date"].min(), g["date"].max()) for s, g in daily.groupby("split")}
    assert ranges["train"][1] < ranges["validation"][0]
    assert ranges["validation"][1] < ranges["test"][0]
    assert daily["split"].value_counts().to_dict() == {"train": 2240, "validation": 680, "test": 680}


def test_rolling_matches_manual_earlier_days_only(daily):
    g = daily[daily["child_id"] == "C007"].reset_index(drop=True)
    x = g["total_risk_events"].to_numpy()
    for i in range(7, 90):
        prev = x[max(0, i - 14):i]
        assert g.loc[i, "total_risk_events_prev14d_mean"] == pytest.approx(prev.mean())
        assert g.loc[i, "total_risk_events_prev14d_z"] == pytest.approx(
            (x[i] - prev.mean()) / (prev.std(ddof=1) + 0.5))


def test_changing_a_later_day_does_not_change_earlier_baselines(daily):
    builder = daily.rename(columns=TO_BUILDER)[DAILY_COLUMNS]
    original = add_baseline_features(builder)
    changed = builder.copy()
    idx = changed.index[(changed["child_id"] == "C003") & (changed["date"] == "2026-08-10")]
    changed.loc[idx, BASELINE_METRICS] = 99
    after = add_baseline_features(changed)
    child = after["child_id"] == "C003"
    cols = [c for c in BASELINE_COLUMNS if c != "baseline_ready"]
    earlier = child & (after["date"] < "2026-08-10")
    pd.testing.assert_frame_equal(after.loc[earlier, cols], original.loc[earlier, cols])
    # the changed day's own baseline (mean of earlier days) is untouched too
    same = child & (after["date"] == "2026-08-10")
    means = [c for c in cols if "_mean" in c]
    pd.testing.assert_frame_equal(after.loc[same, means], original.loc[same, means])
    later = (after["child_id"] == "C003") & (after["date"] == "2026-08-11")
    assert not after.loc[later, cols].equals(original.loc[later, cols])



# --------------------------------------------------------------- anomalies

def test_anomaly_rate_and_types_in_every_split(daily, result):
    assert 0.04 <= daily["is_anomaly"].mean() <= 0.07
    seen = daily[daily["is_anomaly"] == 1].groupby("split")["anomaly_type"].apply(set).to_dict()
    for split in ("train", "validation", "test"):
        assert seen[split] == set(S.ANOMALY_TYPES), split
    assert set(daily.loc[daily["is_anomaly"] == 0, "anomaly_type"]) == {"none"}
    assert (daily.loc[daily["is_anomaly"] == 0, "episode_id"].fillna("") == "").all()


def test_labels_match_observed_episode_records(daily, result):
    eps = result.episodes.set_index("episode_id")
    labelled = daily[daily["is_anomaly"] == 1]
    for eid, g in labelled.groupby("episode_id"):
        ep = eps.loc[eid]
        assert set(g["date"]) == set(ep["observed_dates"].split(";"))
        assert (g["child_id"] == ep["child_id"]).all()
        assert (g["anomaly_type"] == ep["anomaly_type"]).all()
        assert (g["split"] == ep["split"]).all()          # never across a boundary
        start = date.fromisoformat(ep["start_date"])
        end = date.fromisoformat(ep["end_date"])
        slack = 1 if ep["anomaly_type"] == "late_night_violence" else 0   # past midnight
        assert all(start <= date.fromisoformat(d) <= end + timedelta(days=slack) for d in g["date"])
    assert eps["observed_days"].sum() == len(labelled)


def test_episode_placement_rules(result):
    eps = result.episodes
    assert (pd.to_datetime(eps["start_date"]) >= pd.Timestamp(S.START_DATE) + pd.Timedelta(days=14)).all()
    assert eps["child_id"].nunique() == 40 - S.CHILDREN_WITHOUT_EPISODES
    for _, g in eps.sort_values("start_date").groupby("child_id"):
        starts, ends = pd.to_datetime(g["start_date"]), pd.to_datetime(g["end_date"])
        gaps = (starts.iloc[1:].to_numpy() - ends.iloc[:-1].to_numpy()) / np.timedelta64(1, "D")
        assert (gaps > S.MIN_GAP_DAYS).all()


def test_labels_are_never_inputs(result):
    assert not set(LABEL_COLUMNS) & set(MODEL_FEATURES)
    forbidden = {"is_anomaly", "anomaly_type", "episode_id", "episode", "injected"}
    for r in result.c1_records + result.c2_records:
        assert not forbidden & set(r)


# ----------------------------------------------------------- relationships

def test_feature_relationships(daily):
    d = daily
    assert (d["screen_blocked_incidents"] + d["screen_blurred_incidents"] == d["screen_incidents"]).all()
    assert (d["screen_violence_incidents"] == d["screen_incidents"]).all()
    assert (d["screen_max_confidence"] >= d["screen_mean_confidence"]).all()
    none1, some1 = d["screen_incidents"] == 0, d["screen_incidents"] > 0
    assert (d.loc[none1, ["screen_max_confidence", "screen_mean_confidence"]] == 0).all().all()
    assert d.loc[some1, "screen_max_confidence"].between(0.5, 1).all()
    assert d.loc[some1, "screen_mean_confidence"].between(0.5, 1).all()
    assert (d["screen_flagged_frames"] >= d["screen_incidents"]).all()
    assert (d["screen_escalated_incidents"] <= d["screen_blocked_incidents"]).all() and (d["screen_dismissed_incidents"] <= d["screen_blurred_incidents"]).all()
    cats = [f"language_{c}_flags" for c in ("threat", "hate_identity", "bullying", "sexual_harassment",
                                "self_harm", "grooming", "profanity")]
    assert (d[cats].sum(axis=1) == d["language_flags"]).all()
    assert (d["language_screen_text_flags"] + d["language_audio_flags"] + d["language_typed_flags"] == d["language_flags"]).all()
    assert (d["language_flags"] <= d["language_alert_records"]).all()
    assert (d["language_urgent_flags"] >= d["language_grooming_flags"] + d["language_self_harm_flags"]).all()
    none2, some2 = d["language_flags"] == 0, d["language_flags"] > 0
    assert (d.loc[none2, ["language_max_risk", "language_mean_risk"]] == 0).all().all()
    assert d.loc[some2, "language_max_risk"].between(0.5, 1).all()
    assert (d["language_distinct_apps"] <= d["language_flags"]).all()
    assert (d["night_risk_events"] <= d["total_risk_events"]).all()
    assert (d["total_risk_events"] == d["screen_incidents"] + d["language_flags"]).all()


def test_weekends_busier_and_children_differ(daily):
    normal = daily[daily["is_anomaly"] == 0]
    by_weekend = normal.groupby("is_weekend")["total_risk_events"].mean()
    assert by_weekend[True] > by_weekend[False]
    per_child = normal.groupby("child_id")["total_risk_events"].mean()
    assert per_child.max() > 2 * per_child.min()           # distinct baselines
    assert (normal["night_risk_events"] > 0).mean() > 0.02     # normal nights happen


# -------------------------------------------------------------- difficulty

def test_not_trivially_separable(daily):
    normal = daily[daily["is_anomaly"] == 0]
    assert (normal["language_urgent_flags"] > 0).mean() > 0.01        # urgent flags on normal days
    v = daily[(daily["split"] == "validation") & daily["baseline_ready"]]
    y = v["is_anomaly"].to_numpy() == 1
    best = 0.0
    for f in MODEL_FEATURES:
        x = v[f].astype(float).to_numpy()
        for t in np.unique(x):
            for pred in (x >= t, x <= t):
                tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
                if tp:
                    best = max(best, 2 * tp / (2 * tp + fp + fn))
    assert best < 0.90


# ---------------------------------------------------------- reproducibility

def test_saved_files_are_reproducible(written):
    for key, name in (("daily", "behavioral_daily.csv"), ("episodes", "behavioral_episodes.csv"),
                      ("c1", "synthetic_events/c1_incidents.jsonl"),
                      ("c2", "synthetic_events/c2_records.jsonl")):
        assert written[key].read_bytes() == (DATA / name).read_bytes(), name
    meta = json.loads((DATA / "behavioral_daily_metadata.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 42
    for key, digest in meta["sha256"].items():
        assert hashlib.sha256(written[key].read_bytes()).hexdigest() == digest


def test_saved_csv_round_trips(daily):
    saved = pd.read_csv(DATA / "behavioral_daily.csv", keep_default_na=False, na_values=[""])
    assert list(saved.columns) == list(daily.columns)
    assert saved["is_anomaly"].sum() == daily["is_anomaly"].sum()


def test_a_child_depends_on_no_other_child(result):
    children = S.make_children(S.SEED)
    eps = S.schedule_episodes(children, S.SEED)
    alone = S.generate_child_events(children[4], eps, S.SEED)
    assert sorted(alone.c1, key=lambda r: (r["started_at"], r["incident_id"])) == \
        [r for r in result.c1_records if r["child_id"] == "C005"]
    full_c2 = [r for r in result.c2_records if r["child_id"] == "C005"]
    assert sorted(alone.c2, key=lambda r: (r["ts"], r["id"])) == full_c2
    normal_only = S.generate_child_events(children[4], [], S.SEED)
    assert {r["id"] for r in normal_only.c2} <= {r["id"] for r in full_c2}


# ------------------------------------------------------------------ naming

def test_csv_uses_descriptive_names_only(daily):
    assert not [c for c in daily.columns if c.startswith(("c1_", "c2_"))]
    saved = pd.read_csv(DATA / "behavioral_daily.csv", nrows=0)
    assert list(saved.columns) == list(daily.columns)
    meta_and_labels = {"day_index", "split", "persona"} | set(LABEL_COLUMNS)
    catalogued = {d for _, d, _, _ in FEATURE_CATALOG}
    assert set(daily.columns) - meta_and_labels == catalogued


def test_name_mapping_is_one_to_one_and_complete():
    assert len(set(DESCRIPTIVE_NAMES.values())) == len(DESCRIPTIVE_NAMES)
    assert set(DAILY_COLUMNS) | set(BASELINE_COLUMNS) | {"is_weekend"} == set(DESCRIPTIVE_NAMES)
    assert all(src and how for _, _, src, how in FEATURE_CATALOG)


def test_model_features_are_csv_columns(daily):
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES)) == 45
    assert set(MODEL_FEATURES) <= set(daily.columns)
    meta = json.loads((DATA / "behavioral_daily_metadata.json").read_text(encoding="utf-8"))
    assert meta["columns"]["model_features"] == MODEL_FEATURES
    assert not {f for f in MODEL_FEATURES if daily[f].nunique(dropna=True) <= 1}

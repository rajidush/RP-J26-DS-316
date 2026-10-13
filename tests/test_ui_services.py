import io
import json
import shutil
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from db import configure

from ui import services

C4_DATA = services.C4_DIR / "data" / "educator_sessions.csv"


def record(**overrides):
    data = {
        "session_id": "s1",
        "timestamp": "2026-07-20T16:19:00Z",
        "risk_level": "low",
        "emotional_state": "calm",
        "self_regulation_shown": True,
        "escalation_flag": False,
        "dialogue_turns": 3,
    }
    data.update(overrides)
    return data


@pytest.fixture
def seeded(temp_db):
    rows = [
        record(session_id="a", timestamp="2026-07-01T10:00:00Z"),
        record(session_id="b", timestamp="2026-07-05T23:30:00Z", risk_level="high",
               emotional_state="distressed", self_regulation_shown=False, escalation_flag=True),
        record(session_id="c", timestamp="2026-07-06T00:30:00Z", risk_level="moderate",
               emotional_state="defensive"),
        record(session_id="d", timestamp="2026-07-15T10:00:00Z", risk_level="high",
               emotional_state="unclear", self_regulation_shown=False, escalation_flag=True),
    ]
    for r in rows:
        assert services.save_record(r)["ok"]
    return rows


# --- startup / reads ---------------------------------------------------------------

def test_startup_uses_database_url(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'env.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    configure()  # what a fresh process does on first use: read DATABASE_URL
    assert services.startup() == url
    assert (tmp_path / "env.db").exists()


def test_kpis_empty(temp_db):
    assert services.dashboard_kpis() == {"total_sessions": 0, "escalated": 0,
                                         "high_risk": 0, "self_regulation_pct": 0.0}


def test_kpis_and_counts(seeded):
    assert services.dashboard_kpis() == {"total_sessions": 4, "escalated": 2,
                                         "high_risk": 2, "self_regulation_pct": 50.0}
    assert services.risk_level_counts() == {"low": 1, "moderate": 1, "high": 2}
    assert services.emotional_state_counts() == {"calm": 1, "defensive": 1,
                                                 "distressed": 1, "unclear": 1}


def test_date_range_bounds_are_inclusive_utc_days():
    start, end = services.date_range_to_bounds(date(2026, 7, 5), date(2026, 7, 5))
    assert start == datetime(2026, 7, 5, tzinfo=timezone.utc)
    assert end == datetime(2026, 7, 6, tzinfo=timezone.utc)
    assert services.date_range_to_bounds() == (None, None)


def test_list_sessions_filters(seeded):
    ids = lambda **kw: [s["session_id"] for s in services.list_sessions(**kw)]
    assert ids() == ["d", "c", "b", "a"]
    assert ids(risk_levels=["high"], escalation_flag=True) == ["d", "b"]
    # 23:30 on the 5th is in, 00:30 on the 6th is out
    assert ids(start_date=date(2026, 7, 5), end_date=date(2026, 7, 5)) == ["b"]
    assert ids(start_date=date(2026, 7, 6)) == ["d", "c"]


def test_sessions_table_has_only_contract_columns(seeded):
    table = services.sessions_table(services.list_sessions())
    assert list(table.columns) == services.TABLE_COLUMNS
    assert len(table) == 4
    assert list(services.sessions_table([]).columns) == services.TABLE_COLUMNS


def test_escalated_and_get_session(seeded):
    assert [s["session_id"] for s in services.escalated_sessions()] == ["d", "b"]
    assert services.get_session("b")["risk_level"] == "high"
    assert services.get_session("zzz") is None


# --- adding records ----------------------------------------------------------------

def test_save_record_ok(temp_db):
    result = services.save_record(record())
    assert result["ok"] and isinstance(result["id"], int)


def test_save_record_reports_readable_validation_errors(temp_db):
    data = record(risk_level="extreme", dialogue_turns=0)
    del data["emotional_state"]
    result = services.save_record(data)
    assert not result["ok"]
    joined = "\n".join(result["errors"])
    assert "risk_level:" in joined
    assert "dialogue_turns:" in joined
    assert "emotional_state:" in joined


def test_save_record_duplicate(temp_db):
    services.save_record(record())
    result = services.save_record(record())
    assert not result["ok"]
    assert "already stored" in result["errors"][0]


def test_save_record_rejects_non_object(temp_db):
    assert not services.save_record(["not", "a", "dict"])["ok"]


def test_parse_json_records():
    assert services.parse_json_records('{"a": 1}') == [{"a": 1}]
    assert services.parse_json_records('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]
    with pytest.raises(ValueError, match="Not valid JSON"):
        services.parse_json_records("{oops")
    with pytest.raises(ValueError, match="object or a list"):
        services.parse_json_records("42")


def test_import_records_mixed(temp_db):
    text = json.dumps([record(session_id="x"), record(session_id="y", risk_level="bad"),
                       record(session_id="x")])
    results = services.import_records(services.parse_json_records(text))
    assert [r["ok"] for r in results] == [True, False, False]
    assert [r["session_id"] for r in results] == ["x", "y", "x"]
    assert services.dashboard_kpis()["total_sessions"] == 1


# --- Function 1: data sources ------------------------------------------------------

def test_demo_sessions_shape_and_determinism():
    df = services.demo_sessions()
    assert list(df.columns) == services.F1_COLUMNS
    assert services.child_ids(df) == ["C001", "C002", "C003", "C004", "C005"]
    assert df["session_id"].is_unique
    assert df["session_id"].str.startswith("DEMO-").all()
    assert set(df["emotional_state"]) <= set(services.EMOTIONAL_STATES)
    assert set(df["risk_level"]) <= set(services.RISK_LEVELS)
    assert (df["dialogue_turns"] >= 1).all()
    assert df["timestamp"].min() < "2026-08-04" and df["timestamp"].max() > "2026-09-28"
    assert df.equals(services.demo_sessions())
    assert not df.equals(services.demo_sessions(seed=1))


def test_demo_sessions_are_valid_contract_records(temp_db):
    # Not stored by the app, but every row must still satisfy the contract.
    for row in services.demo_sessions().drop(columns="child_id").to_dict("records"):
        result = services.save_record(row)
        assert result["ok"], result


def test_source_falls_back_to_demo_without_child_id(temp_db):
    services.save_record(record())
    source = services.child_session_source()
    assert source["source"] == "demo"
    assert services.child_ids(source["sessions"])[0] == "C001"


def test_source_uses_uploaded_csv_and_drops_extra_columns():
    csv = io.StringIO(
        "child_id,session_id,timestamp,risk_level,emotional_state,self_regulation_shown,"
        "escalation_flag,dialogue_turns,dialogue_summary\n"
        "K1,s1,2026-07-01T00:00:00Z,low,calm,True,False,2,secret text\n")
    source = services.child_session_source(csv)
    assert source["source"] == "upload"
    assert "dialogue_summary" not in source["sessions"].columns
    assert services.child_ids(source["sessions"]) == ["K1"]


def test_csv_without_child_id_is_rejected():
    csv = io.StringIO("session_id,timestamp\ns1,2026-07-01T00:00:00Z\n")
    with pytest.raises(ValueError, match="child_id"):
        services.load_sessions_csv(csv)


# --- Function 1: predictions ---------------------------------------------------------
# The trained model files are committed in component4_profiling_xai/models/, so
# these tests use the real model and fail (rather than skip) if it is missing.

def _copy_artifacts(dest, skip=(), metadata_overrides=None):
    """Copy the real Function 1 artifacts into dest, optionally leaving some out
    or editing the metadata."""
    for name in services.F1_ARTIFACT_FILES:
        if name not in skip:
            shutil.copy(services.f1_artifact_dir() / name, dest / name)
    if metadata_overrides:
        meta_path = dest / "emotional_pattern_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta.update(metadata_overrides)
        meta_path.write_text(json.dumps(meta))
    return dest


def _child_sessions(n, child="K1"):
    """n sessions for one child, one per day, all calm."""
    rows = [{"child_id": child, **record(session_id=f"{child}-{i}",
                                         timestamp=f"2026-07-{i + 1:02d}T10:00:00Z")}
            for i in range(n)]
    return pd.DataFrame(rows, columns=services.F1_COLUMNS)


def test_real_model_is_available_here():
    status = services.function1_status()
    assert status["available"], status["reason"]
    assert status["metadata"]["model"] == "KMeans"


@pytest.mark.parametrize("missing", services.F1_ARTIFACT_FILES)
def test_missing_file_gives_error_and_no_pattern(tmp_path, missing):
    _copy_artifacts(tmp_path, skip=(missing,))
    status = services.function1_status(tmp_path)
    assert not status["available"]
    assert status["missing"] == [missing]
    assert missing in status["reason"]

    with pytest.raises(services.ModelUnavailableError) as exc:
        services.emotional_pattern_report(_child_sessions(5), "K1", artifact_dir=tmp_path)
    assert exc.value.status["missing"] == [missing]


def test_missing_model_blocks_even_the_not_enough_data_result(tmp_path):
    # The model check comes first: no result of any kind without the model.
    with pytest.raises(services.ModelUnavailableError):
        services.emotional_pattern_report(_child_sessions(1), "K1", artifact_dir=tmp_path)


def test_unloadable_files_give_error(tmp_path):
    for name in services.F1_ARTIFACT_FILES:
        (tmp_path / name).write_text("garbage")
    status = services.function1_status(tmp_path)
    assert not status["available"]
    assert status["missing"] == []
    assert status["reason"].startswith("Could not load the model files")
    with pytest.raises(services.ModelUnavailableError):
        services.emotional_pattern_report(_child_sessions(5), "K1", artifact_dir=tmp_path)


def test_scikit_learn_version_mismatch_gives_error(tmp_path):
    _copy_artifacts(tmp_path, metadata_overrides={"sklearn_version": "0.0.1"})
    status = services.function1_status(tmp_path)
    assert not status["available"]
    assert "scikit-learn 0.0.1" in status["reason"]


def test_no_placeholder_code_left():
    for name in ("placeholder_predict", "placeholder_trend", "PLACEHOLDER_PATTERN"):
        assert not hasattr(services, name)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_too_few_sessions_gives_no_pattern(n, monkeypatch):
    # Function 1 must not even be called below the minimum.
    monkeypatch.setattr(services, "_run_function1",
                        lambda *a, **k: pytest.fail("Function 1 ran on too few sessions"))
    report = services.emotional_pattern_report(_child_sessions(n), "K1")
    assert report["enough_data"] is False
    assert report["sessions"] == n
    assert report["min_sessions"] == services.MIN_SESSIONS_FOR_PATTERN == 3
    assert report["current_pattern"] is None and report["overall_pattern"] is None
    assert report["features"] is None and report["window_features"] is None
    assert report["timeline"] == []


def test_too_few_counts_only_the_requested_child():
    sessions = pd.concat([_child_sessions(5, "OTHER"), _child_sessions(1, "K1")])
    report = services.emotional_pattern_report(sessions, "K1")
    assert (report["enough_data"], report["sessions"]) == (False, 1)
    assert services.emotional_pattern_report(sessions, "NOBODY")["sessions"] == 0


def test_three_sessions_gives_a_real_prediction():
    report = services.emotional_pattern_report(_child_sessions(3), "K1")
    assert report["enough_data"] is True
    assert report["status"]["available"]
    assert report["sessions"] == 3
    assert report["current_pattern"] == "Calm / reflective"
    assert report["features"]["p_calm"] == 1.0
    assert len(report["timeline"]) == 1  # one 2-week window with exactly 3 sessions


def test_report_matches_the_model_directly():
    """The page's pattern is exactly what component4's own pipeline predicts."""
    sessions = services.demo_sessions()
    child_sessions = sessions[sessions["child_id"] == "C002"]
    direct = services._f1_module().run_emotional_pattern(child_sessions)
    report = services.emotional_pattern_report(child_sessions, "C002")
    assert report["overall_pattern"] == direct.loc["C002", "pattern"]


def test_features_table_labels():
    table = services.features_table({"A": {"p_calm": 0.5, "mean_turns": 2.0},
                                     "B": {"p_calm": 0.25, "mean_turns": 3.0}})
    assert table.columns.tolist() == ["Feature", "A", "B"]
    assert table["Feature"].tolist() == ["Share of calm sessions", "Average dialogue turns"]
    assert table["A"].tolist() == [0.5, 2.0]
    assert table["B"].tolist() == [0.25, 3.0]


def _predict_from_features(features):
    """The trained model's pattern for one feature dict."""
    model, scaler, meta = services._f1_module().load_artifacts(services.f1_artifact_dir())
    X = pd.DataFrame([features])[meta["features"]]
    return meta["cluster_names"][int(model.predict(scaler.transform(X))[0])]


@pytest.mark.parametrize("child", ["C001", "C002", "C003", "C004", "C005"])
def test_latest_window_features_give_the_badge_pattern(child):
    report = services.emotional_pattern_report(services.demo_sessions(), child)
    assert _predict_from_features(report["window_features"]) == report["current_pattern"]
    assert _predict_from_features(report["features"]) == report["overall_pattern"]


def test_c004_window_and_overall_features_differ():
    """C004's badge (latest window) is Defensive while most of its sessions are calm."""
    report = services.emotional_pattern_report(services.demo_sessions(), "C004")
    assert report["current_pattern"] == "Defensive"
    assert report["window_features"]["p_defensive"] > report["features"]["p_defensive"]
    table = services.pattern_features_table(report)
    assert table.columns.tolist() == ["Feature", "Latest 2-week window: Defensive",
                                      f"All sessions: {report['overall_pattern']}"]


def test_no_window_column_without_a_window_pattern():
    report = services.emotional_pattern_report(_child_sessions(3), "K1")
    report.update(window_features=None, timeline=[])  # e.g. no window has 3 sessions
    table = services.pattern_features_table(report)
    assert table.columns.tolist() == ["Feature", "All sessions: Calm / reflective"]


@pytest.mark.parametrize("child, current, trend", [
    ("C001", "Calm / reflective", "stable"),
    ("C002", "Distressed", "worsening"),
    ("C003", "Calm / reflective", "improving"),
    ("C004", "Defensive", "worsening"),
])
def test_real_model_on_demo_children(child, current, trend):
    report = services.emotional_pattern_report(services.demo_sessions(), child)
    assert report["current_pattern"] == current
    assert report["trend"] == trend
    assert len(report["timeline"]) == 5
    assert list(report["features"]) == [
        "p_calm", "p_defensive", "p_distressed", "p_unclear", "selfreg_rate", "mean_turns"]


@pytest.mark.skipif(not C4_DATA.exists(), reason="training CSV not present")
def test_real_model_on_training_csv():
    sessions = services.child_session_source(C4_DATA)["sessions"]
    assert len(services.child_ids(sessions)) == 40
    report = services.emotional_pattern_report(sessions, "C001")
    assert report["overall_pattern"] == "Defensive"  # matches component4 outputs/


# --- Function 2 ----------------------------------------------------------------------

def test_behavioral_anomaly_on_sample():
    status = services.behavioral_anomaly_status()
    assert status["available"], status["reason"]
    sample = services.behavioral_sample_input()
    assert set(status["features"]) <= set(sample)
    result = services.predict_behavioral_anomaly(sample)
    assert result["status"] in ("Normal", "Anomaly")
    assert result["anomaly_score"] == pytest.approx(-result["decision_score"])


def test_behavioral_anomaly_rejects_bad_input():
    sample = services.behavioral_sample_input()
    sample[services.behavioral_anomaly_status()["features"][0]] = float("nan")
    with pytest.raises(ValueError):
        services.predict_behavioral_anomaly(sample)


def test_behavioral_anomaly_unavailable_gives_error_never_a_result(monkeypatch):
    module = services._f2_module()

    def missing():
        raise FileNotFoundError("Behavioral anomaly artifact not found: x.pkl")

    monkeypatch.setattr(module, "load_artifacts", missing)
    monkeypatch.setattr(module, "predict_behavioral_anomaly",
                        lambda data: pytest.fail("model ran without its artifacts"))
    status = services.behavioral_anomaly_status()
    assert not status["available"]
    assert "x.pkl" in status["reason"]
    with pytest.raises(services.ModelUnavailableError):
        services.predict_behavioral_anomaly(services.behavioral_sample_input())


def test_feature_label():
    assert services.feature_label("Daily_Usage_Hours") == "Daily usage hours"


# --- Function 1: database children ----------------------------------------------------

def test_source_prefers_database_children(temp_db):
    for i, (child, emo) in enumerate([("C002", "calm"), ("C001", "defensive"),
                                      ("C001", "defensive"), ("C001", "calm")]):
        services.save_record(record(session_id=f"s{i}", child_id=child, emotional_state=emo,
                                    timestamp=f"2026-07-0{i + 1}T10:00:00Z"))
    services.save_record(record(session_id="no-child"))
    source = services.child_session_source()
    assert source["source"] == "database"
    assert source["child_ids"] == ["C001", "C002"]

    sessions = services.sessions_for_child(source, "C001")
    assert list(sessions.columns) == services.F1_COLUMNS
    assert sessions["session_id"].tolist() == ["s1", "s2", "s3"]
    assert (sessions["child_id"] == "C001").all()


def test_sessions_for_child_from_demo(temp_db):
    source = services.child_session_source()
    assert source["source"] == "demo"
    sessions = services.sessions_for_child(source, "C003")
    assert set(sessions["child_id"]) == {"C003"}
    assert len(sessions) == (services.demo_sessions()["child_id"] == "C003").sum()


def test_report_for_database_child(temp_db):
    for i, emo in enumerate(["calm", "calm", "calm", "defensive"]):
        services.save_record(record(session_id=f"k{i}", child_id="K1", emotional_state=emo,
                                    timestamp=f"2026-07-0{i + 1}T10:00:00Z", dialogue_turns=2))
    source = services.child_session_source()
    report = services.emotional_pattern_report(services.sessions_for_child(source, "K1"), "K1")
    assert report["sessions"] == 4
    assert report["overall_pattern"] == "Calm / reflective"

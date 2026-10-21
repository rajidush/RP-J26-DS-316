"""Function 2 (Behavioral Anomaly Detection) in the UI service layer.

The saved Step 3 model is committed in component4_profiling_xai/models/, so
these tests use the real model and fail (rather than skip) if it is missing."""
import io
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ui import services

F2_DATA = services.F2_DEMO_DATA


def _copy_f2_artifacts(dest, skip=(), metadata_overrides=None):
    for name in services.F2_ARTIFACT_FILES:
        if name not in skip:
            shutil.copy(services.f2_artifact_dir() / name, dest / name)
    if metadata_overrides:
        path = dest / "behavioral_anomaly_metadata.json"
        meta = json.loads(path.read_text())
        meta.update(metadata_overrides)
        path.write_text(json.dumps(meta))
    return dest


def _demo():
    return services.behavioral_daily_source()["daily"]


def _csv(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


# --- loading and checks ------------------------------------------------------------------

def test_real_model_is_available_here():
    status = services.function2_status()
    assert status["available"], status["reason"]
    meta = status["metadata"]
    assert (meta["algorithm"], meta["feature_set"], meta["n_features"]) == ("one_class_svm", "A_daily", 27)
    assert list(status["pipeline"].feature_names_in_) == meta["features"]


def test_package_loader_does_not_register_src():
    # Fresh interpreter: other test files import component4's `src` themselves.
    code = ("import sys; from ui import services; "
            "assert services.function2_status()['available']; "
            "assert services.function1_status()['available']; "
            f"assert '{services.F2_PACKAGE}.behavioral_anomaly' in sys.modules; "
            "assert 'src' not in sys.modules, 'src was imported'")
    result = subprocess.run([sys.executable, "-c", code], cwd=services.ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("missing", services.F2_ARTIFACT_FILES)
def test_missing_file_gives_error_and_no_prediction(tmp_path, missing):
    _copy_f2_artifacts(tmp_path, skip=(missing,))
    status = services.function2_status(tmp_path)
    assert not status["available"] and status["missing"] == [missing]
    assert missing in status["reason"]
    with pytest.raises(services.ModelUnavailableError):
        services.behavioral_report(_demo(), "C001", artifact_dir=tmp_path)


def test_unloadable_files_give_error(tmp_path):
    for name in services.F2_ARTIFACT_FILES:
        (tmp_path / name).write_text("garbage")
    status = services.function2_status(tmp_path)
    assert not status["available"]
    assert status["reason"].startswith("Could not load the model files")


def test_scikit_learn_version_mismatch_gives_error(tmp_path):
    _copy_f2_artifacts(tmp_path, metadata_overrides={"sklearn_version": "0.0.1"})
    assert "scikit-learn 0.0.1" in services.function2_status(tmp_path)["reason"]


@pytest.mark.parametrize("override, words", [
    ({"features": None}, "different columns"),
    ({"algorithm": "isolation_forest"}, "Step 3 model"),
    ({"feature_set": "B_full"}, "Step 3 model"),
    ({"n_features": 45}, "Step 3 model"),
    ({"threshold": None}, "finite"),
])
def test_metadata_that_disagrees_with_the_pipeline_gives_error(tmp_path, override, words):
    meta = json.loads((services.f2_artifact_dir() / "behavioral_anomaly_metadata.json").read_text())
    if override.get("features", 0) is None:
        override = {"features": meta["features"][::-1]}
    _copy_f2_artifacts(tmp_path, metadata_overrides=override)
    status = services.function2_status(tmp_path)
    assert not status["available"]
    assert words in status["reason"]


def test_function1_and_function2_status_are_independent(tmp_path, monkeypatch):
    monkeypatch.setenv("C4_F2_ARTIFACT_DIR", str(tmp_path))
    assert not services.function2_status()["available"]
    assert services.function1_status()["available"]
    monkeypatch.delenv("C4_F2_ARTIFACT_DIR")
    monkeypatch.setenv("C4_F1_ARTIFACT_DIR", str(tmp_path))
    assert not services.function1_status()["available"]
    assert services.function2_status()["available"]


# --- data sources ------------------------------------------------------------------------

def test_demo_source_has_all_children_and_no_labels():
    source = services.behavioral_daily_source()
    assert source["source"] == "demo"
    assert source["child_ids"] == [f"C{i:03d}" for i in range(1, 41)]
    daily = source["daily"]
    assert len(daily) == 3600
    assert not {"is_anomaly", "anomaly_type", "episode_id", "split"} & set(daily.columns)


def test_upload_drops_extra_columns_and_parses_text_booleans():
    raw = pd.read_csv(F2_DATA).query("child_id == 'C003'")
    raw["is_weekend"] = raw["is_weekend"].map({True: "True", False: "False"})
    raw["unrelated"] = "x"
    source = services.behavioral_daily_source(_csv(raw))
    assert source["source"] == "upload" and source["child_ids"] == ["C003"]
    assert "unrelated" not in source["daily"] and "is_anomaly" not in source["daily"]
    assert set(source["daily"]["is_weekend"]) == {0.0, 1.0}


@pytest.mark.parametrize("edit, message", [
    (lambda d: d.drop(columns="language_bullying_flags"), "missing required columns"),
    (lambda d: d.drop(columns="date"), "missing required columns"),
    (lambda d: d.assign(date="01/07/2026"), "YYYY-MM-DD"),
    (lambda d: d.assign(is_weekend="maybe"), "is_weekend"),
    (lambda d: d.assign(language_flags="many"), "must be numeric"),
    (lambda d: pd.concat([d, d.head(1)]), "more than one row"),
    (lambda d: d.assign(child_id=np.nan), "without a child_id"),
])
def test_bad_uploads_are_rejected_with_a_readable_error(edit, message):
    raw = pd.read_csv(F2_DATA).head(20)
    with pytest.raises(ValueError, match=message):
        services.load_daily_csv(_csv(edit(raw)))


def test_upload_without_self_harm_imagery_column_still_runs():
    raw = pd.read_csv(F2_DATA).query("child_id == 'C001'").drop(
        columns="screen_self_harm_imagery_incidents")
    daily = services.load_daily_csv(_csv(raw))
    assert daily["screen_self_harm_imagery_incidents"].isna().all()
    services.behavioral_report(daily, "C001")


# --- predictions -------------------------------------------------------------------------

def test_report_matches_the_model_directly():
    f2 = services._f2_module()
    pipe, meta = f2.load_artifacts()
    daily = _demo()
    report = services.behavioral_report(daily, "C009")
    rows = daily[daily.child_id == "C009"].sort_values("date")
    expected = f2.anomaly_scores(pipe, f2.feature_matrix(rows, meta["features"]))
    assert np.allclose(report["daily"]["anomaly_score"], expected)
    assert (report["daily"]["unusual"] == (expected > meta["threshold"])).all()
    assert (report["daily"]["label"] == np.where(expected > meta["threshold"], "Unusual", "Normal")).all()
    assert np.allclose(report["daily"]["margin"], expected - meta["threshold"])
    assert report["threshold"] == meta["threshold"]


def test_bundled_test_split_reproduces_the_saved_step3_result():
    raw = pd.read_csv(F2_DATA)
    daily = _demo()
    meta = services.function2_status()["metadata"]
    flagged = pd.concat([services.behavioral_report(daily, c)["daily"].assign(child_id=c)
                         for c in services.child_ids(daily)])
    flagged["date"] = flagged["date"].astype(str)
    merged = raw.merge(flagged, on=["child_id", "date"])
    test = merged[merged.split == "test"]
    tp = int((test.unusual & (test.is_anomaly == 1)).sum())
    fp = int((test.unusual & (test.is_anomaly == 0)).sum())
    fn = int((~test.unusual & (test.is_anomaly == 1)).sum())
    assert (tp, fp, fn) == (meta["test"]["tp"], meta["test"]["fp"], meta["test"]["fn"])


def test_synthetic_labels_never_change_predictions():
    raw = pd.read_csv(F2_DATA).query("child_id == 'C002'")
    flipped = raw.assign(is_anomaly=1 - raw.is_anomaly, anomaly_type="volume_spike")
    a = services.behavioral_report(services.load_daily_csv(_csv(raw)), "C002")["daily"]
    b = services.behavioral_report(services.load_daily_csv(_csv(flipped)), "C002")["daily"]
    pd.testing.assert_frame_equal(a, b)


def test_missing_values_are_imputed_and_flagged():
    raw = pd.read_csv(F2_DATA).query("child_id == 'C004'").copy()
    raw.iloc[3, raw.columns.get_loc("language_mean_risk")] = np.nan
    report = services.behavioral_report(services.load_daily_csv(_csv(raw)), "C004")
    assert report["daily"]["imputed"].tolist() == [i == 3 for i in range(len(raw))]
    assert np.isfinite(report["daily"]["anomaly_score"]).all()


def test_unknown_child_is_a_readable_error():
    with pytest.raises(ValueError, match="No daily rows"):
        services.behavioral_report(_demo(), "NOPE")


def test_explanations_cover_exactly_the_unusual_days():
    report = services.behavioral_report(_demo(), "C009")
    unusual = list(report["daily"].loc[report["daily"]["unusual"], "date"])
    assert unusual and list(report["explanations"]) == unusual
    table = services.explanation_table(report["explanations"][unusual[0]])
    assert len(table) == services.F2_TOP_FEATURES
    assert set(table["Feature"]) <= set(services.F2_FEATURE_LABELS.values())
    assert set(services.F2_FEATURE_LABELS) == set(services._f2_module().FEATURE_SET_A)


def test_weekly_summary_is_monday_based_and_adds_up():
    report = services.behavioral_report(_demo(), "C001")
    daily, weekly = report["daily"], report["weekly"]
    assert all(pd.Timestamp(w).dayofweek == 0 for w in weekly["week_start"])
    assert weekly["days"].sum() == len(daily) == 90
    assert weekly["unusual_days"].sum() == daily["unusual"].sum()
    assert weekly["urgent_days"].sum() == daily["urgent"].sum()
    assert weekly["partial_week"].tolist() == (weekly["days"] < 7).tolist()
    assert weekly["partial_week"].iloc[0]          # 2026-07-01 is a Wednesday


# --- separation ---------------------------------------------------------------------------

def test_urgent_alerts_come_from_the_rule_not_the_model():
    daily = _demo()
    rule = services._urgent_module().urgent_alerts(daily)
    days = []
    for c in services.child_ids(daily):
        report = services.behavioral_report(daily, c)
        expected = rule[daily.child_id == c]["is_urgent"].to_numpy()
        assert (report["daily"]["urgent"].to_numpy() == expected).all()
        assert len(report["urgent"]) == expected.sum()
        days.append(report["daily"])
    days = pd.concat(days)
    # Both kinds of disagreement occur, so the outputs really are separate.
    assert (days.urgent & ~days.unusual).any()
    assert (days.unusual & ~days.urgent).any()


def test_ui_never_references_the_experimental_step4_model():
    for path in (services.ROOT / "ui").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for word in ("step4a", "Model_Improvement", "heldout", "selection_lock"):
            assert word not in text, f"{path.name} mentions {word}"
    assert services.F2_ARTIFACT_FILES == ("behavioral_anomaly_pipeline.pkl",
                                          "behavioral_anomaly_metadata.json")


def test_model_card_reads_the_saved_metadata():
    meta = services.function2_status()["metadata"]
    card = services.model_card(meta)
    assert card["Algorithm"].startswith("One-Class SVM")
    assert card["Features"].startswith("27 ")
    assert f"{meta['threshold']:.4f}" in card["Decision threshold"]

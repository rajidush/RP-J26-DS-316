"""Function 2 (Behavioral Anomaly Detection) in the UI service layer.

The saved Step 3 model is committed in component4_profiling_xai/models/, so
these tests use the real model and fail (rather than skip) if it is missing."""
import io
import json
import shutil
import subprocess
import sys
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ui import services


@pytest.fixture
def research_source(monkeypatch):
    monkeypatch.setenv("C4_F2_RESEARCH_SHAP", "1")
    clear = services._research_shap_resources.cache_clear
    clear()
    yield services.behavioral_daily_source()
    clear()


@pytest.mark.parametrize(("child", "day", "state"), [
    ("C008", date(2026, 9, 15), "reliable"),
    ("C033", date(2026, 9, 25), "uncertain"),
])
def test_research_shap_uses_frozen_score_and_group_diagnostics(research_source, child, day, state):
    result = services.research_shap_report(research_source, child, day)
    expected = services.behavioral_report(research_source["daily"], child)["daily"].set_index("date").loc[day]
    assert result["score"] == pytest.approx(expected["anomaly_score"], abs=1e-8)
    assert result["threshold"] == expected["threshold"]
    assert result["flag"] == expected["unusual"]
    assert result["status"] == state
    assert [(ref["size"], ref["seed"]) for ref in result["references"]] == [
        (32, 17), (32, 29), (64, 17), (64, 29),
    ]
    assert all(len(ref["groups"]) == 3 for ref in result["references"])
    if state == "uncertain":
        assert result["reasons"]
        assert result["neutral_text"] == (
            "The system identified an unusual pattern in the available recorded activity, "
            "but cannot reliably identify which recorded factors contributed most."
        )
    else:
        assert result["neutral_text"] is None


def test_research_shap_never_explains_unflagged_or_unassessable_days(research_source, monkeypatch):
    gate, daily = services._research_shap_resources(*services.research_shap_context(research_source))

    def unexpected(*args, **kwargs):
        pytest.fail("SHAP must not run")

    monkeypatch.setattr(gate.explainer, "explain", unexpected)
    child, day = "C002", date(2026, 9, 18)
    unflagged = services.research_shap_report(research_source, child, day)
    assert unflagged["status"] == "not_flagged" and not unflagged["references"]
    incomplete = daily.copy()
    incomplete.loc[incomplete.child_id.eq(child) & incomplete.date.eq(day), "language_monitor_observed"] = 0
    monkeypatch.setattr(services, "_research_shap_resources", lambda *_: (gate, incomplete))
    result = services.research_shap_report({**research_source, "daily": incomplete}, child, day)
    assert result["status"] == "insufficient_data" and not result["references"]
    assert result["score"] is result["flag"] is result["neutral_text"] is None


def test_research_shap_rejects_uploads_and_changed_assets(research_source, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="bundled synthetic data"):
        services.research_shap_context({**research_source, "source": "upload"})
    artifacts = _copy_f2_artifacts(tmp_path, metadata_overrides={"threshold": -11})
    monkeypatch.setenv("C4_F2_ARTIFACT_DIR", str(artifacts))
    with pytest.raises(ValueError, match="match the frozen files"):
        services.research_shap_context(research_source)
    monkeypatch.delenv("C4_F2_ARTIFACT_DIR")
    altered = tmp_path / "altered.csv"
    altered.write_bytes(F2_DATA.read_bytes() + b"\n")
    monkeypatch.setattr(services, "F2_DEMO_DATA", altered)
    with pytest.raises(ValueError, match="match the frozen files"):
        services.research_shap_context(research_source)


def test_research_shap_rejects_altered_demo_rows(research_source):
    changed = research_source["daily"].assign(language_flags=500)
    with pytest.raises(ValueError, match="unmodified bundled"):
        services.research_shap_report({**research_source, "daily": changed}, "C002", date(2026, 9, 18))


@pytest.fixture
def parent_demo(research_source, monkeypatch):
    monkeypatch.delenv("C4_F2_RESEARCH_SHAP")
    monkeypatch.setenv("C4_F2_PARENT_XAI_DEMO", "1")
    return research_source


@pytest.mark.parametrize(("child", "day", "message"), [
    ("C008", date(2026, 9, 15),
     "We noticed an unusual activity pattern. Your child's recorded screen-related activity contributed most to this pattern."),
    ("C002", date(2026, 9, 23),
     "We noticed an unusual activity pattern. Your child's recorded language-related activity contributed most to this pattern."),
    ("C033", date(2026, 9, 25),
     "The system identified an unusual pattern in the available recorded activity, "
     "but cannot reliably identify which recorded factors contributed most."),
    ("C002", date(2026, 9, 18),
     "No unusual activity pattern was detected in the available recorded data."),
])
def test_parent_activity_uses_verified_gate_wording(parent_demo, child, day, message):
    assert services.parent_activity_update(parent_demo, child, day, "127.0.0.1") == message


def test_parent_activity_insufficient_coverage_skips_attribution(parent_demo, monkeypatch):
    context = services.parent_xai_demo_context(parent_demo, "::1")
    gate, daily = services._research_shap_resources(*context)
    child, day = "C008", date(2026, 9, 15)
    incomplete = daily.copy()
    incomplete.loc[incomplete.child_id.eq(child) & incomplete.date.eq(day), "screen_monitor_observed"] = 0
    monkeypatch.setattr(services, "_research_shap_resources", lambda *_: (gate, incomplete))
    monkeypatch.setattr(gate.explainer, "explain", lambda *_: pytest.fail("Unexpected attribution"))
    assert services.parent_activity_update({**parent_demo, "daily": incomplete}, child, day, "::1") == (
        "There isn't enough recorded activity to assess today's pattern."
    )


def test_parent_activity_fails_closed_without_approved_factor(parent_demo, monkeypatch):
    decision = SimpleNamespace(status=SimpleNamespace(value="reliable"), is_flagged=True,
                               reasons=(), diagnostics={"stability": {"leader": "unapproved factor"}})
    monkeypatch.setattr(services, "_verified_shap_decision", lambda *_: (None, decision))
    assert services.parent_activity_update(parent_demo, "C008", date(2026, 9, 15), "127.0.0.1") is None
    decision.diagnostics["stability"]["leader"] = "screen-related recorded activity"
    decision.reasons = ("failed_evidence",)
    assert services.parent_activity_update(parent_demo, "C008", date(2026, 9, 15), "127.0.0.1") is None


def test_parent_activity_rejects_public_bind_and_uploads(parent_demo):
    with pytest.raises(ValueError, match="loopback-only"):
        services.parent_xai_demo_context(parent_demo, "0.0.0.0")
    with pytest.raises(ValueError, match="bundled synthetic data"):
        services.parent_xai_demo_context({**parent_demo, "source": "upload"}, "127.0.0.1")

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
    (lambda d: d.assign(screen_incidents=-1), "screen_incidents.*nonnegative whole counts"),
    (lambda d: d.assign(language_urgent_flags=0.5), "language_urgent_flags.*nonnegative whole counts"),
    (lambda d: d.assign(is_weekend=2), "is_weekend.*1/0"),
    (lambda d: d.assign(screen_flagged_seconds=float("inf")), "screen_flagged_seconds.*finite"),
    (lambda d: d.assign(screen_monitor_observed="unknown"), "screen_monitor_observed.*1/0"),
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


def test_incomplete_daily_inputs_do_not_become_normal_or_suppress_urgent_matches():
    f2 = services._f2_module()
    rows = _demo().query("child_id == 'C001'").head(5).copy().reset_index(drop=True)
    rows.loc[0, f2.FEATURE_SET_A] = np.nan
    rows.loc[1, "screen_monitor_observed"] = 0
    rows.loc[1, "language_grooming_flags"] = 2
    rows.loc[2, "language_flags"] = np.nan
    rows.loc[3, "language_monitor_observed"] = np.nan
    rows.loc[4, list(f2.DAILY_COUNTS)] = 0
    rows.loc[4, "screen_flagged_seconds"] = 0
    report = services.behavioral_report(rows, "C001")
    table = report["daily"]
    assert table.loc[:3, "label"].tolist() == ["Insufficient data"] * 4
    assert table.loc[:3, ["anomaly_score", "margin", "unusual"]].isna().all().all()
    assert table.loc[4, "label"] in ("Normal", "Unusual")
    assert "grooming language flag (2)" in report["urgent"]["reasons"].str.cat(sep=" ")
    assert report["urgent_incomplete"]
    assert f2.predict_anomalies(rows)["is_flagged"].iloc[:4].isna().all()


def test_urgent_report_does_not_load_model_and_accepts_partial_sources(monkeypatch):
    rows = pd.DataFrame({"child_id": ["C001"], "date": ["2026-07-01"],
                         "language_urgent_flags": [2]})
    monkeypatch.setattr(services, "function2_status", lambda *args: pytest.fail("model loaded"))
    monkeypatch.setattr(services, "_f2_module", lambda: pytest.fail("model module imported"))
    report = services.behavioral_urgent_report(rows, "C001")
    assert report["urgent"]["reasons"].tolist() == ["urgent language flag (2)"]
    assert report["urgent_incomplete"]


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

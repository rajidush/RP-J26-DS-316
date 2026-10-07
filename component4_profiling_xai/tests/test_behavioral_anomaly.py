"""
Function 2 (Behavioral Anomaly Detection) tests.
Run with: pytest component4_profiling_xai/tests/test_behavioral_anomaly.py -v

Test records are derived from the saved scaler's training statistics
(mean_ / scale_), so they sit on the real data's scale without hard-coding
feature names or values. Nothing here fits or refits any artifact.
"""
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component4_profiling_xai.src import behavioral_anomaly  # noqa: E402
from component4_profiling_xai.src.behavioral_anomaly import (  # noqa: E402
    FEATURES_PATH,
    MODEL_PATH,
    SCALER_PATH,
    load_artifacts,
    predict_behavioral_anomaly,
    preprocess_behavioral_input,
    validate_behavioral_input,
)

RESULT_KEYS = ["status", "anomaly_flag", "decision_score", "anomaly_score"]


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


@pytest.fixture(scope="module")
def feature_names():
    # Read straight from the .pkl so tests check against the stored order.
    return list(joblib.load(FEATURES_PATH))


@pytest.fixture
def normal_record(artifacts, feature_names):
    """Every feature at its training mean."""
    _, scaler, _ = artifacts
    return {name: float(m) for name, m in zip(feature_names, scaler.mean_)}


@pytest.fixture
def unusual_record(artifacts, feature_names):
    """Every feature 4 training standard deviations above its mean."""
    _, scaler, _ = artifacts
    return {
        name: float(m + 4 * s)
        for name, m, s in zip(feature_names, scaler.mean_, scaler.scale_)
    }


# --- artifacts ---------------------------------------------------------------

def test_artifacts_describe_13_features_in_the_same_order(artifacts, feature_names):
    model, scaler, loaded_names = artifacts
    assert len(feature_names) == 13
    assert loaded_names == feature_names
    assert scaler.n_features_in_ == 13
    assert list(scaler.feature_names_in_) == feature_names
    assert model.n_features_in_ == 13


def test_artifact_paths_are_the_notebook_variation_files():
    # Step 21 of the training notebook saves these exact file names.
    assert MODEL_PATH.name == "behavioral_anomaly_ocsvm_variation.pkl"
    assert SCALER_PATH.name == "behavioral_scaler_variation.pkl"
    assert FEATURES_PATH.name == "behavioral_features_variation.pkl"
    for path in (MODEL_PATH, SCALER_PATH, FEATURES_PATH):
        assert path.is_absolute() and path.is_file()


def test_feature_list_matches_training_notebook(feature_names):
    assert feature_names == [
        "Daily_Usage_Hours",
        "Sleep_Hours",
        "Academic_Performance",
        "Social_Interactions",
        "Exercise_Hours",
        "Screen_Time_Before_Bed",
        "Phone_Checks_Per_Day",
        "Apps_Used_Daily",
        "Time_on_Social_Media",
        "Time_on_Gaming",
        "Time_on_Education",
        "Family_Communication",
        "Weekend_Usage_Hours",
    ]


def test_model_has_tuned_notebook_hyperparameters(artifacts):
    model, _, _ = artifacts
    params = model.get_params()
    assert (params["kernel"], params["nu"], params["gamma"]) == ("rbf", 0.02, 0.05)


def test_missing_artifact_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.pkl"):
        behavioral_anomaly._load_artifact(tmp_path / "missing.pkl")


# --- A. valid input ----------------------------------------------------------

def test_valid_input_returns_one_row_in_training_order(normal_record, feature_names):
    df = validate_behavioral_input(normal_record)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 13)
    assert list(df.columns) == feature_names


def test_valid_input_preserves_values(normal_record, feature_names):
    df = validate_behavioral_input(normal_record)
    assert df.iloc[0].tolist() == [normal_record[name] for name in feature_names]


def test_numpy_and_int_values_are_accepted(normal_record, feature_names):
    record = dict(normal_record)
    record[feature_names[0]] = np.float32(4.5)
    record[feature_names[1]] = np.int64(7)
    record[feature_names[2]] = 70
    df = validate_behavioral_input(record)
    assert df.iloc[0, 0:3].tolist() == [4.5, 7.0, 70.0]


def test_non_dict_input_raises_type_error(feature_names):
    with pytest.raises(TypeError):
        validate_behavioral_input([1.0] * len(feature_names))


# --- B. missing feature ------------------------------------------------------

def test_missing_feature_raises_and_names_it(normal_record, feature_names):
    missing = feature_names[3]
    record = {k: v for k, v in normal_record.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        validate_behavioral_input(record)


def test_all_missing_features_are_reported(normal_record, feature_names):
    dropped = [feature_names[0], feature_names[-1]]
    record = {k: v for k, v in normal_record.items() if k not in dropped}
    with pytest.raises(ValueError) as exc_info:
        validate_behavioral_input(record)
    for name in dropped:
        assert name in str(exc_info.value)


# --- C-F. invalid values -----------------------------------------------------

@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param("5", id="numeric-string"),
        pytest.param("high", id="string"),
        pytest.param(None, id="none"),
        pytest.param([5.0], id="list"),
        pytest.param(1 + 2j, id="complex"),
        pytest.param(True, id="bool-true"),
        pytest.param(False, id="bool-false"),
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(np.nan, id="numpy-nan"),
        pytest.param(float("inf"), id="pos-inf"),
        pytest.param(float("-inf"), id="neg-inf"),
        pytest.param(-np.inf, id="numpy-neg-inf"),
    ],
)
def test_invalid_value_raises_value_error(normal_record, feature_names, bad_value):
    target = feature_names[5]
    record = {**normal_record, target: bad_value}
    with pytest.raises(ValueError, match=target):
        validate_behavioral_input(record)


def test_invalid_value_also_rejected_by_prediction(normal_record, feature_names):
    record = {**normal_record, feature_names[0]: float("nan")}
    with pytest.raises(ValueError):
        predict_behavioral_anomaly(record)


# --- G. extra keys -----------------------------------------------------------

def test_extra_keys_are_ignored(normal_record, feature_names):
    record = {**normal_record, "Addiction_Level": 9.0, "unrelated": "text"}
    df = validate_behavioral_input(record)
    assert list(df.columns) == feature_names
    assert predict_behavioral_anomaly(record) == predict_behavioral_anomaly(normal_record)


# --- H. preprocessing --------------------------------------------------------

def test_preprocessing_returns_finite_1x13_array(normal_record):
    scaled = preprocess_behavioral_input(normal_record)
    assert isinstance(scaled, np.ndarray)
    assert scaled.shape == (1, 13)
    assert np.isfinite(scaled).all()


def test_preprocessing_uses_saved_scaler(normal_record):
    # A record at the training mean scales to ~0 for every feature.
    scaled = preprocess_behavioral_input(normal_record)
    np.testing.assert_allclose(scaled, np.zeros((1, 13)), atol=1e-9)


# --- I/J. prediction ---------------------------------------------------------

def test_normal_record_is_predicted_normal(normal_record):
    result = predict_behavioral_anomaly(normal_record)
    assert list(result) == RESULT_KEYS
    assert result["status"] == "Normal"
    assert result["anomaly_flag"] == 0
    assert result["decision_score"] > 0


def test_unusual_record_is_predicted_anomaly(unusual_record):
    result = predict_behavioral_anomaly(unusual_record)
    assert list(result) == RESULT_KEYS
    assert result["status"] == "Anomaly"
    assert result["anomaly_flag"] == 1
    assert result["decision_score"] < 0


@pytest.mark.parametrize("record_fixture", ["normal_record", "unusual_record"])
def test_status_matches_sklearn_prediction(request, artifacts, record_fixture):
    model, _, _ = artifacts
    record = request.getfixturevalue(record_fixture)
    raw = int(model.predict(preprocess_behavioral_input(record))[0])
    result = predict_behavioral_anomaly(record)

    assert raw in (1, -1)
    expected = {1: ("Normal", 0), -1: ("Anomaly", 1)}[raw]
    assert (result["status"], result["anomaly_flag"]) == expected


# --- K. scores ---------------------------------------------------------------

@pytest.mark.parametrize("record_fixture", ["normal_record", "unusual_record"])
def test_scores_are_finite_python_floats(request, record_fixture):
    result = predict_behavioral_anomaly(request.getfixturevalue(record_fixture))
    for key in ("decision_score", "anomaly_score"):
        assert type(result[key]) is float
        assert math.isfinite(result[key])
    assert result["anomaly_score"] == -result["decision_score"]


def test_more_unusual_record_has_higher_anomaly_score(normal_record, unusual_record):
    normal = predict_behavioral_anomaly(normal_record)
    unusual = predict_behavioral_anomaly(unusual_record)
    assert unusual["anomaly_score"] > normal["anomaly_score"]


@pytest.mark.parametrize(
    "pattern",
    [
        # The three synthetic anomaly patterns from the training notebook
        # (values beyond its 95th / 5th percentile thresholds).
        pytest.param({"Daily_Usage_Hours": 10.0, "Sleep_Hours": 3.5,
                      "Phone_Checks_Per_Day": 148}, id="high_usage_low_sleep"),
        pytest.param({"Screen_Time_Before_Bed": 2.3, "Time_on_Social_Media": 4.7,
                      "Sleep_Hours": 3.5}, id="night_social_low_sleep"),
        pytest.param({"Weekend_Usage_Hours": 12.0, "Time_on_Gaming": 3.7,
                      "Daily_Usage_Hours": 10.0}, id="weekend_gaming_high_usage"),
    ],
)
def test_notebook_anomaly_pattern_raises_anomaly_score(normal_record, pattern):
    # Only a relative check: with every other feature at its mean, the SVM
    # may still place the record inside the boundary (notebook recall ~0.8).
    baseline = predict_behavioral_anomaly(normal_record)["anomaly_score"]
    patterned = predict_behavioral_anomaly({**normal_record, **pattern})["anomaly_score"]
    assert patterned > baseline


# --- no fitting --------------------------------------------------------------

def test_prediction_does_not_change_fitted_artifacts(artifacts, normal_record, unusual_record):
    model, scaler, _ = artifacts
    model_attrs = ("support_vectors_", "dual_coef_", "intercept_", "offset_")
    scaler_attrs = ("mean_", "var_", "scale_", "n_samples_seen_")
    before = {a: np.copy(getattr(model, a)) for a in model_attrs}
    before.update({a: np.copy(getattr(scaler, a)) for a in scaler_attrs})

    predict_behavioral_anomaly(normal_record)
    predict_behavioral_anomaly(unusual_record)

    for attr in model_attrs:
        assert np.array_equal(before[attr], getattr(model, attr)), attr
    for attr in scaler_attrs:
        assert np.array_equal(before[attr], getattr(scaler, attr)), attr

"""
Component 4 -- Function 2: Behavioral Anomaly Detection.

Validates one behavioral record, scales it with the StandardScaler fitted in
the Colab training notebook and scores it with the trained One-Class SVM
(see predict_behavioral_anomaly). The saved artifacts in ../models are
used as-is; nothing here trains or refits anything.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Paths are resolved relative to this file, so the module works no matter
# which directory it is run or imported from.
_SRC_DIR = Path(__file__).resolve().parent
MODELS_DIR = _SRC_DIR.parent / "models"

# Artifact names as saved in Step 21 of
# notebooks/Function_2_–_Behavioral_Anomaly_Detection_variation.ipynb.
MODEL_PATH = MODELS_DIR / "behavioral_anomaly_ocsvm_variation.pkl"
SCALER_PATH = MODELS_DIR / "behavioral_scaler_variation.pkl"
FEATURES_PATH = MODELS_DIR / "behavioral_features_variation.pkl"


def load_artifacts():
    """Load and return (model, scaler, feature_names) from the models directory.

    feature_names keeps the exact order stored at training time; the model
    expects input columns in that order. Raises FileNotFoundError if an
    artifact is missing, or RuntimeError if the artifacts disagree with
    each other on the feature set.
    """
    model = _load_model()
    scaler = _load_scaler()
    feature_names = list(_required_features())
    _check_artifacts_consistent(model, scaler, feature_names)
    return model, scaler, feature_names


def _load_artifact(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Behavioral anomaly artifact not found: {path}. Copy the files "
            f"saved by the Function 2 training notebook into {MODELS_DIR}."
        )
    return joblib.load(path)


def _check_artifacts_consistent(model, scaler, feature_names) -> None:
    """The scaler was fitted on a DataFrame with these columns, the SVM on
    the scaled array; all three must describe the same features in order."""
    scaler_names = list(getattr(scaler, "feature_names_in_", feature_names))
    if scaler_names != feature_names:
        raise RuntimeError(
            "Scaler feature names do not match behavioral_features_variation.pkl: "
            f"{scaler_names} vs {feature_names}"
        )
    if model.n_features_in_ != len(feature_names):
        raise RuntimeError(
            f"Model expects {model.n_features_in_} features but the feature "
            f"list has {len(feature_names)}"
        )


@lru_cache(maxsize=1)
def _required_features() -> tuple[str, ...]:
    """Feature names from the features .pkl, in stored order (loaded once)."""
    return tuple(_load_artifact(FEATURES_PATH))


@lru_cache(maxsize=1)
def _load_scaler():
    """The StandardScaler fitted in the notebook (loaded once, never refitted)."""
    return _load_artifact(SCALER_PATH)


@lru_cache(maxsize=1)
def _load_model():
    """The One-Class SVM trained in the notebook (loaded once, never refitted)."""
    return _load_artifact(MODEL_PATH)


def _is_valid_number(value) -> bool:
    """True for finite real numbers (Python or NumPy); False for bools, NaN, inf."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value))


def validate_behavioral_input(input_data) -> pd.DataFrame:
    """Check a single behavioral record and return it as model-ready input.

    input_data must be a dict containing every feature in
    behavioral_features_variation.pkl, each a finite real number (bools, NaN and
    +/-inf are rejected). Extra keys are ignored.

    Returns a one-row DataFrame whose columns are exactly the required
    features in training order. Raises ValueError listing every missing or
    invalid feature, or TypeError if input_data is not a dict.
    """
    if not isinstance(input_data, Mapping):
        raise TypeError(
            f"input_data must be a dict, got {type(input_data).__name__}"
        )

    required = _required_features()

    missing = [name for name in required if name not in input_data]
    if missing:
        raise ValueError(f"Missing required behavioral features: {missing}")

    invalid = {
        name: input_data[name]
        for name in required
        if not _is_valid_number(input_data[name])
    }
    if invalid:
        details = ", ".join(f"{name}={value!r}" for name, value in invalid.items())
        raise ValueError(
            "Behavioral features must be finite numbers (not bool, NaN or inf); "
            f"invalid values: {details}"
        )

    row = {name: float(input_data[name]) for name in required}
    return pd.DataFrame([row], columns=list(required))


def preprocess_behavioral_input(input_data) -> np.ndarray:
    """Validate a behavioral record and scale it with the saved training scaler.

    Returns a (1, n_features) NumPy array in training feature order, ready
    for the One-Class SVM (which was fitted on unnamed scaled arrays).
    Raises the same errors as validate_behavioral_input.
    """
    validated = validate_behavioral_input(input_data)
    return _load_scaler().transform(validated)


# sklearn OneClassSVM: +1 = inlier, -1 = outlier.
_PREDICTION_TO_RESULT = {
    1: ("Normal", 0),
    -1: ("Anomaly", 1),
}


def predict_behavioral_anomaly(input_data) -> dict:
    """Run the saved One-Class SVM on one behavioral record.

    Returns:
        status:         "Normal" or "Anomaly"
        anomaly_flag:   0 (normal) or 1 (anomaly)
        decision_score: model.decision_function, the signed distance from
                        the SVM boundary; the boundary is 0 and negative
                        values fall outside the learned normal region
        anomaly_score:  -decision_score (the notebook's anomaly_score()),
                        so higher means more anomalous.
                        A relative ranking value only -- not a probability,
                        percentage, confidence or clinical risk score.

    Raises the same errors as validate_behavioral_input.
    """
    scaled = preprocess_behavioral_input(input_data)
    model, _, _ = load_artifacts()

    prediction = int(model.predict(scaled)[0])
    decision_score = float(model.decision_function(scaled)[0])

    if prediction not in _PREDICTION_TO_RESULT:
        raise RuntimeError(
            f"Unexpected One-Class SVM prediction {prediction!r}; expected +1 or -1"
        )
    status, anomaly_flag = _PREDICTION_TO_RESULT[prediction]

    return {
        "status": status,
        "anomaly_flag": anomaly_flag,
        "decision_score": decision_score,
        "anomaly_score": -decision_score,
    }


if __name__ == "__main__":
    model, scaler, feature_names = load_artifacts()
    print(f"Model loaded successfully:  {type(model).__name__}")
    print(f"Scaler loaded successfully: {type(scaler).__name__}")
    print(f"Feature list loaded successfully: {len(feature_names)} features")
    for i, name in enumerate(feature_names, start=1):
        print(f"  {i:2d}. {name}")

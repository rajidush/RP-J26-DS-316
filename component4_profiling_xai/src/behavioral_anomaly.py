"""Function 2 - Behavioral Anomaly Detection, Step 3: training and inference code.

Used by the Function 2 notebook for training and by the dashboard for
prediction, so feature sets, preprocessing, scoring and thresholds are defined
in one place only.

    from src.behavioral_anomaly import predict_anomalies
    flagged = predict_anomalies(daily)          # daily = behavioral_daily.csv rows

Models are novelty detectors (Isolation Forest, One-Class SVM) fitted on normal
days only. Anomaly score: higher = more unusual. A day is flagged when its score
is above a threshold, stored as a quantile of the normal training days' scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .behavioral_dataset import BASELINE_COLUMNS, DESCRIPTIVE_NAMES, LABEL_COLUMNS, MODEL_FEATURES

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models"
PIPELINE_FILE = "behavioral_anomaly_pipeline.pkl"
METADATA_FILE = "behavioral_anomaly_metadata.json"

# Never model input: keys, bookkeeping and the synthetic ground truth.
META_COLUMNS = ["child_id", "date", "day_index", "split", "persona", "child_id_source",
                "demo_only", "baseline_ready"]
NON_FEATURE_COLUMNS = META_COLUMNS + LABEL_COLUMNS

# Set B: the full proposed model input. Set A: the same minus the 14-day
# per-child baselines (daily C1 + C2 + combined + calendar features only).
BASELINE_FEATURES = [DESCRIPTIVE_NAMES[c] for c in BASELINE_COLUMNS if c != "baseline_ready"]
FEATURE_SET_B = list(MODEL_FEATURES)
FEATURE_SET_A = [f for f in FEATURE_SET_B if f not in BASELINE_FEATURES]
FEATURE_SETS = {"A_daily": FEATURE_SET_A, "B_full": FEATURE_SET_B}

ALGORITHMS = {"isolation_forest": IsolationForest, "one_class_svm": OneClassSVM}


# ------------------------------------------------------------------ features

def feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """The model input: the named columns as floats (booleans become 0 / 1)."""
    bad = [f for f in features if f in NON_FEATURE_COLUMNS]
    if bad:
        raise ValueError(f"metadata / label columns cannot be model input: {bad}")
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return df[features].astype(float)


def check_features(train: pd.DataFrame, features: list[str], *, corr_limit: float = 0.95) -> dict:
    """Train-only checks: constant columns, exact duplicates, highly correlated pairs.

    Returns {"constant": [...], "duplicates": [(kept, dropped), ...],
    "high_corr": DataFrame of pairs with |r| >= corr_limit}.
    """
    X = feature_matrix(train, features)
    constant = [f for f in features if X[f].nunique(dropna=True) <= 1]
    duplicates, seen = [], {}
    for f in features:
        if f in constant:
            continue
        key = tuple(X[f].fillna(np.inf).to_numpy())
        if key in seen:
            duplicates.append((seen[key], f))
        else:
            seen[key] = f
    kept = [f for f in features if f not in constant]
    corr = X[kept].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    high = (upper[upper >= corr_limit].rename("abs_r").reset_index()
            .rename(columns={"level_0": "feature_1", "level_1": "feature_2"})
            .sort_values("abs_r", ascending=False, ignore_index=True))
    return {"constant": constant, "duplicates": duplicates, "high_corr": high}


def clean_feature_list(train: pd.DataFrame, features: list[str]) -> list[str]:
    """Drop constant columns and exact duplicates found in the training data."""
    report = check_features(train, features)
    drop = set(report["constant"]) | {d for _, d in report["duplicates"]}
    return [f for f in features if f not in drop]


# ------------------------------------------------------------------ models

def build_pipeline(algorithm: str, params: dict | None = None, *, seed: int = 42) -> Pipeline:
    """Median imputer -> StandardScaler -> novelty detector."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; choose from {list(ALGORITHMS)}")
    params = dict(params or {})
    if algorithm == "isolation_forest":
        params.setdefault("random_state", seed)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ALGORITHMS[algorithm](**params)),
    ])


def anomaly_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous (negated sklearn score_samples)."""
    return -pipeline.score_samples(X)


def threshold_from_quantile(train_scores: np.ndarray, quantile: float) -> float:
    """Score above which a day is flagged: a quantile of normal training scores."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    return float(np.quantile(train_scores, quantile))


def evaluate(y_true, scores, threshold: float) -> dict:
    """Precision / recall / F1 and confusion-matrix counts at a threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def tune_threshold(train_scores, val_scores, y_val, quantiles) -> tuple[dict, pd.DataFrame]:
    """Pick the training-score quantile with the best validation F1.

    Ties go to the higher quantile (fewer alerts). Returns the best row and the
    full table.
    """
    rows = []
    for q in quantiles:
        thr = threshold_from_quantile(train_scores, q)
        rows.append({"quantile": float(q), "threshold": thr, **evaluate(y_val, val_scores, thr)})
    table = pd.DataFrame(rows)
    best = table.sort_values(["f1", "quantile"], ascending=[False, False]).iloc[0].to_dict()
    return best, table


# ------------------------------------------------------------------ artifacts

def save_artifacts(pipeline: Pipeline, metadata: dict, artifact_dir=ARTIFACT_DIR) -> None:
    """Save the fitted pipeline and its metadata (features, threshold, config)."""
    for key in ("features", "threshold", "algorithm"):
        if key not in metadata:
            raise ValueError(f"metadata needs {key!r}")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_dir / PIPELINE_FILE)
    with open(artifact_dir / METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_artifacts(artifact_dir=ARTIFACT_DIR) -> tuple[Pipeline, dict]:
    """Load the trained pipeline and metadata saved by the notebook."""
    artifact_dir = Path(artifact_dir)
    pipeline = joblib.load(artifact_dir / PIPELINE_FILE)
    with open(artifact_dir / METADATA_FILE, encoding="utf-8") as f:
        metadata = json.load(f)
    return pipeline, metadata


PIPELINE_STEPS = ["imputer", "scaler", "model"]


def validate_artifacts(pipeline: Pipeline, metadata: dict, expected_features=None) -> None:
    """Check that a loaded pipeline and its metadata belong together.

    Raises ValueError when the pipeline steps, the fitted column order, the
    metadata feature list (and expected_features, when given) or the threshold
    disagree. scikit-learn itself rejects reordered named columns at scoring
    time; this catches a mismatched pair of files before anything is scored.
    """
    features = list(metadata.get("features") or [])
    steps = [name for name, _ in pipeline.steps]
    if steps != PIPELINE_STEPS or metadata.get("pipeline_steps", steps) != steps:
        raise ValueError(f"pipeline steps {steps} do not match {PIPELINE_STEPS}")
    fitted = list(getattr(pipeline, "feature_names_in_", []))
    if fitted != features:
        raise ValueError("the pipeline was fitted on different columns (or order) "
                         "than the metadata feature list")
    if expected_features is not None and features != list(expected_features):
        raise ValueError("the metadata feature list differs from the expected feature set")
    if metadata.get("n_features", len(features)) != len(features):
        raise ValueError(f"metadata n_features={metadata.get('n_features')} but "
                         f"{len(features)} features are listed")
    threshold = metadata.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
            or not np.isfinite(threshold):
        raise ValueError(f"metadata threshold {threshold!r} is not a finite number")


def explain_day(pipeline: Pipeline, row: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Descriptive view of one day: how far each observed value is from the
    average normal training day, in the pipeline's own standardised units.

    row: a one-row feature_matrix (the pipeline's columns, in order). Returns
    the top_n features by |z| with feature, observed, typical (training mean),
    z and direction. This describes the input; it is NOT an attribution of the
    model's score (the One-Class SVM combines all features non-linearly).
    """
    if len(row) != 1:
        raise ValueError("explain_day takes exactly one row")
    features = list(pipeline.feature_names_in_)
    z = pipeline[:-1].transform(row[features])[0]
    scaler = pipeline.named_steps["scaler"]
    table = pd.DataFrame({
        "feature": features,
        "observed": row[features].iloc[0].to_numpy(dtype=float),
        "typical": scaler.mean_,
        "z": z,
    })
    table["direction"] = np.where(table["z"] >= 0, "higher than typical", "lower than typical")
    order = table["z"].abs().sort_values(ascending=False, kind="stable").index
    return table.loc[order].head(top_n).reset_index(drop=True)


def predict_anomalies(daily: pd.DataFrame, artifact_dir=ARTIFACT_DIR) -> pd.DataFrame:
    """Score daily rows with the saved model.

    Returns child_id, date, anomaly_score and is_flagged (score > threshold),
    in the input order.
    """
    pipeline, metadata = load_artifacts(artifact_dir)
    scores = anomaly_scores(pipeline, feature_matrix(daily, metadata["features"]))
    out = daily[[c for c in ("child_id", "date") if c in daily.columns]].copy()
    out["anomaly_score"] = scores
    out["is_flagged"] = scores > metadata["threshold"]
    return out

"""Function 1 - Emotional Pattern Clustering: reusable pipeline code.

Used by the notebook for training and by the dashboard pipeline for prediction,
so cleaning, features and prediction are defined in one place only.

    from emotional_pattern import run_emotional_pattern
    patterns = run_emotional_pattern("educator_sessions.csv")
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

EMOTIONS = ["calm", "defensive", "distressed", "unclear"]
FEATURES = [f"p_{e}" for e in EMOTIONS] + ["selfreg_rate", "mean_turns"]
REQUIRED_COLUMNS = ["child_id", "session_id", "timestamp", "emotional_state",
                    "self_regulation_shown", "dialogue_turns"]
BOOL_COLUMNS = ["self_regulation_shown", "escalation_flag"]

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models"


def clean_sessions(df):
    """Validate and clean Component 3 session records.

    Raises ValueError instead of silently producing wrong features.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    sessions = df.copy()

    for col in [c for c in BOOL_COLUMNS if c in sessions.columns]:
        sessions[col] = (
            sessions[col].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False})
        )
        if sessions[col].isna().any():
            raise ValueError(f"{col} has values that are not True/False")

    sessions["timestamp"] = pd.to_datetime(sessions["timestamp"], utc=True)
    if sessions["timestamp"].isna().any():
        raise ValueError("timestamp is required for every session")
    sessions["emotional_state"] = sessions["emotional_state"].astype(str).str.strip().str.lower()

    unknown = set(sessions["emotional_state"]) - set(EMOTIONS)
    if unknown:
        raise ValueError(f"Unknown emotional_state values: {sorted(unknown)}")

    sessions["dialogue_turns"] = pd.to_numeric(sessions["dialogue_turns"], errors="raise")
    turns = sessions["dialogue_turns"]
    if (turns.isna().any() or not np.isfinite(turns).all()
            or not turns.ge(1).all() or not turns.mod(1).eq(0).all()):
        raise ValueError("dialogue_turns must be a positive whole number for every session")

    sessions = sessions.drop_duplicates()
    if sessions["session_id"].duplicated().any():
        raise ValueError("session_id is not unique after removing exact duplicates")

    return sessions.sort_values(["child_id", "timestamp"]).reset_index(drop=True)


def build_features(sessions):
    """Aggregate cleaned sessions into one row of emotional-pattern features per child."""
    g = sessions.groupby("child_id")

    X = pd.DataFrame({
        f"p_{e}": g["emotional_state"].apply(lambda x, e=e: (x == e).mean())
        for e in EMOTIONS
    })
    X["selfreg_rate"] = g["self_regulation_shown"].mean().astype(float)
    X["mean_turns"] = g["dialogue_turns"].mean()

    return X[FEATURES]


def load_artifacts(artifact_dir=ARTIFACT_DIR):
    """Load the trained model, scaler and metadata saved by the notebook."""
    artifact_dir = Path(artifact_dir)
    model = joblib.load(artifact_dir / "emotional_pattern_kmeans.pkl")
    scaler = joblib.load(artifact_dir / "emotional_pattern_scaler.pkl")
    with open(artifact_dir / "emotional_pattern_metadata.json") as f:
        metadata = json.load(f)

    # JSON stores the cluster ids as strings
    metadata["cluster_names"] = {int(c): n for c, n in metadata["cluster_names"].items()}
    return model, scaler, metadata


def predict_patterns(sessions, artifact_dir=ARTIFACT_DIR):
    """Assign each child in cleaned session data to a trained emotional pattern."""
    model, scaler, metadata = load_artifacts(artifact_dir)

    X = build_features(sessions)[metadata["features"]]
    labels = model.predict(scaler.transform(X))

    return pd.DataFrame({
        "sessions": sessions.groupby("child_id").size().loc[X.index].values,
        "cluster": labels,
        "pattern": [metadata["cluster_names"][c] for c in labels]
    }, index=X.index)


def run_emotional_pattern(data, artifact_dir=ARTIFACT_DIR):
    """Full pipeline: CSV path or DataFrame -> validate -> clean -> features -> patterns."""
    df = pd.read_csv(data) if isinstance(data, (str, Path)) else data
    return predict_patterns(clean_sessions(df), artifact_dir)


def explain_pattern(sessions, artifact_dir=ARTIFACT_DIR):
    """Explain the trained model's prediction for one child's cleaned sessions."""
    if sessions["child_id"].nunique() != 1 or sessions["child_id"].isna().any():
        raise ValueError("A pattern explanation requires sessions for exactly one child")
    model, scaler, metadata = load_artifacts(artifact_dir)
    features = metadata["features"]
    observed = build_features(sessions)[features]
    scaled = scaler.transform(observed)
    # Let predict decide the assignment, including ties.
    assigned = int(model.predict(scaled)[0])
    squared_terms = (scaled[0] - model.cluster_centers_) ** 2
    distances = squared_terms.sum(axis=1)
    alternative = min(
        (i for i in range(len(distances)) if i != assigned),
        key=lambda i: (distances[i], i),
    )
    contributions = squared_terms[alternative] - squared_terms[assigned]
    rows = [
        {"feature": feature, "observed_value": float(value), "contribution": float(term)}
        for feature, value, term in zip(features, observed.iloc[0], contributions)
    ]
    supporting = sorted(
        (row for row in rows if row["contribution"] > 0),
        key=lambda row: -row["contribution"],
    )
    return {
        "status": "available",
        "assigned_cluster": assigned,
        "pattern": metadata["cluster_names"][assigned],
        "alternative_cluster": alternative,
        "alternative_pattern": metadata["cluster_names"][alternative],
        "assigned_squared_distance": float(distances[assigned]),
        "alternative_squared_distance": float(distances[alternative]),
        "distance_margin": float(distances[alternative] - distances[assigned]),
        "equal_distance": bool(np.isclose(distances[assigned], distances[alternative],
                                          rtol=1e-12, atol=1e-12)),
        "feature_contributions": rows,
        "top_supporting_features": supporting[:3],
        "session_count": len(sessions),
    }


def explain_emotion_trend(trend, metadata):
    """Describe the same severity comparison used by the trend classifier."""
    severity = metadata["severity_order"]
    settings = metadata["trend_settings"]
    threshold = settings["change_threshold"]
    reliable = [
        {**window, "severity": severity[window["pattern"]]}
        for window in trend["windows"]
        if window["reliable"] and window["pattern"] in severity
    ]
    status = trend["trend"]
    change = trend["severity_change"]
    earlier, recent = [], []
    earlier_mean = recent_mean = None
    if change is None:
        message = f"At least {settings['min_valid_windows']} reliable windows are needed to explain the trend."
    else:
        earlier, recent = reliable[:2], reliable[-2:]
        earlier_mean = sum(w["severity"] for w in earlier) / 2
        recent_mean = sum(w["severity"] for w in recent) / 2
        reason = {
            "worsening": f"meets the worsening threshold of +{threshold}",
            "improving": f"meets the improving threshold of −{threshold}",
            "stable": f"is between the improving and worsening thresholds of −{threshold} and +{threshold}",
        }[status]
        message = (f"Mean emotional severity went from {earlier_mean:.1f} to {recent_mean:.1f}. "
                   f"The change of {change:+.2f} {reason}.")
    return {
        "status": status,
        "reliable_window_count": len(reliable),
        "earlier_windows": earlier,
        "recent_windows": recent,
        "earlier_mean": earlier_mean,
        "recent_mean": recent_mean,
        "severity_change": change,
        "explanation": message,
    }


def build_emotion_trend(sessions, artifact_dir=ARTIFACT_DIR):
    """Pattern per 2-week window, anchored to each child's first UTC day."""
    model, scaler, metadata = load_artifacts(artifact_dir)
    names = metadata["cluster_names"]
    severity = metadata["severity_order"]
    cfg = metadata["trend_settings"]

    window_days = cfg["window_days"]
    min_sessions = cfg["min_sessions"]
    min_valid = cfg["min_valid_windows"]
    threshold = cfg["change_threshold"]

    sessions = sessions.copy()
    starts = sessions.groupby("child_id")["timestamp"].min().dt.normalize()
    sessions["window"] = (
        (sessions["timestamp"] - sessions["child_id"].map(starts)).dt.days
        // window_days + 1
    )
    window_counts = sessions.groupby("child_id")["window"].max()
    counts = sessions.groupby(["child_id", "window"]).size()

    patterns = {}
    for w, part in sessions.groupby("window"):
        per_child = part.groupby("child_id").size()
        enough = per_child[per_child >= min_sessions].index

        if len(enough) == 0:
            continue

        X = build_features(
            part[part["child_id"].isin(enough)]
        )[metadata["features"]]

        for child, label in zip(
            X.index, model.predict(scaler.transform(X))
        ):
            patterns[(child, int(w))] = names[int(label)]

    result = {}
    for child in sorted(sessions["child_id"].unique()):
        start = starts.loc[child]
        n_windows = int(window_counts.loc[child])
        windows, sev = [], []

        for w in range(1, n_windows + 1):
            n = int(counts.get((child, w), 0))
            p = patterns.get((child, w)) or (
                "No sessions" if n == 0 else "Not enough data"
            )

            windows.append({
                "window": w,
                "window_start": (
                    start + pd.Timedelta(
                        days=window_days * (w - 1)
                    )
                ).date().isoformat(),
                "sessions": n,
                "reliable": n >= min_sessions,
                "pattern": p
            })

            if p in severity:
                sev.append(severity[p])

        if len(sev) < min_valid:
            status, change = "not enough data", None
        else:
            change = sum(sev[-2:]) / 2 - sum(sev[:2]) / 2
            status = (
                "worsening" if change >= threshold
                else "improving" if change <= -threshold
                else "stable"
            )
            change = round(float(change), 2)

        result[child] = {
            "trend": status,
            "worsening": status == "worsening",
            "severity_change": change,
            "windows": windows
        }

    return result


def run_dashboard_outputs(data, artifact_dir=ARTIFACT_DIR):
    """Return emotional patterns and trends."""
    df = pd.read_csv(data) if isinstance(data, (str, Path)) else data
    sessions = clean_sessions(df)

    patterns = predict_patterns(sessions, artifact_dir)
    trend = build_emotion_trend(sessions, artifact_dir)

    for child in trend:
        trend[child]["overall_pattern"] = patterns.loc[
            child, "pattern"
        ]

    return patterns, trend

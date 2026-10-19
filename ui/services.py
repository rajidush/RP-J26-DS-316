"""
Everything the Component 4 UI does with data: DB queries (via db/),
Function 1 (Emotional Pattern Clustering) and the demo dataset.
The Streamlit pages only call functions
from this module; colours and styling live in ui/theme.py.

Only contract fields are ever read or shown -- never raw conversation content.
"""
import importlib.util
import json
import os
import random
import warnings
from datetime import datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

from db import DuplicateRecordError, InvalidRecordError, get_database_url, init_db
from db import comp3
from db.comp3 import EMOTIONAL_STATES, RISK_LEVELS

ROOT = Path(__file__).resolve().parent.parent
C4_DIR = ROOT / "component4_profiling_xai"
F1_MODULE_PATH = C4_DIR / "src" / "emotional_pattern.py"
F1_ARTIFACT_FILES = ("emotional_pattern_kmeans.pkl", "emotional_pattern_scaler.pkl",
                     "emotional_pattern_metadata.json")

# Contract columns shown in tables, in this order. dialogue_summary is the
# contract's parent-safe summary, shown only in the detail view.
TABLE_COLUMNS = ["session_id", "timestamp", "risk_level", "emotional_state",
                 "self_regulation_shown", "escalation_flag", "dialogue_turns"]

# Function 1's pattern (cluster) names, least to most severe -- the model
# metadata's severity_order. ui/theme.py maps each to a colour and face.
PATTERN_ORDER = ["Calm / reflective", "Withdrawn / unclear", "Defensive", "Distressed"]

# Fewest sessions a pattern is computed from -- for a child overall and for
# each 2-week trend window (the trained model's trend_settings.min_sessions).
MIN_SESSIONS_FOR_PATTERN = 3


# --- Startup ----------------------------------------------------------------------

def startup():
    """Create tables if needed. The URL comes from DATABASE_URL, exactly as in db/base.py."""
    init_db()
    return get_database_url()


# --- Reading sessions ---------------------------------------------------------------

def dashboard_kpis():
    stats = comp3.session_stats()
    total = stats["total"]
    return {
        "total_sessions": total,
        "escalated": stats["escalated"],
        "high_risk": stats["high_risk"],
        "self_regulation_pct": round(100 * stats["self_regulation"] / total, 1) if total else 0.0,
    }


def risk_level_counts():
    return comp3.risk_summary()


def emotional_state_counts():
    return comp3.emotional_state_summary()


def date_range_to_bounds(start_date=None, end_date=None):
    """Inclusive calendar dates (UTC) -> (start, end) datetimes, end exclusive."""
    start = datetime.combine(start_date, time.min, timezone.utc) if start_date else None
    end = (datetime.combine(end_date + timedelta(days=1), time.min, timezone.utc)
           if end_date else None)
    return start, end


def list_sessions(risk_levels=None, emotional_states=None, escalation_flag=None,
                  start_date=None, end_date=None):
    """Filtered sessions, newest first. Dates are inclusive calendar days in UTC."""
    start, end = date_range_to_bounds(start_date, end_date)
    return comp3.list_sessions(risk_levels=risk_levels, emotional_states=emotional_states,
                               escalation_flag=escalation_flag, start=start, end=end)


def sessions_table(sessions):
    """Sessions as a DataFrame with only the table columns."""
    return pd.DataFrame(sessions, columns=TABLE_COLUMNS)


def get_session(session_id):
    return comp3.get_session(session_id)


def escalated_sessions():
    return comp3.get_escalated_sessions()


# --- Adding records -----------------------------------------------------------------

def _format_errors(exc):
    """Human-readable list of problems from a db error."""
    cause = exc.__cause__
    if hasattr(cause, "errors"):  # pydantic ValidationError
        out = []
        for err in cause.errors():
            field = ".".join(str(p) for p in err["loc"]) or "(record)"
            out.append(f"{field}: {err['msg']}")
        return out
    return [str(exc)]


def save_record(data):
    """Save one record. Returns {"ok": True, "id": row_id} or {"ok": False, "errors": [...]}."""
    if not isinstance(data, dict):
        return {"ok": False, "errors": [f"Expected a JSON object, got {type(data).__name__}"]}
    try:
        return {"ok": True, "id": comp3.save_comp3_output(data)}
    except (InvalidRecordError, DuplicateRecordError) as exc:
        return {"ok": False, "errors": _format_errors(exc)}


def parse_json_records(text):
    """JSON text (one object or a list of objects) -> list of records.
    Raises ValueError with a readable message on bad JSON."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})") from exc
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    raise ValueError("JSON must be an object or a list of objects")


def import_records(records):
    """Save each record independently; returns one result per record, in order."""
    results = []
    for i, data in enumerate(records):
        result = save_record(data)
        sid = data.get("session_id") if isinstance(data, dict) else None
        results.append({"index": i, "session_id": sid, **result})
    return results


# --- Component 4 modules --------------------------------------------------------------

def _load_module(name, path):
    # Loaded by file path under a unique name: component1/2/4 each have a
    # top-level `src` package, so `import src...` would collide in one process.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _f1_module():
    return _load_module("c4_emotional_pattern", F1_MODULE_PATH)


# --- Function 1: where the per-child sessions come from ------------------------------

F1_COLUMNS = ["child_id"] + TABLE_COLUMNS

# Each demo child's dominant emotional_state per 2-week phase (~9 weeks).
DEMO_TRAJECTORIES = {
    "C001": ["calm", "calm", "calm", "calm", "calm"],
    "C002": ["unclear", "unclear", "defensive", "defensive", "distressed"],
    "C003": ["distressed", "defensive", "unclear", "calm", "calm"],
    "C004": ["calm", "calm", "unclear", "defensive", "defensive"],
    "C005": ["defensive", "unclear", "unclear", "unclear", "calm"],
}
DEMO_START = datetime(2026, 8, 3, tzinfo=timezone.utc)
DEMO_DAYS = 63
_DEMO_RISK = {"calm": ("low", "low", "moderate"), "unclear": ("low", "moderate", "moderate"),
              "defensive": ("moderate", "moderate", "high"),
              "distressed": ("moderate", "high", "high")}


def demo_sessions(seed=316):
    """Deterministic, clearly synthetic sessions for C001-C005 over ~2 months,
    in the Component 3 contract shape plus child_id. Only used until the
    Component 3 contract carries a child_id. Never written to the database."""
    rng = random.Random(seed)
    rows = []
    for child, phases in DEMO_TRAJECTORIES.items():
        day, n = 0, 0
        while day < DEMO_DAYS:
            dominant = phases[min(day // 14, len(phases) - 1)]
            emotion = dominant if rng.random() < 0.8 else rng.choice(EMOTIONAL_STATES)
            risk = rng.choice(_DEMO_RISK[emotion])
            n += 1
            when = DEMO_START + timedelta(days=day, hours=rng.randint(15, 20),
                                          minutes=rng.choice((0, 15, 30, 45)))
            rows.append({
                "child_id": child,
                "session_id": f"DEMO-{child}-{n:03d}",
                "timestamp": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "risk_level": risk,
                "emotional_state": emotion,
                "self_regulation_shown": emotion == "calm" and rng.random() < 0.3,
                "escalation_flag": risk == "high" and emotion == "distressed",
                "dialogue_turns": 3 if emotion == "unclear" else 2,
            })
            day += rng.choice((1, 2, 2, 3))
    return pd.DataFrame(rows, columns=F1_COLUMNS)


def load_sessions_csv(file):
    """CSV with child_id + the contract fields (e.g.
    component4_profiling_xai/data/educator_sessions.csv). dialogue_summary and
    any other extra columns are dropped as soon as the file is read."""
    df = pd.read_csv(file)
    if "child_id" not in df.columns:
        raise ValueError("The CSV has no child_id column")
    return df[[c for c in F1_COLUMNS if c in df.columns]]


def child_session_source(uploaded_csv=None):
    """Where Function 1's per-child data comes from, in priority order:
    an uploaded CSV, else database records that carry child_id (from
    Component 3's data harness), else the demo dataset.

    Returns {"source": "upload" | "database" | "demo", "child_ids": [...],
    "sessions": DataFrame, or None for "database" (queried per child)}.
    """
    if uploaded_csv is not None:
        sessions = load_sessions_csv(uploaded_csv)
        return {"source": "upload", "child_ids": child_ids(sessions), "sessions": sessions}
    ids = comp3.list_child_ids()
    if ids:
        return {"source": "database", "child_ids": ids, "sessions": None}
    sessions = demo_sessions()
    return {"source": "demo", "child_ids": child_ids(sessions), "sessions": sessions}


def sessions_for_child(source, child_id):
    """One child's sessions (child_id + contract columns) from a child_session_source()."""
    if source["source"] == "database":
        return pd.DataFrame(comp3.get_sessions_for_child(child_id), columns=F1_COLUMNS)
    sessions = source["sessions"]
    return sessions[sessions["child_id"].astype(str) == child_id].reset_index(drop=True)


def child_ids(sessions):
    return sorted(sessions["child_id"].astype(str).unique())


# --- Function 1: Emotional Pattern Clustering ----------------------------------------
#
# Only the trained model is ever used. If it cannot be loaded there is no
# fallback: callers get ModelUnavailableError and must show no pattern.

class ModelUnavailableError(RuntimeError):
    """A trained Component 4 model could not be loaded; .status says why."""

    def __init__(self, status):
        super().__init__(status["reason"])
        self.status = status


def f1_artifact_dir():
    """Where the trained Function 1 artifacts live; override with C4_F1_ARTIFACT_DIR."""
    return Path(os.environ.get("C4_F1_ARTIFACT_DIR", C4_DIR / "models"))


def _load_strictly(load):
    """Call load() with scikit-learn's version-mismatch warning turned into an
    error: unpickling a model under another sklearn version is not reliable."""
    from sklearn.exceptions import InconsistentVersionWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        return load()


def function1_status(artifact_dir=None):
    """Can the trained Function 1 model be used?
    {"available", "artifact_dir", "missing", "reason", "metadata"}."""
    artifact_dir = Path(artifact_dir or f1_artifact_dir())
    missing = [f for f in F1_ARTIFACT_FILES if not (artifact_dir / f).is_file()]
    status = {"available": False, "artifact_dir": str(artifact_dir), "missing": missing,
              "reason": None, "metadata": None}
    if missing:
        status["reason"] = f"Missing in {artifact_dir}: " + ", ".join(missing)
        return status
    try:
        import sklearn

        _, _, metadata = _load_strictly(lambda: _f1_module().load_artifacts(artifact_dir))
        trained_with = metadata.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            raise RuntimeError(f"model was trained with scikit-learn {trained_with}, "
                               f"but {sklearn.__version__} is installed")
    except Exception as exc:  # unreadable pickle, missing/mismatched scikit-learn, bad JSON
        status["reason"] = f"Could not load the model files: {exc}"
        return status
    status.update(available=True, metadata=metadata)
    return status


def _run_function1(df, artifact_dir=None):
    """df must have child_id + the contract columns. Runs the trained model;
    raises ModelUnavailableError if it cannot be loaded."""
    status = function1_status(artifact_dir)
    if not status["available"]:
        raise ModelUnavailableError(status)
    f1 = _f1_module()
    sessions = f1.clean_sessions(df[[c for c in F1_COLUMNS if c in df.columns]])
    patterns, trend = f1.run_dashboard_outputs(sessions, status["artifact_dir"])
    return {"status": status, "sessions": sessions, "patterns": patterns,
            "features": f1.build_features(sessions).round(3), "trend": trend}


def _window_features(sessions, child_id, window_start, window_days):
    """Features of one child's sessions in the 2-week trend window starting at
    window_start (ISO date). Windows are counted from the first session's day,
    as in component4's build_emotion_trend, so these are the features behind
    that window's pattern."""
    begin = pd.Timestamp(window_start, tz="UTC")
    in_window = ((sessions["child_id"].astype(str) == child_id)
                 & (sessions["timestamp"] >= begin)
                 & (sessions["timestamp"] < begin + pd.Timedelta(days=window_days)))
    return _f1_module().build_features(sessions[in_window]).round(3).loc[child_id].to_dict()


def emotional_pattern_report(sessions, child_id, artifact_dir=None):
    """Everything the Emotional Pattern page shows for one child.

    sessions: usually just this child's (see sessions_for_child()); the
    2-week trend windows then start at the child's first session.

    Raises ModelUnavailableError if the trained model cannot be loaded --
    checked first, so no result of any kind is produced without it.
    With fewer than MIN_SESSIONS_FOR_PATTERN sessions the model is not run:
    enough_data is False and every pattern field is None / empty.
    Otherwise current_pattern is the latest window with enough sessions
    (falling back to the overall pattern); timeline lists only windows that
    have a pattern.
    """
    status = function1_status(artifact_dir)
    if not status["available"]:
        raise ModelUnavailableError(status)

    n_sessions = int((sessions["child_id"].astype(str) == child_id).sum())
    if n_sessions < MIN_SESSIONS_FOR_PATTERN:
        return {"enough_data": False, "child_id": child_id, "sessions": n_sessions,
                "min_sessions": MIN_SESSIONS_FOR_PATTERN, "status": status,
                "overall_pattern": None, "current_pattern": None, "current_since": None,
                "trend": None, "severity_change": None, "timeline": [], "features": None,
                "window_features": None}

    run = _run_function1(sessions, artifact_dir)
    trend = run["trend"][child_id]
    timeline = [{"date": w["window_start"], "pattern": w["pattern"], "sessions": w["sessions"]}
                for w in trend["windows"] if w["pattern"] in PATTERN_ORDER]
    overall = run["patterns"].loc[child_id, "pattern"]
    window_days = status["metadata"]["trend_settings"]["window_days"]
    return {
        "enough_data": True,
        "min_sessions": MIN_SESSIONS_FOR_PATTERN,
        "status": run["status"],
        "child_id": child_id,
        "sessions": int(run["patterns"].loc[child_id, "sessions"]),
        "overall_pattern": overall,
        "current_pattern": timeline[-1]["pattern"] if timeline else overall,
        "current_since": timeline[-1]["date"] if timeline else None,
        "trend": trend["trend"],
        "severity_change": trend["severity_change"],
        "timeline": timeline,
        "features": run["features"].loc[child_id].to_dict(),
        # Features behind current_pattern (the latest window); None when there is
        # no window with enough sessions and current_pattern is the overall one.
        "window_features": (_window_features(run["sessions"], child_id, timeline[-1]["date"],
                                             window_days) if timeline else None),
    }


FEATURE_LABELS = {
    "p_calm": "Share of calm sessions",
    "p_defensive": "Share of defensive sessions",
    "p_distressed": "Share of distressed sessions",
    "p_unclear": "Share of unclear sessions",
    "selfreg_rate": "Share with self-regulation shown",
    "mean_turns": "Average dialogue turns",
}


def features_table(columns):
    """{column header: feature dict} -> readable table, one value column per entry."""
    names = list(next(iter(columns.values())))
    return pd.DataFrame({"Feature": [FEATURE_LABELS.get(k, k) for k in names],
                         **{header: [f[k] for k in names] for header, f in columns.items()}})


def pattern_features_table(report):
    """The "Why this pattern?" table: the latest 2-week window (behind the badge)
    next to all sessions (behind the overall pattern), each header naming its pattern."""
    columns = {}
    if report["window_features"] is not None:
        columns[f"Latest 2-week window: {report['current_pattern']}"] = report["window_features"]
    columns[f"All sessions: {report['overall_pattern']}"] = report["features"]
    return features_table(columns)

# Component 4 — Behavioral Profiling / XAI Reporting

**Owner:** IT23135116
**Status:** Function 1 complete — Component 3 session records → per-child emotional pattern (K-Means) + 2-week trend.

## What it does

Groups each child's Component 3 sessions, turns them into six emotional-pattern
features, and assigns the child to one of four trained K-Means clusters. It then
re-runs the same model on every 2-week window to tell whether the child's pattern
is **stable**, **improving** or **worsening** over time. Only session metadata is
used — no dialogue text is read.

Input rows follow
[`docs/interface-contracts/comp3_to_comp4.schema.json`](../docs/interface-contracts/comp3_to_comp4.schema.json)
plus a `child_id` column. The schema does not carry `child_id`, so
[`integration/comp3_to_comp4.py`](../integration/comp3_to_comp4.py) validates
each record against the schema and adds it from device configuration (gap logged
in `docs/interface-contracts/CHANGELOG.md`). Component 3 needs no changes.

| Input column | Use |
|---|---|
| `child_id` | groups sessions per child (added by the adapter) |
| `session_id` | must be unique after exact duplicates are dropped |
| `timestamp` | ISO 8601, puts sessions into 2-week trend windows |
| `emotional_state` | one of `calm`, `defensive`, `distressed`, `unclear` |
| `self_regulation_shown` | `True` / `False` |
| `dialogue_turns` | integer ≥ 1 |
| `escalation_flag` | optional; checked to be `True` / `False` if present |
| `risk_level`, `dialogue_summary` | carried along, not used |

Bad input raises `ValueError` (missing column, unknown emotion label,
non-boolean flag, `dialogue_turns < 1`, duplicate `session_id`) instead of
silently producing wrong features.

## Model

| | |
|---|---|
| Algorithm | K-Means, `n_clusters=4`, `n_init=10`, `random_state=42` |
| Features (per child) | `p_calm`, `p_defensive`, `p_distressed`, `p_unclear` (share of sessions in each state), `selfreg_rate` (share with self-regulation), `mean_turns` |
| Scaling | `StandardScaler` fitted on the training children |
| Training data | `data/educator_sessions.csv` — 754 sessions, 40 children |
| Silhouette / Davies–Bouldin | 0.643 / 0.521 |
| Stability (ARI over seeds) / hold-out ARI | 1.0 / 1.0 |

| Cluster | Pattern | Severity |
|---|---|---|
| 1 | Calm / reflective | 0 |
| 3 | Withdrawn / unclear | 1 |
| 2 | Defensive | 2 |
| 0 | Distressed | 3 |

Training, evaluation, the DBSCAN comparison and the shares-only ablation are in
[`notebooks/Function_1___Emotional_Pattern_Clustering.ipynb`](notebooks/Function_1___Emotional_Pattern_Clustering.ipynb).
The artifacts it saves are committed (they are ~15 KB):

```
models/emotional_pattern_kmeans.pkl
models/emotional_pattern_scaler.pkl
models/emotional_pattern_metadata.json   # features, cluster names, severity order, trend settings, metrics
```

## Trend over time

Sessions are split into 14-day windows starting from the earliest session. In each
window, a child with **≥ 3 sessions** is classified with the same model; windows
with fewer are marked `"Not enough data"` (or `"No sessions"`) and skipped.

With **≥ 3 reliable windows**, the severity change is
mean(last 2 windows) − mean(first 2 windows):

| Change | Trend |
|---|---|
| ≥ +1 | `worsening` |
| ≤ −1 | `improving` |
| in between | `stable` |
| < 3 reliable windows | `not enough data` |

All of these settings are read from `trend_settings` in the metadata file.

## Setup

```bash
cd component4_profiling_xai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`scikit-learn` is pinned to **1.6.1**, the version the artifacts were trained
with; unpickling under another version is not guaranteed to be correct.

## Run it

All commands from `component4_profiling_xai/`:

```bash
# Patterns + trends for every child in a CSV → outputs/
python run_pipeline.py data/educator_sessions.csv

# Write somewhere else
python run_pipeline.py data/educator_sessions.csv --out path/to/dir
```

It prints the pattern counts and the children whose trend is worsening, and writes:

| File | Contents |
|---|---|
| `outputs/child_emotional_patterns.csv` | `child_id, sessions, cluster, pattern` |
| `outputs/emotion_trend.json` | per child: `trend`, `worsening`, `severity_change`, `overall_pattern`, and each window's `window_start`, `sessions`, `reliable`, `pattern` |

On the training data: 12 Defensive, 12 Calm / reflective, 10 Distressed,
6 Withdrawn / unclear; trends 30 stable, 4 improving, 2 worsening (C004, C012),
4 not enough data.

From Python:

```python
from component4_profiling_xai.src.emotional_pattern import (
    run_emotional_pattern, run_dashboard_outputs)

patterns = run_emotional_pattern("data/educator_sessions.csv")     # CSV path or DataFrame
patterns, trend = run_dashboard_outputs(df)                        # patterns + per-child trend dict
```

The building blocks — `clean_sessions`, `build_features`, `predict_patterns`,
`build_emotion_trend` — are shared by the notebook and the pipeline, so cleaning
and features are defined in one place only.

## Integration

`integration/end_to_end_pipeline.py` passes each Component 3 record through
`to_emotional_pattern_row()` and logs the resulting Component 4 row.
`integration/integration_tests/test_comp3_to_comp4.py` checks that adapted rows
have every column in `REQUIRED_COLUMNS`, that a schema-breaking record never
reaches Component 4, and that FSM sessions get an emotional pattern.

## Tests

```bash
pytest component4_profiling_xai/tests -v
```

`tests/test_emotional_pattern.py` uses two mock children (no CSV needed) to check
that a mostly-calm child is labelled Calm / reflective and a mostly-unclear child
Withdrawn / unclear, that unknown emotion labels and missing columns are
rejected, and that every child gets a valid trend and an `overall_pattern`.

---

Function 2 (behavioral anomaly detection, One-Class SVM) lives in
`src/behavioral_anomaly.py` and is exposed through
`component4.detect_behavioral_anomaly()`. `component4.generate_report()` is still
the placeholder parent-facing summary.

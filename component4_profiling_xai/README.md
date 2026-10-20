# Component 4 — Behavioral Profiling / XAI Reporting

**Owner:** IT23135116
**Status:** Function 1 complete — Component 3 session records → per-child emotional pattern (K-Means) + 2-week trend.
Function 2 integrated in the dashboard — daily C1/C2 risk features → Normal / Unusual day (One-Class SVM), on **synthetic** data only ([details](#function-2--behavioral-anomaly-detection)).

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
in `docs/interface-contracts/CHANGELOG.md`). The adapter currently defaults to a
placeholder device child ID; a real device-configuration source remains planned.
Component 3 needs no changes. Exported synthetic sessions already carry `child_id`.

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

These are clustering evaluation metrics, not classification accuracy. The recorded
silhouette and Davies–Bouldin values are 0.6428149815444244 and
0.5213693812644375 respectively; the table rounds them. Metadata also records
DBSCAN (`eps=1.2`, `min_samples=3`, 4 noise children) and the shares-only ablation
(silhouette 0.7333882955552604, ARI versus the final model 1.0).
The trend-window correction did not retrain or modify the trained model, scaler,
or metadata, and did not change overall emotional pattern predictions.

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

Sessions are converted to UTC and split into 14-day windows starting at midnight
on **each child's own earliest UTC session date**. Each child's final window is
also determined from that child's history. Previously, batch processing used the
earliest session across all children, while the dashboard supplied one child at a
time. That shifted window boundaries and produced different trend labels for 7
of 40 children. Per-child anchors make complete trend records identical whether
the same history is processed individually or as part of the full dataset.

In each
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

The **overall emotional pattern** uses all supplied sessions for a child. The
dashboard's **current pattern** uses the latest 14-day window with at least three
sessions, falling back to the overall pattern if no window qualifies. With fewer
than three sessions overall, the dashboard shows insufficient data instead of a
pattern. A trend describes change across reliable windows, not either pattern
alone; a latest-window pattern can differ from the overall pattern.

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

On the 40-child synthetic training dataset (754 sessions): 12 Defensive,
12 Calm / reflective, 10 Distressed, 6 Withdrawn / unclear. The regenerated
`outputs/emotion_trend.json` contains all 40 children with these verified trends:

| Trend | Children |
|---|---|
| Stable | 26 |
| Improving | 7 |
| Worsening | 3 |
| Not enough data | 4 |

The worsening children are C004, C012 and C035. The metadata's
`results.trend_status_counts` still records the historical pre-correction counts
(30 stable, 4 improving, 2 worsening, 4 not enough data); use the regenerated JSON
for current trend results. `outputs/child_emotional_patterns.csv` was unchanged.
The previous JSON is preserved as
`outputs/emotion_trend.before_per_child_windowing.json.bak`.

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

The implemented Streamlit application (`streamlit run ui/app.py` from the
repository root) reads existing SQLite session records on application reruns
(`DATABASE_URL`, default `sqlite:///project.db`). With the database source active,
Function 1 queries the selected child's sessions and recomputes its report.
Records need `child_id` for per-child analysis. An uploaded CSV takes priority
over the database; if no database child IDs exist, synthetic demo data is used.
The dashboard computes results from sessions rather than loading the saved batch
JSON.

New Component 3 CSV/JSONL exports are **not automatically imported**. Run the
existing JSONL import script separately from the repository root, then rerun the
dashboard:

```bash
python scripts/import_comp3_sessions.py
# Or specify a JSONL export:
python scripts/import_comp3_sessions.py path/to/educator_sessions.jsonl
```

The default input is `component3_socratic_educator/out/educator_sessions.jsonl`;
duplicate session IDs are skipped. CSV upload supports temporary dashboard
analysis but does not import records into SQLite. Automatic ingestion and
periodic dashboard refresh remain **future integration work**; an idle page does
not automatically refresh when exports or database records change.

## Tests

From the repository root:

```bash
python -B -m pytest -p no:cacheprovider component4_profiling_xai/tests/test_emotional_pattern.py -v
```

**Verified Function 1 result: 5 passed.** Four tests use two mock children to check
that a mostly-calm child is labelled Calm / reflective and a mostly-unclear child
Withdrawn / unclear, that unknown emotion labels and missing columns are
rejected, and that every child gets a valid trend and an `overall_pattern`.

The fifth test loads `data/educator_sessions.csv` and verifies that complete trend
records and overall pattern predictions match between batch and individual-child
processing for **all 40 children**, and that each first window starts on the
child's earliest UTC date. The regenerated saved JSON was also checked to match
the corrected batch results exactly.

Manual Streamlit dashboard verification reported by the user:

| Child | Trend |
|---|---|
| C001 | Improving |
| C024 | Improving |
| C035 | Worsening |

These three dashboard checks agree with the saved JSON. They check displayed
trend labels for those children, not every UI state or automatic ingestion.
The automated tests verify the specified mock behavior, validation cases and
dataset consistency; they do not establish classification accuracy, clinical
validity, generalization to real children, or complete real-time integration.

## Function 2 — Behavioral Anomaly Detection

Flags days on which a child's Component 1 (screen) and Component 2 (language)
risk events look unusual compared with normal days. **Synthetic data only**:
the model was trained and evaluated on `data/behavioral_daily.csv` (40 generated
children × 90 days, `src/behavioral_synthetic.py`). Nothing here establishes
real-world detection accuracy.

| | |
|---|---|
| Model (production) | Step 3 One-Class SVM, `nu=0.01`, `gamma=0.001` (RBF), notebook `notebooks/Function_2_Behavioral_Anomaly_Detection.ipynb` |
| Input | 27 daily features (`FEATURE_SET_A` in `src/behavioral_anomaly.py`): screen, language, combined counts, `is_weekend`; no 14-day baselines |
| Preprocessing | median imputer → `StandardScaler`, inside the saved pipeline, fitted on normal training days |
| Score / threshold | `anomaly_score = -score_samples(X)`; a day is **Unusual** when score > `-12.5158` (97 % quantile of normal training days) |
| Artifacts | `models/behavioral_anomaly_pipeline.pkl`, `models/behavioral_anomaly_metadata.json` (scikit-learn 1.6.1) |
| Synthetic test result | precision 0.627, recall 0.604, F1 0.615 (one-time Step 3 evaluation) |

The Step 4A model-improvement notebook (`notebooks/Function_2_Model_Improvement.ipynb`)
is **experimental**. Its detectors are not saved as artifacts and the dashboard
never loads them; a test checks that the production files still match the
hashes Step 4A protected.

Known limits (Step 4A review): detections are driven mainly by rare categories
(grooming, self-harm, urgent, night violence); multi-day bullying runs and
volume spikes are often missed; very quiet days can score close to the
threshold. A **Normal** day is therefore not proof that a child is safe.

### In code

```python
from src.behavioral_anomaly import predict_anomalies
flags = predict_anomalies(daily)        # child_id, date, anomaly_score, is_flagged

from src.urgent_alerts import urgent_alerts
alerts = urgent_alerts(daily)           # child_id, date, is_urgent, reasons
```

`validate_artifacts()` checks that a pipeline and its metadata belong together
(steps, fitted column order, feature list, threshold). `explain_day()` lists a
day's observed values furthest from an average normal training day, in the
pipeline's own standardised units. It describes the input, not the model's
reasoning: the One-Class SVM combines all 27 features non-linearly.

### Urgent alerts are separate

`src/urgent_alerts.py` flags a child-day with any urgent, grooming or self-harm
language flag, or self-harm imagery on screen (the Step 4A rule, unchanged). It
never reads anomaly scores or labels, and the model never reads its output.
Grooming and self-harm counts are also model inputs, so urgent days are often
Unusual too; the dashboard shows the two side by side and never merges them.

### Dashboard

`streamlit run ui/app.py` from the repository root, then open **Behavioral
Anomaly** in the sidebar (Emotional Pattern stays the default page). The page
shows, for the selected child:

- the latest day's Normal / Unusual status with score, saved threshold and margin (score − threshold);
- a daily score chart with the threshold line, and a daily predictions table;
- a Monday-based weekly summary (unusual days, urgent-rule days, partial weeks marked);
- "Why was this day unusual?": the top observed deviations, with the child's previous 14-day median for context;
- urgent rule-based alerts in their own card;
- "About this model", read from the saved metadata.

Data: components 1 and 2 have no database storage yet (`db/comp1.py` and
`db/comp2.py` are placeholders), so the page uses the bundled synthetic dataset,
labelled **SYNTHETIC DATA – DEMO PREDICTIONS**, or an uploaded daily-feature CSV
(`child_id`, `date` as YYYY-MM-DD, the 27 features; `screen_self_harm_imagery_incidents`
optional). Synthetic labels and other extra columns are dropped as soon as the
file is read, and missing values are imputed and marked. The demo IDs C001–C040
are not the Function 1 demo children with the same names.

The UI loads `src/` as a package under a unique name (`c4_component4_src`),
because `behavioral_anomaly.py` uses relative imports. `src` itself is never
imported, so it cannot clash with other components' `src` packages. Set
`C4_F2_ARTIFACT_DIR` to load the model files from another folder. If they are
missing, unreadable, from another scikit-learn version, or not the Step 3
model, the page shows only an error card.

### Function 2 tests

From the repository root:

```bash
python -B -m pytest -p no:cacheprovider component4_profiling_xai/tests/test_behavioral_anomaly.py component4_profiling_xai/tests/test_urgent_alerts.py tests/test_ui_f2_services.py tests/test_ui_pages.py tests/test_ui_theme.py -q
```

They cover artifact validation, explanation maths, the urgent rule, model
loading failures, CSV validation, agreement with the model and the saved Step 3
test counts, label independence, the weekly summary, the separation of urgent
alerts and the experimental model, and page rendering with and without the
model. The full UI and Component 4 suite (`python -m pytest tests component4_profiling_xai/tests`)
passed with **244 tests** after the dashboard integration (172 before it).

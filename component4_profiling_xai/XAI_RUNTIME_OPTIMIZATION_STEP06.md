# Step 06 · XAI runtime and memory optimization

## Scope and computational audit

The deployed One-Class SVM, scores, thresholds, original temporal rules, adaptive profile and independent urgent rules are held fixed. The experiment compares two calculation paths on the same saved model, reference rows and data.

The original explainer predicts every background donor for each of eight binary source-group coalitions. It builds eight copied frames, concatenates them, and invokes the pipeline. Both LIME and SHAP already share that scored neighborhood. The full coalition contains identical copies of the explained row, and the empty coalition is the same background for every explained record. Step 02 separately scores raw perturbed inputs for verification. Temporal replay invokes the full daily explainer for each observed record in the two windows, then repeats it for supported patterns under the two Step 02 perturbations. Temporal comparison consumes grouped SHAP and reconstruction evidence. It does not consume local LIME diagnostics. The selected day can be rescored by the daily path after the temporal path has scored the same record.

Adaptive profile replay, feature preparation and anomaly scoring remain separate. Adaptive comparison is measured as its own stage and is also included in the model-side end-to-end path.

## Exact optimization candidates

- Precompute mask membership for the request and construct one bounded coalition batch with NumPy. Recompute the cross-source totals exactly as before. Process up to two distinct rows per prediction call.
- Score the full coalition once per distinct feature row and the empty coalition once per request. Restore the original ordered donor score grid before the original SHAP reductions and seeded LIME fit. This retains tied ranks, signed values, donor splits, weights and score orientation.
- Use the same reconstruction and Step 02 decisions. Verify perturbed scores against raw scoring, and verify the SHAP reference with an independent reference calculation shared within the request. The temporal path avoids fitting an unused LIME surrogate. Daily explanations still compute the original weighted R², RMSE and MAE.
- Share coalition results across identical feature inputs inside one dashboard render, including the selected day reused by daily review after temporal review. Up to 44 exact feature-vector keys are retained and then discarded. The request fixes the model, preprocessing, saved donor set and policy. Child and date scope is applied before rows reach temporal replay. There is no database, global cache, disk cache or memo shared with another display request. A changed feature row produces a distinct key.

Larger materialized batches, approximate attribution, background reduction, lower-precision calculations and a persistent cache were excluded. The development probe also tested batching without deduplicating the empty and full coalitions. It used more scored rows and higher traced peaks than needed. The reference path remains available for reproducibility and for fallback if a batched evaluation raises an explanation error.

## Protocol and acceptance

The original path was benchmarked on the development data before dashboard activation. The final candidate and benchmark hashes are locked before generating seed **2026112633**, a separate synthetic draw. Eight child histories are selected by a fixed case rule that includes an early short history and later model-flagged and unflagged days. Three separate stress cases exercise stable activity, gradual changes and missing selected-day monitoring using a fresh earlier anchor. Native final records are used as generated. Both arms receive the same controlled records. The seven-day temporal windows, Step 02 policy and model artifacts are fixed.

Each stage uses five stopwatch measurements and separate instrumentation runs for model calls, scored-row counts and traced Python peaks. Warm execution uses a warmed Python process with a fresh request on each sample. The first sample per case and a separate model-load-plus-first-request measurement describe cold request state. These do not flush the operating-system cache. The benchmark reports model-side end-to-end computation, not Streamlit rendering or browser transfer.

The gate requires exact categorical, review and parent-message parity, numerical agreement to the established 1e−8 absolute tolerance, lower warm median temporal-review and end-to-end times, and acceptable traced memory ratios recorded in the lock. The under-500 ms temporal review objective is exploratory. Desktop timing does not establish mobile-device performance.

## Original development measurements

The original-only run measured warm medians of **29.6 ms** for standalone daily XAI, **101.5 ms** for daily with review, **318.5 ms** for temporal XAI, **583.0 ms** for temporal with review, and **887.2 ms** for model-side end to end. Adaptive comparison took **55.7 ms**, profile preparation **93.7 ms**, and model-side history scoring **23.2 ms**. Median traced peaks were **383 KB** for temporal review and **816 KB** end to end. Model load plus the first measured request took **716.8 ms**.

Step 01 previously reported approximately 214 ms for temporal explanation on its unmodified holdout and Step 02 reported approximately 574 ms for temporal review in its mixed benchmark. Those measurements cover different cases and runtime conditions. The paired Step 06 holdout comparisons below provide the controlled comparison.

## Locked holdout measurements

Warm medians pool the four later samples per case. P95 and maxima pool all five samples per case. Traced memory and calls below are medians across case-specific instrumented runs. Figures in KB use decimal units.

| Stage | Warm median, original → optimized | P95, original → optimized | Maximum, original → optimized | Scoring calls, original → optimized | Traced peak, original → optimized |
| --- | ---: | ---: | ---: | ---: | ---: |
| Daily XAI | 17.4 → 7.1 ms | 51.0 → 15.6 ms | 85.8 → 20.5 ms | 1 → 1 | 240 → 223 KB |
| Daily with review | 70.2 → 37.7 ms | 153.0 → 62.9 ms | 186.2 → 70.2 ms | 6 → 5 | 268 → 276 KB |
| Temporal XAI | 240.7 → 29.2 ms | 464.7 → 46.7 ms | 545.5 → 49.2 ms | 15 → 6 | 274 → 404 KB |
| Temporal with review | 331.4 → 38.7 ms | 895.3 → 229.0 ms | 1,299.9 → 270.8 ms | 15 → 6 | 286 → 410 KB |
| Model-side end to end | 625.8 → 167.6 ms | 1,414.5 → 378.5 ms | 2,957.1 → 607.3 ms | 22 → 9 | 553 → 648 KB |

Median scored rows fell from **602 to 485** for daily review, **2,712 to 1,353** for temporal review, and **3,388 to 1,654** end to end. Each removed scoring call also avoids a pipeline imputation and scaling invocation. Median single-call anomaly-history inference stayed around **11–12 ms** for 63 median rows. Adaptive comparison and feature preparation were unchanged code paths and took approximately **33–37 ms** and **42–45 ms**, respectively.

At the coalition level, the original daily request scores 192 rows. The optimized first row scores 169, because it evaluates the full coalition once, and later uncached rows need only 145 after reusing the empty coalition. Original temporal review reached 47 prediction calls in the cases that ran all probes. The optimized maximum in the matched cases was 24 calls.

The temporal-review median improved **88.3%** and its P95 improved **74.4%**. All timed optimized temporal-review runs were below the 500 ms desktop target. The end-to-end maximum remained above 500 ms. Across nine-day, 40–74-day and 90-day histories, median per-case end-to-end times moved from **264 to 112 ms**, **667 to 161 ms**, and **633 to 172 ms** respectively. Temporal replay remains capped by the selected window.

Model load plus first request went from **414.9 to 306.5 ms**. The median first temporal-review sample per case went from **282.4 to 40.3 ms**. A repeated formatter and explanation call with an already populated memo inside the *same* request took **97.9 ms** median and one median scoring call. That probe excludes profile preparation and does not represent a warm cache across page renders.

The optimization improves latency at a transient memory cost for temporal review (**43.3%**) and end-to-end processing (**17.1%**). The temporal-only peak ratio was **1.476**, within the locked **1.50** tolerance. The largest retained memo payload in equivalence cases was **70,080 bytes** for 40 rows plus the 192-byte empty coalition and Python metadata. The joint experiment process reported approximately **274 MB RSS** at completion and **280 MB peak working set** for both arms combined. Those process measurements cannot assign a memory difference to one arm. The deployed bundle remains **6,209 bytes**. The runtime saves no new model, donor or explanation artifact in the application path.

## Equivalence and activation

All **11** fresh-cohort cases passed. The comparison checked the score and prediction used in formatter output, daily signed SHAP, local LIME coefficient and fidelity fields, reconstruction, temporal classification and detector annotations, reliability decisions, adaptive wording and independent urgent matches. Parent messages matched exactly. Cases included flagged, unflagged, missing, limited, stable and isolated explanations. Focused tests separately exercised a sustained pattern with detector annotations, a corrupted recorded score, child/date isolation, changed values on the same date, memory bounds, and offline inference.

The locked acceptance file records both median memory ratios and the latency verdict. After the gate passed, the seven-section dashboard began constructing a fresh optimized request per render. The original calculation path remains available on evaluation errors. There is no approximate method or altered explanation policy.

## Files, limitations and evidence

- New source and test files are `src/behavioral_xai_runtime.py`, `evaluate_xai_runtime.py` and `tests/test_function02_step06.py`.
- Existing files changed are `src/behavioral_explain.py` for sharing original numerical routines, `src/behavioral_explanation_reliability.py` for reusing an independently computed reference, and `ui/pages/_behavioral_upgrade.py` for routing a render through one request.
- New research files include this report and `outputs/xai_runtime_optimization_step06/` containing `development_baseline.csv`, `development_environment.json`, `selection_lock.json`, `equivalence.json`, `holdout_runtime.csv`, `holdout_summary.csv`, `holdout_environment.json`, `acceptance.json`, and `runtime_comparison.png`.

The benchmark uses a finite set of synthetic desktop cases and does not time UI rendering, system startup, or mobile devices. Tracemalloc does not capture all native allocations. Some individual stages and cases have higher measured peaks despite fewer scored rows. The memo never persists between requests.

The active-path focused suite passed **68** cases, including ten new Step 06 checks. The active application and shared-service suite passed **247**, and the profiling suite passed **125** with two skips. The pre-activation application suite passed 237 existing tests. Repository-wide collection reproduced the ten known missing-dependency, cross-package import, and duplicate module errors. There were no failures in the focused or isolated suites.

## Reproduction

```cmd
cd /d E:\RP-J26-DS-316
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
python -m component4_profiling_xai.evaluate_xai_runtime baseline
python -m component4_profiling_xai.evaluate_xai_runtime lock
python -m component4_profiling_xai.evaluate_xai_runtime holdout
python -m pytest -q tests/test_function02_step06.py tests/test_function02_step05.py tests/test_function02_step04.py tests/test_function02_optimization.py tests/test_function02_upgrade.py tests/test_function02_temporal_explain.py tests/test_function02_explanation_reliability.py tests/test_function02_adaptive_profile.py --basetemp .venv/step06-active-focused
python -m pytest -q tests --basetemp .venv/step06-active-app
python -m pytest -q component4_profiling_xai/tests --basetemp .venv/step06-active-profiling
```

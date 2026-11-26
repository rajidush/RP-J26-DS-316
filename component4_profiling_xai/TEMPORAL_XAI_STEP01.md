# Temporal Explainable AI for Function 02

## Objective and implementation audit

Step 01 explains how the deployed One-Class SVM’s feature-group contributions differ across a child’s recorded days. It adds a longitudinal view of the model’s assessment to the existing Parent Dashboard.

| Existing component | Audit finding |
| --- | --- |
| Deployment | The parent page loads `outputs/function02_upgrade/selected.joblib`. It contains the Global One-Class SVM pipeline, 27 inputs, imputer and scaler, fixed anomaly threshold, score calibration, detector settings and 24 background records sampled from normal training rows. The repository also contains older standalone Function 02 model artifacts and separate research variants. |
| Inputs and profiling | Screen, language and calendar features are fused. `behavioral_profile.py` builds chronological per-child histories using prior-only windows and retains missing observations. The global model uses the raw feature design. Personalized comparisons still support other dashboard text. |
| Daily XAI | `behavioral_explain.py` enumerates interventional group SHAP coalitions and fits regression LIME on the same coalition score samples. It recomputes cross-source totals from the coalition inputs. The parent view describes a signed group for the selected day and can mention LIME agreement. |
| Temporal detectors | `behavioral_profile.py` applies EWMA and CUSUM to clipped, standardized anomaly scores. Missing observations or calendar gaps reset the detector runs. The parent view already distinguishes recent evidence from carry-over and can show an earlier alert date. |
| Dashboard and tests | The blue-and-white dashboard already shows daily results, observed changes, temporal flags, guidance and independent urgent rules. It had no across-date SHAP comparison. Existing tests cover causal profiling, detector resets, model-score reconstruction and offline child/date selection with urgent alerts. Archived research previews provide separate explanations and metrics. |

## Method and mathematics

For a scored day *t*, the deployed anomaly score is *s(t) = −score_samples(x(t))*. Higher scores indicate a more unusual pattern in the model. Let *vₜ(S)* be the model’s mean anomaly score across the saved background rows after replacing all feature groups in coalition *S* with the corresponding values for day *t*. The existing explainer evaluates every coalition of the screen, language and calendar groups. It assigns each group its average marginal contribution over group orderings. Derived cross-source totals are recomputed inside every coalition, so their effects can be shared across groups.

The common reference *b = vₜ(∅)* and signed SHAP contributions *φg(t)* satisfy

```text
s(t) = b + sum_g φg(t)
Δs = mean_recent(s) − mean_earlier(s)
Δg = mean_recent(φg) − mean_earlier(φg)
Δs = sum_g Δg
```

Positive daily values push the score higher relative to the shared reference. Negative values push it lower. A positive *Δg* can also occur when a negative contribution becomes less negative. Parent wording that names a stronger unusual-activity factor requires positive daily contributions as well as evidence of an increase. Contributions are measured in model-score units. Raw recorded counts and clinical or causal interpretations are outside this calculation. Another group can offset a positive *Δg*.

The module checks the SHAP full-coalition score and the reconstructed daily score against the recorded model output, verifies a common reference, then verifies window-level reconstruction. Each check uses an absolute tolerance of `1e-8` score units and zero relative tolerance. Invalid attribution for any scored row in an otherwise eligible comparison gives an unavailable result.

### Algorithm and inputs

1. Use the selected child, date, cached deployment bundle, model feature matrix and chronological score history. Compare two adjacent windows of *W* calendar days. The recent window includes the selected date and the earlier window excludes all recent dates.
2. Omit absent calendar days and nonfinite scores. Require at least 5 observations per 7-day window, or 10 per 14-day window, and a scored selected date. A recorded zero is an observation. A coverage gap contributes neither a zero score nor zero SHAP values.
3. For each group, use the earlier median and a band of three scaled median absolute deviations. The band floor is the larger of `4e-8` and one tenth of the frozen training score standard deviation. A recent contribution is elevated when it is positive and exceeds the earlier median by more than this band.
4. A sustained result requires three consecutive elevated days ending on the selected date and a mean contribution increase exceeding the band. Any missing day breaks the streak. An isolated result means exactly one observed recent date had an elevated contribution. A group is withheld when competing candidates fall within the top group’s band. Stable requires all observed recent deviations and all mean differences to fit within their bands. Other cases return unclear, insufficient or unavailable as appropriate.
5. Return a qualitative status and optional group. For local research, the result also contains signed window deltas, score delta, reconstruction error, valid-record counts and supported detector overlap.

### Relationship to EWMA and CUSUM

The existing detectors act on clipped standardized scores. EWMA blends the current standardized score with prior state. CUSUM accumulates deviations after drift and floors its state at zero. Both use run lengths, thresholds and calendar resets. The temporal explanation validates additivity for anomaly scores and their window means. It does not decompose either detector statistic.

A sustained group can carry a separate overlap annotation if EWMA or CUSUM is active on the selected date and the same three-day stretch contains a detector onset or at least two active flagged days. The parent sees that a change alert was also active on the selected day. An old onset on a latched detector with no new evidence does not produce this annotation.

### Dashboard behavior before and after

| Existing section | Before | Step 01 |
| --- | --- | --- |
| Recent Behavioral Pattern | Qualitative summary of recent flags and sequential detector activity. | The same summary plus a short statement about sustained or isolated group evidence, similar contribution patterns, or the corresponding neutral fallback. Earlier alerts retain their dated context. |
| Why Did the AI Say This? | Selected-day SHAP and optional LIME agreement. | The explanation stays at the selected-day level, and the caption explicitly identifies that scope. |

Research numbers and figures are saved separately. Urgent safety rules are evaluated independently before the AI sections render.

## Chronological evaluation

The window selection used endpoints in the original synthetic dataset’s validation split, August 26 to September 11, 2026, with four pre-validation normal daily records as anchors. Each anchor contributed eight cases. The cases were a stable pattern, missing records and one spike, sustained change and gradual ramp for each of the screen and language sources. The harness repeats the anchor across a 28-day calendar, keeps real weekday/weekend transitions, and edits selected, related counts or durations in memory. Gradual changes are evaluated as sustained at the endpoint. Each individual benchmark uses only feature records through that case’s endpoint.

The selection rule maximizes exact scenario-status agreement on those 32 validation cases and selects the shorter window on a tie. The final evaluation uses the separate `step4a_heldout_seed20261020` synthetic file, with September 12–28 test endpoints and pre-test anchors. It includes another 32 controlled cases and one unmodified endpoint per child for 40 children. Holdout results did not determine the window.

Injected source describes what the harness changed. A separate paired model measurement compares manipulated rows with date-matched control rows over the last seven calendar days. It names the group with the largest positive mean SHAP difference only when the mean anomaly score also increased. This uses the same model and SHAP background, so it is a controlled model contrast and an internal check on attribution direction. The daily comparator follows the existing dashboard’s sign rule for flagged versus unflagged days, with LIME direction agreement recorded where a daily group is named.

Stability repeats the analysis after increasing existing positive maximum screen confidence and maximum language risk values by 0.02, capped at 1. The metrics count cases where the perturbation actually moved the selected-day score or any comparison-window score by more than `1e-8`. Timing uses local wall-clock durations for explanation only. Profiling, score-history preparation and page rendering are excluded. Temporal timing includes replay of daily explanations, including their existing LIME fits, and the temporal evidence checks.

### Results

| Group of cases | Status classification | Temporal groups named | Daily groups named | Verified model group, temporal / daily |
| --- | ---: | ---: | ---: | ---: |
| Validation, 7 days | 32/32 | 24/32 | 28/32 | 24/24 / 24/24 |
| Validation, 14 days | 32/32 | 24/32 | 28/32 | 24/24 / 24/24 |
| Controlled holdout, 7 days | 32/32 | 24/32 | 30/32 | 24/24 / 24/24 |
| Unmodified holdout endpoints, 7 days | No scenario labels | 8/40 | 28/40 | No paired target |

- **Window selection.** The 7-day comparison matched the 14-day result on all validation classifications and all 24 named groups while using half as many calendar days. Median runtime in the saved run was 292 ms for 7 days and 902 ms for 14 days. Seven days is the application default.
- **Fidelity.** All 124 eligible temporal comparisons passed the `1e-8` reconstruction tolerance. The maximum absolute error was approximately `8.44e-15`. The held-out paired comparisons agreed within `1.78e-15`. The small residuals are consistent with floating-point arithmetic. The JSON retains their full magnitude.
- **Controlled cases.** The held-out temporal analysis found 16 sustained patterns, 8 isolated patterns and 4 stable cases. The 4 missing-record scenarios returned insufficient history. Both methods matched the injected source and the verified model group in all 24 changed cases. Both methods agreed with the verified positive direction in those 24 cases. All 16 sustained cases overlapped current evidence from both legacy detectors.
- **Unmodified cohort.** The temporal method named a group for 8 children, all isolated. It returned unclear for 26 and stable for 6. No sustained result qualified at these endpoints. Daily and temporal groups agreed for 3 of the 6 children where both methods named a group, which is consistent with the two analyses answering different questions. LIME agreed with the daily group’s direction for all named daily explanations.
- **Stability and cost.** The perturbation changed relevant scores for 16 of 32 controlled holdout cases for each method, and for 28 daily and 40 temporal cases in the unmodified cohort. Each method retained its output in all of those affected cases. The unmodified cohort’s median explanation time was 14.5 ms for daily XAI and 214.3 ms for temporal XAI, with 95th percentiles of 29.0 ms and 413.7 ms, respectively. These are local desktop wall-clock measurements. System load varied across runs.

Results and plots are in [metrics.json](outputs/temporal_xai_step01/metrics.json), [comparisons.csv](outputs/temporal_xai_step01/comparisons.csv), [controlled scenario plot](outputs/temporal_xai_step01/heldout_scenarios.png) and [holdout coverage plot](outputs/temporal_xai_step01/heldout_coverage.png).

### Research contribution and limitations

The contribution is a local, signed, score-reconstructing temporal comparison layered on the deployed model. It distinguishes attribution magnitude, persistence, coverage and detector overlap. The controlled benchmark demonstrated discrimination between isolated and sustained model contributions. Both methods recovered the same factors in the controlled changes, so these results provide no evidence of improved source identification over daily XAI.

The cases and cohort are synthetic. Controlled scenarios use a small set of repeated daily anchors and separate source changes. The unmodified holdout uses one endpoint per child. The high unclear rate illustrates how conservative wording limits coverage. Small confidence and risk perturbations test a narrow form of input stability. Correlated groups, feature replacements and derived totals are properties of the attribution model. No experiment here establishes causes of behavior, safety accuracy, mobile-device performance or parent comprehension.

All temporal computations use only child observations available on or before the selected date. The fixed deployed model and training reference are reused even in retrospective playback of the original development period, so a playback within that period is not a reconstruction of what model was available on the historical date. The final cohort uses a separate synthetic seed. The window comparison is a descriptive use of the trained model and its fixed reference.

## Tests and reproduction

Run from Windows Command Prompt with Python and the repository dependencies installed. Installing dependencies is a separate setup step. The app, tests for local inference and evaluator use local artifacts.

```cmd
cd /d E:\RP-J26-DS-316
python -m pip install -r requirements.txt
python -m streamlit run ui/app.py --server.address 127.0.0.1
```

Open [the local Streamlit dashboard](http://127.0.0.1:8501), select **Behavioral Anomaly**, then choose a child and activity date.

```cmd
cd /d E:\RP-J26-DS-316
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
python -m component4_profiling_xai.evaluate_temporal_explain
python -m pytest -q tests/test_function02_temporal_explain.py tests/test_function02_upgrade.py --basetemp .venv/txai-focused
python -m pytest -q tests --basetemp .venv/txai-root-after
python -m pytest -q component4_profiling_xai/tests --basetemp .venv/txai-component-after
```

The new focused tests cover signed directions, stable and gradual patterns, isolated and sustained shifts, missing dates, stale and overlapping alerts, unsupported or tied explanations, reconstruction against the deployed model, child and date isolation, and as-of replay. The extended Streamlit test blocks DNS and outbound socket connections, checks selected child/date captions, and exercises independent urgent alerts.

The repository-wide `python -m pytest -q` attempts stopped at collection with the same 10 errors and 2 skips before and after the change. The blockers were missing Torch and ONNX, conflicting `src` imports across components, and duplicate `test_pipeline` module names. The profiling component suite ran separately and passed 125 tests with 2 skips on both runs. All 7 original Function 02 upgrade tests passed against the original dashboard module. The post-change focused run passed 18 tests, including 11 new cases and the extended offline dashboard regression.

The separate app and shared-services regression suite (`python -m pytest -q tests`) passed 192 tests at baseline and 203 tests after the change. The focused tests are included in the 203.

## File inventory

| File | Change |
| --- | --- |
| `component4_profiling_xai/src/behavioral_temporal_explain.py` | New attribution validation, window comparison and detector overlap logic. |
| `ui/pages/_behavioral_upgrade.py` | Temporal wording in Recent Behavioral Pattern and explicit daily scope in the XAI caption. |
| `tests/test_function02_temporal_explain.py` | New temporal evidence and deployment replay tests. |
| `tests/test_function02_upgrade.py` | Extended offline dashboard regression. |
| `component4_profiling_xai/evaluate_temporal_explain.py` | New chronological synthetic benchmark. |
| `component4_profiling_xai/TEMPORAL_XAI_STEP01.md` | This implementation report. |
| `component4_profiling_xai/outputs/temporal_xai_step01/{comparisons.csv,metrics.json,heldout_scenarios.png,heldout_coverage.png}` | New benchmark measurements and figures. |

The deployment artifact used for the experiments has SHA-256 `61a5e67700d1a6e499ec610bc23d95b6db318d3b79e1c189d18781d0b85d2ec5`.

## Viva explanations

- **What did you add?** I compare how the same model’s screen, language and calendar contributions changed across two calendar windows for the same child.
- **What does a positive contribution mean?** That group pushed the model score toward more unusual activity relative to the shared reference. I check the sign and the history before naming it in the dashboard.
- **Why seven days?** Seven and fourteen days tied on validation classifications and group identification. Seven used fewer daily replays and ran faster.
- **How do you know it explains the model?** The SHAP values reconstruct every accepted daily score. The signed group differences reconstruct the difference in average scores.
- **Does this explain the change detector?** It explains the underlying anomaly scores and reports when validated contribution evidence overlaps a current change alert. The detector statistics use their own sequential rules.
- **What did the evaluation show?** Both methods found the controlled factors. Temporal XAI added duration and selectively withheld a group on noisier unmodified histories.

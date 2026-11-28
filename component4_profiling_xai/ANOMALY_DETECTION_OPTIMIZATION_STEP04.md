# Step 04 · Anomaly detection optimization

## Deployed model audit

The Parent Dashboard loads `outputs/function02_upgrade/selected.joblib`. Its pipeline uses training medians, standardization, and an RBF One-Class SVM with `gamma=0.001` and `nu=0.01`. The decision is `-score_samples > -14.178161304275191`. The 27 ordered inputs comprise nine screen features, fifteen language features, two cross-source totals and weekend context. The original training script fits normal training days and chooses among training-score quantiles on chronological validation.

The earlier optimization study evaluated 176 configurations. Its consumed cohort gave F1 of 0.640 for optimized global, 0.639 for selected personalization and 0.613 for the hybrid. Step 04 reproduced the deployed comparator at F1 0.667, recall 0.656 and 61 TP / 29 FP / 32 FN / 1,238 TN on those 1,360 consumed days. The earlier comparison table has a misaligned deployed row, so the reproduced values use the saved scorer and predictions directly. The deployed model missed all 21 bullying days and seven of eight volume-spike days in that cohort. The earlier selected challenger missed 20 bullying days and had mean LIME R² of 0.609. Those are challenger results.

## Experiment and leakage controls

- **Development.** The existing development source is used after removing its consumed test rows. All fits and transformations use assessable normal training rows. Later validation labels determine thresholds and selection. IDs group histories and evaluation results. Identifiers, injection metadata and outcome labels are excluded from predictors.
- **Features.** Compare raw inputs, removal of cross-source totals, single-source ablations, same-day nighttime shares and duration per screen incident, bullying share, and existing personalized deviations and activity frequencies. Three prior variability features use the original closed-left rolling standard deviations. Ratios return zero for a recorded zero numerator and denominator, and remain missing for unavailable or invalid values. Current unassessable records are excluded before scoring. The existing hybrid treats absent prior deviations as no positive rarity evidence.
- **Candidates.** Test SVM values of 0.01 and 0.05 for nu and 0.001 and 0.01 for gamma, plus 64-tree Isolation Forests using 128 or 256 training samples per tree. Compare median imputation and standardization with the same pipeline plus signed asinh compression before standardization. Three empirical rarity weights supply hybrid challengers. A threshold-only arm retains the deployed scorer. Eight feature designs are compared under the same standard SVM recipe as well as their best tuned recipe. Direct safety-counter removal is a proxy-sensitivity ablation. Related counts can still encode the same signals.
- **Selection.** Maximize validation F1 subject to the deployed validation FPR plus 0.5 percentage points. Break ties using lower FPR, higher average precision, then deterministic configuration name. Thresholds use strict comparisons and complete tied score groups. The replacement gate requires a positive lower paired child-bootstrap F1 bound, no more than a 0.5-point FPR increase, and acceptable XAI and runtime. F1 of 0.80 is a research target.
- **Holdout.** Verify the candidate, donor pool, deployed artifact, development data and evaluator hashes before generating seed 2026112619. The final 34 dates are evaluated on common assessable rows. Earlier dates supply history only. The same generator family is used with a previously unused independent draw.
- **XAI.** Compare up to 16 common days stratified by deployed flags and eight common final-date child histories. The deployed bundle keeps its reference. The candidate has a separate training-only reference and score center and scale. Five matched training donor subsets provide an alternative-reference sensitivity check. Replaying the existing temporal and reliability policies tests interface and numerical compatibility. It does not establish that the production policy is calibrated for a replacement candidate.
- **Efficiency.** Time single-row and 40-row score calls in the same process. Record median and 95th-percentile time, peak traced Python memory and artifact size. These exclude feature construction, process startup, dashboard rendering and untraced native allocations.

## Validation and feature ablations

The search fitted or evaluated 100 configurations against 680 validation days with 49 positives. The normal-training fit set contained 2,174 days. The deployed threshold gave F1 0.621 and FPR 1.74%, yielding a 2.24% validation FPR budget. With identical scores, threshold recalibration alone brought F1 to 0.634 and FPR to 1.11%, showing that operating point already matters.

| Design | Matched SVM F1 | Best validation F1 |
| --- | ---: | ---: |
| Global with totals | 0.634 | **0.650** |
| Global without totals | 0.645 | 0.650 |
| Screen only | 0.233 | 0.250 |
| Language only | 0.587 | 0.608 |
| Global plus ratios | 0.603 | 0.642 |
| Personalized history | 0.628 | 0.641 |
| Ratios plus personalized history | 0.614 | 0.634 |
| Combined without direct safety counters | 0.529 | 0.529 |
| Rarity-blend hybrid | — | 0.635 |

Language carried more standalone signal than screen activity. Removing overlapping totals helped the matched standard recipe but tied once preprocessing and SVM parameters were tuned. Ratios and historical features did not beat the best global representation at the locked operating objective. The combined design had higher validation AP (0.601 versus 0.578) despite lower F1, so the result is specific to this alert budget. The best Isolation Forest scored 0.568 F1. The chosen challenger retained all 27 deployed inputs and used median imputation, asinh, standardization, a global SVM with gamma 0.01 and nu 0.01, and a threshold of −2.574410180807367 in its own score units. It won the F1/FPR tie over the arm without totals on validation AP.

## Independent synthetic holdout

The locked evaluation used 1,360 assessable days across 40 generated children, including 127 labeled positives. There was no use of these outcomes to choose features, parameters, or thresholds.

| Metric | Deployed | Challenger |
| --- | ---: | ---: |
| Precision | 0.743 | 0.849 |
| Recall | 0.638 | 0.575 |
| F1 | 0.6864 | 0.6854 |
| False-positive rate | 2.27% | 1.05% |
| False-negative rate | 36.22% | 42.52% |
| PR-AUC (average precision) | 0.673 | 0.647 |
| ROC-AUC | 0.727 | 0.729 |
| TN / FP / FN / TP | 1,205 / 28 / 46 / 81 | 1,220 / 13 / 54 / 73 |

The paired child bootstrap 95% interval for challenger minus deployed F1 was **−0.062 to +0.048**. The paired FPR difference interval was **−1.81 to −0.64 percentage points**. Per-child F1 improved for six children, tied for 28 and declined for six. The lower false-positive rate came with lower recall. Neither model reached the 0.80 F1 target.

### Category and error analysis

| Injected category | Deployed hits | Challenger hits |
| --- | ---: | ---: |
| Bullying run | 1/39 | 0/39 |
| Grooming burst | 49/52 | 44/52 |
| Late-night violence | 18/19 | 17/19 |
| Self-harm day | 11/11 | 11/11 |
| Volume spike | 2/6 | 1/6 |

The challenger removed 16 deployed false positives and added one different false positive. It missed nine deployed true positives and recovered one the deployed scorer missed. On normal-labeled days the deployed false positives averaged 2.57 screen incidents and 2.93 language flags, versus 0.47 and 0.97 on its true negatives. Nine deployed and three challenger false positives also had recorded urgent matches. The injected-behavior label and the independent urgent policy define different outcomes, so those overlaps do not make the urgent alerts errors.

Bullying explains 38 of the deployed model's 46 misses and 39 of the challenger's 54. Deployed misses averaged 2.07 bullying flags versus 0.62 on detected positives, consistent with weak ranking of that category in this normal-fit global model. Both models also struggled on the six volume-spike days. The threshold-only validation gain supports an operating-point contribution, while the feature and model results indicate the ranking itself remains weak for some categories. The synthetic generator creates positive labels from emitted injected events, while normal-labeled activity can include genuine simulated signals. These comparisons cannot isolate which differences would persist in actual monitoring.

### Explanation compatibility and efficiency

| Measure | Deployed | Challenger |
| --- | ---: | ---: |
| Maximum daily grouped SHAP reconstruction error | 1.78e−15 | 4.72e−16 |
| Mean held-out local LIME R² across donor probes | 0.495 | 0.539 |
| SHAP leading-group agreement across donor probes | 86.3% | 97.5% |
| SHAP signed-group agreement across donor probes | 87.5% | 85.8% |
| LIME leading-group agreement across donor probes | 78.8% | 85.0% |
| Daily reliability-supported explanations | 14/16 | 13/16 |
| Daily corroborated explanations | 8/16 | 7/16 |
| Supported temporal endpoints | 1/8 | 2/8 |
| Maximum temporal score-delta reconstruction error | 2.78e−15 | 5.55e−16 |
| Median daily review time | 80.9 ms | 91.2 ms |
| Median temporal review time | 289.8 ms | 340.2 ms |
| Median single-row score time | 4.12 ms | 1.42 ms |
| Median 40-row score time | 4.19 ms | 1.56 ms |
| Peak traced Python allocations for 40 rows | 28,816 bytes | 28,720 bytes |
| Serialized bundle size | 6,209 bytes | 23,855 bytes |

The candidate could run the attribution and review routines, with accurate additivity and mixed consistency changes. Higher mean LIME fidelity across alternate donors did not yield more corroborated explanations with the runtime policy and fixed reference. The eight temporal cases are a compatibility sample. The candidate scored faster in this desktop run, while its explanation reviews took longer and the saved bundle was larger. Runtime depends on system load and should not be read as mobile latency.

## Recommendation and limitations

**Retain the deployed model.** Lower false positives under this cutoff are a useful research result, but the candidate misses more anomalies, reduces average precision, and does not clear the locked F1-improvement gate. No candidate is proposed for deployment. The completed holdout is now consumed. Any follow-up study should distinguish the alert-budget trade-off from improvements in ranking and use a new holdout. If a candidate is eventually considered for deployment, its donor reference, LIME/reliability evidence and EWMA/CUSUM calibration require separate validation.

The synthetic benchmark uses one new draw from the existing generator family with full monitoring coverage, shared scenario definitions and sparse category support. The test does not establish performance under observed missingness or in real use. Missingness and leakage are covered by focused feature tests. Scoring memory is a Python-traced measurement, not process memory. Urgent rules, the published model, Function 01, the dashboard, and prior Step 01–03 artifacts were preserved.

## Files and verification

- `evaluate_anomaly_optimization.py` implements the research-only validate and holdout commands.
- `tests/test_function02_step04.py` covers provenance, causality, coverage, cutoff ties, and refusing a mismatched lock.
- `ANOMALY_DETECTION_OPTIMIZATION_STEP04.md` documents this study.
- `outputs/anomaly_detection_optimization_step04/` contains the following new artifacts.

| Evidence | Files |
| --- | --- |
| Locked configuration and provenance | `selection_lock.json`, `candidate.joblib`, `training_donors.csv`, `reference_audit.json` |
| Development | `validation_search.csv`, `validation_ablations.csv`, `validation_matched_ablations.csv` |
| Independent evaluation | `fresh_holdout.csv`, `holdout_metrics.json`, `holdout_predictions.csv`, `per_child.csv`, `per_category.csv` |
| Errors and explanations | `deployed_errors.csv`, `candidate_errors.csv`, `daily_xai.csv`, `temporal_xai.csv` |

The focused suite passed 51 tests, including the four new Step 04 tests. The existing application and shared-service suite passed 226. The profiling suite passed 125 with two skips. Repository-wide test collection reproduced the same 10 known errors involving Torch, ONNX, conflicting `src` packages and a duplicate test module.

## Reproduction

```cmd
cd /d E:\RP-J26-DS-316
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
python -m component4_profiling_xai.evaluate_anomaly_optimization validate
python -m component4_profiling_xai.evaluate_anomaly_optimization holdout
python -m pytest -q tests/test_function02_step04.py tests/test_function02_optimization.py tests/test_function02_upgrade.py tests/test_function02_temporal_explain.py tests/test_function02_explanation_reliability.py tests/test_function02_adaptive_profile.py --basetemp .venv/step04-focused
python -m pytest -q tests --basetemp .venv/step04-app
python -m pytest -q component4_profiling_xai/tests --basetemp .venv/step04-profiling
```

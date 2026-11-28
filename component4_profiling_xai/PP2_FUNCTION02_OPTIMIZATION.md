# Function 02 optimization and validation

**Keep the deployed parent model for PP2.** Validation favored a 14-day One-Class SVM with three selected personal deviations. On the fresh synthetic cohort it tied the optimized global model to three decimals and fell below the deployed model’s F1. All evaluated models missed the 0.80 target. The results support presenting personalization as a tested research hypothesis with a precision–recall trade-off.

## Method and audit

- **Split and leakage controls.** The original seed-42 test and the earlier Step 4A holdout were already consumed. The search used 2,174 covered, labeled-normal training rows from July 1 to August 25, 2026 and 680 later validation rows from August 26 to September 11, including 49 positives. The consumed seed-42 test rows were dropped before profile computation. Median imputation, standardization, Spearman screening, and empirical rarity references used normal training data only. It rebuilt prior-only calendar profiles from raw daily aggregates and excluded label, persona, episode and same-day decision fields from the inputs. Earlier observed anomalies could enter later baselines. The profiles used five prior observations as the minimum, with cold starts handled by the existing imputer.
- **Fresh cohort.** The chosen models, thresholds and temporal settings were hashed and written before generating the predetermined seed 2026112607 with the existing generator. The final 34 dates supplied 1,360 scored child-days, 93 positives (6.84%), 40 generated profiles and 40 labeled episodes. Its first 56 dates supplied history only. The generator family, scenario definitions, full monitoring assumptions and simulated calendar are shared across draws. The new cohort has been evaluated and is now consumed. The separately registered `model_comparison_v1` protocol remains a distinct study.
- **Optimization.** The 176 configurations included Isolation Forest sample-size and feature-subsampling choices, One-Class SVM `nu` and RBF `gamma`, 7/14/21/30-day baseline windows, single-feature additions, group removals, and a rarity-score hybrid. The hybrid calibrated the optimized global score and the maximum positive robust deviation separately against normal training scores, then blended their negative log empirical tails with a validation-selected weight. Precision–recall curves used observed prevalence. F1 selected the strict operating cutoff with fewer false positives as the first tie-breaker. The test curves are descriptive and did not set cutoffs.
- **Redundancy and preprocessing.** Normal training contained 14 absolute Spearman correlations of at least 0.95 among fused raw fields. The pruned global arm reduced test FPR from 1.89% to 1.34% and F1 from 0.640 to 0.634. Three of six robust deviations had a normal-training 90th percentile of zero, which is consistent with sparse features receiving disproportionate weight after standardization. Rolling event frequency can also encode a recent episode’s aftermath. The frequency-only arm achieved F1 0.630 and the deviations-only context arm achieved 0.646. These ablations suggest why adding all history can dilute signal. They do not isolate a causal mechanism.
- **Labels.** The generator marks days with an emitted injected event. Scheduled days without an emitted event remain unlabeled as anomalies. Normal-labeled days can include real detection signals in the simulation, including urgent categories, because the label measures injected activity. Direct urgent, grooming and self-harm fields are plausible generator proxies. Excluding them reduced F1 to 0.410. This sensitivity limits claims about detecting general behavioral changes.

## Final synthetic holdout

| Arm | Validation F1 | Precision | Recall | F1 | FPR | AP | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---|
| Optimized global | 0.634 | 0.696 | 0.591 | 0.640 | 1.89% | 0.664 | 1243 / 24 / 38 / 55 |
| Full personalized | 0.644 | 0.663 | 0.570 | 0.613 | 2.13% | 0.686 | 1240 / 27 / 40 / 53 |
| Selected personalized | **0.674** | 0.711 | 0.581 | 0.639 | 1.74% | 0.666 | 1245 / 22 / 39 / 54 |
| Hybrid | 0.635 | 0.663 | 0.570 | 0.613 | 2.13% | 0.637 | 1240 / 27 / 40 / 53 |
| Fixed deployed reference | Previously selected | 0.678 | **0.656** | **0.667** | 2.29% | 0.664 | 1238 / 29 / 32 / 61 |

The selected SVM used `nu=0.01`, `gamma=0.01`, and 14-day deviations for bullying flags, screen incidents and language flags. The selected-minus-global child-cluster 95% interval was **−0.047 to +0.049 F1** and **−0.79 to +0.54 percentage points FPR** from 2,000 paired bootstrap draws. These are descriptive intervals. The deployed and optimized global arms had identical AP on this cohort. Their observed operating-point difference came from the threshold. The full model improved AP point estimates, though the fixed threshold yielded worse F1.

| Full personalized baseline window | 7 days | 14 days | 21 days | 30 days |
|---|---:|---:|---:|---:|
| Best validation SVM F1 | 0.644 | 0.643 | 0.643 | 0.644 |
| Best validation Isolation Forest F1 | 0.583 | 0.563 | 0.563 | 0.581 |

The same covered validation dates were available for all four windows, with at least five prior observations. The full 7-day arm was locked as the representative. The holdout was used for locked design representatives, with the complete window search reported on validation. The screen-only arm scored 0.217 F1, language-only scored 0.607 and combined raw scored 0.632. The fixed deployed reference scored 0.667. This suggests that combining screen and language helped on this benchmark, with most standalone signal in language. The best exploratory single-feature addition, escalated-screen deviation, scored 0.659. Dropping screen history scored 0.658 with 0.87% FPR. These exploratory findings are candidates for a new study and do not override the locked decision.

### Error analysis

- The selected model missed 20 of 21 bullying days, 7 of 8 volume spikes, 9 of 36 grooming days, 2 of 16 late-night violence days, and 1 of 12 self-harm days. The deployed reference missed all 21 bullying days and the same number of volume spikes, but missed only two grooming days. Improving discrimination on bullying and volume shifts requires better scenario coverage and signal design.
- On its 22 false positives, the selected model averaged 2.91 language flags versus 1.19 on correctly unflagged normal days. Nineteen false positives contained language flags, 19 contained screen incidents, and seven contained an urgent language flag. These errors are relative to the injected-behavior endpoint. Independent safety rule outputs are evaluated against a different policy and remain intact.

### Temporal detection

| Rule | Detected episodes | Mean delay among detections | Normal-day false alert rate |
|---|---:|---:|---:|
| EWMA, alpha 0.70, cutoff 2 | 16 / 40 | 1.56 days | 9 / 1,267 (0.71%) |
| CUSUM, drift 0.75, cutoff 8 | 15 / 40 | 1.60 days | 9 / 1,267 (0.71%) |
| Selected model daily flags, descriptive comparator | 27 / 40 | 0.37 days | 22 / 1,267 (1.74%) |

Validation searched 28 settings with the existing two-consecutive-exceedance and duplicate suppression rules. Delay starts at planned episode onset. A hit requires an alert on an observed injected day. This differs from the earlier upgrade’s date-range credit, so the old 7/22 and 5/22 results are unsuitable for a direct improvement claim. Temporal rules traded detection and speed for fewer alerts on this cohort.

### Explainability

Across 16 flagged and 16 unflagged cases and five seeded normal-training donor samples, exact grouped SHAP mean maximum additivity error was about 1.57 × 10⁻¹⁶ score units. This verifies reconstruction of the specified group game. LIME’s average held-out weighted R² was **0.609**, below the 0.90 target for the challenger. Its mean absolute error at the original observation was 0.0073 score units. The earlier 0.947 used a different model and sampled flagged days only.

| Stability or perturbation measure | SHAP | LIME |
|---|---:|---:|
| Pairwise top absolute group agreement | 98.1% | 100% |
| Mean group sign agreement | 89.2% | 85.0% |
| Mean absolute rank Spearman | 0.831 | 0.791 |
| Top group mean signed score drop with alternate donors | 0.778 | 0.778 |
| Top group absolute drop advantage versus mean other groups | 0.852 | 0.852 |

Perturbations substitute entire feature groups, recompute overlapping source totals when present, and retain within-source history together. The deletion effect measures model sensitivity to this reference, not a causal effect. Tied ranks with zero variance were excluded from the rank correlation. Neither group-level value establishes individual feature importance or probability of harm.

## Emotional linkage and required contract

The inspected educator export has 754 unique sessions. Its child IDs overlap all 40 synthetic behavior IDs, and 653 unique child/local-day keys coincide after timezone conversion. The export has no originating event, incident or trigger key. The educator data harness assigned the synthetic child IDs independently, and the existing Component 3 production contract requires a session ID but no child key. The local database’s optional child field does not establish a shared source of truth for the two historical exports.

To enable verified day-level fusion, supply a trusted local mapping from `session_id` to the same pseudonymous `child_id` used by behavioral events, a timezone-aware session timestamp, provenance and coverage, and the existing enumerated emotional/self-regulation fields. Deduplicate by session ID and use Asia/Colombo calendar dates for the daily join. For event-level attribution, add a verified upstream event ID and source namespace. Preserve unknown coverage or unlinked records as missing. The study performed no emotional ablation.

## Files, verification and PP2 next steps

- New research files are `src/behavioral_optimization.py`, `optimize_function02.py`, `experiments/FUNCTION02_OPTIMIZATION.md`, this report and `tests/test_function02_optimization.py`. The source for the former research interface is archived at `experiments/archived_research_function02.py`. The Component 4 README links to this report. The deployed parent view, Function 01, urgent rules, existing datasets and existing models were preserved.
- New `outputs/function02_optimization/` contains the lock and hashed synthetic cohort, frozen research candidate bundle, validation/holdout PR tables and chart, model/confusion/ablation tables, error slices, per-episode outcomes and XAI measurements.
- This turn’s core Function 02, urgent-rule and Function 01 subset passed **99 tests**. The new optimization and existing upgrade integration subset passed **11 tests**, including the updated research view. The separate lock and test commands completed in order.
- For PP2, report the synthetic F1 target as unmet and keep the deployed model while refining weak categories. Predefine an alert-budget or recall objective before future tuning. Validate on a new sealed cohort with partial coverage and altered generation assumptions, then on authorized labeled events when available. Require an independent representative donor-fidelity check before using LIME statements for a new model. Mobile testing and participant SUS remain outside this study.

The research results are available in the report and saved artifacts. The product launches through [the Parent Dashboard](PP2_FUNCTION02.md#windows-command-prompt).

The completed commands were `python -m component4_profiling_xai.optimize_function02 fit` followed by `python -m component4_profiling_xai.optimize_function02 test`. The saved lock and evaluation prevent accidental reselection or another claim of a fresh test on the same cohort.

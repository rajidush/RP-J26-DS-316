# Step 4A result — 2026-10-20

Executed notebook: `component4_profiling_xai/notebooks/Function_2_Model_Improvement.ipynb`.

The original dataset, Step 3 notebook and saved model are unchanged (SHA-256 checks passed). A separate dataset was generated with seed **20261020**, frozen before development in `data/step4a_heldout_seed20261020`, and first opened for evaluation after candidate and threshold selection were locked. The freeze manifest covers the new data and protected original artifacts. The original test partition was excluded from development. No historical test metrics were used for tuning.

## Protocol

Expanding forward-time folds train through days 35, 49 and 63; validate on days 36–49, 50–63 and 64–73, respectively. Behavioral evaluation begins after a 14-day warmup. All novelty detectors fit normal training days only. Each proposed detector's normal-score quantile is selected from 0.90, 0.95, 0.97 and 0.99 using mean fold F2, then recall, then fewer false alerts. Legacy configuration retains its previously fixed 0.97 quantile. Final fitting uses original development days 15–73. All candidates and final thresholds were locked before fresh holdout evaluation; the winner was not changed afterward.

The comparator refits the existing saved model configuration within each fold, rather than using a model already exposed to validation days. Proposed features are log activity counts, causal 14-day within-child standardized deviations, and prior three-day persistence. The statistical detector takes maximum absolute standardized change. Activity-model inputs omit risk severity and direct urgent-category features. Behavioral labels include volume spikes, bullying runs and late-night violence; grooming/self-harm episode days are excluded from behavioral evaluation. Urgent alerts are independent rule outputs.

## Actual results

Activity SVM was selected on validation: mean F2 **0.4469**, versus **0.3760** for the existing configuration. Validation selection estimates are optimistic; final results below use the frozen fresh dataset.

| Detector | Heldout F2 | Precision | Recall | F1 | False alerts/child | Observed episode detection |
|---|---:|---:|---:|---:|---:|---:|
| Activity SVM (selected) | 0.3414 | 0.2078 | 0.4068 | 0.2751 | 4.575 | 31/47 |
| Activity Isolation Forest | 0.3081 | 0.1366 | 0.4492 | 0.2095 | 8.375 | 33/47 |
| Existing SVM configuration, refitted | 0.2504 | 0.2109 | 0.2627 | 0.2340 | 2.900 | 19/47 |
| Statistical change detector | 0.2829 | 0.1493 | 0.3644 | 0.2118 | 6.125 | 30/47 |

Behavioral metrics cover **2,926 child-days**, including **118 positive days**. False alerts per child cover the 76-day evaluation window, after excluding high-risk episode days; they are not daily rates. Planned-episode detection, including unobserved injected episodes, is **31/52 (59.62%)** for the selected model versus **19/52 (36.54%)** for the comparator. Mean first-detection delay among detected episodes is **0.387 days**, versus **0.158 days**. Volume-spike detection is weak: **3/16** planned episodes.

Separate rules flag any observed urgent language event, grooming, self-harm or screen self-harm imagery. Against eligibility independently reconstructed from raw event records, the rule output matched **201 eligible child-days**, with precision/recall/F1 **1.0** and zero extra alerts. This checks aggregation and policy wiring. It does not measure upstream classification accuracy or independent safety effectiveness.

## Verification

Eight notebook checks passed: future causality, child isolation, label independence, rule fixtures, original test exclusion, behavioral feature separation, protected/frozen hashes, and selection-lock integrity. Notebook execution completed with all assertions and nbformat validation. Execution used a shared Python namespace for sequential cells because nbclient is unavailable; all outputs are embedded.

Regression command: `python -m pytest component4_profiling_xai/tests/test_behavioral_anomaly.py component4_profiling_xai/tests/test_behavioral_dataset.py component4_profiling_xai/tests/test_behavioral_synthetic.py -q --basetemp <writable temporary directory> --junitxml=component4_profiling_xai/outputs/step4a_improvement/regression_tests.xml`.

Result: **64 passed**. Initial run: 60 passed and four setup errors from an unwritable default temporary directory; the explicit temporary directory resolved these errors without code changes.

## Limits and disposition

Synthetic-only, same-generator evaluation; no real-world accuracy or distribution-shift evidence. These are counts of flagged events, not comprehensive device activity. Folds have unequal duration and are averaged equally. Excluding high-risk episode days narrows the behavioral target. Cold-start days are not behaviorally evaluated. Screen activity can still reflect high-risk event exposure. Episode detection accepts any alert during an episode; not every episode has an observable injected event. No uncertainty intervals or independent human labels are available.

The selected approach raises recall and F2 but also raises false alerts and retains low absolute recall. **Production model unchanged; further validation and approval required.** UI unchanged. No commit, push or PR created. The fresh holdout is now consumed: do not use its results for subsequent tuning.

The builder script reproduces the research notebook using the frozen files; it never regenerates the holdout. Re-execution is a reproducibility check, not a new independent evaluation. Any future model-development iteration needs another independently frozen evaluation dataset.

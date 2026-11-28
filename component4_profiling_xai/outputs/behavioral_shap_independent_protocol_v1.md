# Frozen SHAP reliability policy and independent synthetic validation protocol v1

Status at freeze: protocol and runner specified before generating or scoring this validation cohort. The reliability decision is the unchanged v3 implementation. This protocol evaluates synthetic model behavior and explanation evidence.

## Prior development evidence

- [SHAP v1](behavioral_shap_report_v1.md) established score agreement and compared Kernel SHAP with exact group-aware Shapley. Ten fixed test cases were used.
- [SHAP v2](behavioral_shap_expanded_report_v2.md) expanded to 130 rows across the seed-42 and seed-20261020 test splits. It identified lead-rank swaps and sparse masks.
- [SHAP v3](behavioral_shap_reliability_report_v3.md) added the current reliability gate and replayed the same v2 rows. It reported 26 specific, 38 neutral and 66 unflagged outcomes. All prior cases were development evidence for this protocol.

## Frozen policy and inputs

The fitted One-Class SVM retains its imputer, scaler, 27-column input order and score `−pipeline.score_samples(X)`. The strict anomaly decision is `score > −12.515804087687126`. Group-aware attribution enumerates all eight coalitions over these three players:

| Player | Frozen inputs |
|---|---|
| Screen-related recorded activity | `screen_incidents`, `screen_flagged_frames`, `screen_flagged_seconds`, `screen_max_confidence`, `screen_mean_confidence`, `screen_night_incidents`, `screen_blocked_incidents`, `screen_escalated_incidents`, `screen_dismissed_incidents` |
| Language-related recorded signals | `language_flags`, `language_threat_flags`, `language_hate_identity_flags`, `language_bullying_flags`, `language_self_harm_flags`, `language_grooming_flags`, `language_profanity_flags`, `language_urgent_flags`, `language_block_rung_flags`, `language_max_risk`, `language_mean_risk`, `language_night_flags`, `language_audio_flags`, `language_distinct_apps`, `language_distinct_app_categories` |
| Weekday or weekend context | `is_weekend` |

Every coalition copies complete source blocks. It rebuilds `total_risk_events = screen_incidents + language_flags` and `night_risk_events = screen_night_incidents + language_night_flags` before scoring. The combined totals are derived model inputs. The historical Kernel SHAP analysis retains its individual-column game and stays outside the new runner.

- **Backgrounds.** Use the 1,894 eligible, baseline-ready, negative-label original training rows. Select references at (size, seed) (32, 17), (32, 29), (64, 17), (64, 29) without replacement within each sample. The selector rotates children and within-child event-volume terciles. Deletion uses (64, 101) after excluding the union of reference rows. The manifest contains all selected IDs. Validation records enter neither the background nor the support index.
- **Coverage.** Required counts, screen duration and the calendar indicator must be available. Explicit coverage flags, when present, must indicate observation for both sources. Confidence/risk stats can pass through the saved imputer. Insufficient data and assessable unflagged days suppress parent wording, without calculating SHAP for it.
- **Stable lead.** A flagged day needs score reconstruction within the existing `1e-8` tolerance and threshold agreement. One group must lead under all full and support-restricted references, stay positive, and satisfy `min(abs(lead)) − max(abs(other groups)) > 1e-8` over those references. Both full and support-restricted independent donor deletion must agree on the absolute leader and its positive direction. The ranges describe fixed samples.
- **Donor support.** The original normal-training index requires exact matches from at least two distinct children for each of these keys: joint screen incident count, language flag count and weekend; screen incident count, screen nighttime count and weekend; language flag count, language nighttime count and weekend. Apply to target and donor endpoints and paired mixed masks. Keep a common set of donors across six mixed coalitions, or the three single deletions for the separate deletion reference. Each retained set needs at least two distinct children. Recompute and check the restricted attributions. Deterministic structure checks and empirical support checks both apply.
- **Near threshold.** A flagged score whose positive margin is at most the four full backgrounds' expected-score range plus `1e-8` gets the neutral fallback. The detector still uses its saved cutoff. The v3 full-background range was about 0.04546.
- **Wording.** A flagged case passing every check may name the one verified score-increasing group. Failed evidence, sparse support, proximity, or instability yields “The system identified an unusual pattern in the available recorded activity, but cannot reliably identify which recorded factors contributed most.” Insufficient-data and unflagged statuses produce no parent-facing text. Urgent rules and reasons remain separate.

See the [freeze manifest](behavioral_shap_policy_freeze_manifest_v1.json) for fingerprinted code, binary artifacts, previous evaluations, training data, configuration and runtime versions. The runner checks those hashes and required package versions before and after execution. Any policy or scenario revision requires a new protocol and a new untouched evaluation cohort.

## Independent synthetic cohort design

Use a separate normalized-event simulator. The older simulator uses per-child gamma rates, mild weekend multipliers and labelled multi-day episodes. This protocol uses 24 new synthetic IDs, 28 new local calendar dates starting 2026-10-05, independent streams for activity, anomaly labels and outages, and new event and label sampling. The original seed-42 and Step 4a seed-20261020 datasets are kept unchanged. Cohort seed is 2026112301. There are 12 profiles per scenario and 672 days total.

| Parameter | Screen-heavy weekends | Language-heavy bursts |
|---|---:|---:|
| Baseline screen / language rate | 1.40 / 0.65 | 0.45 / 2.30 |
| Weekend multiplier | 2.20 | 0.70 |
| Night assignment probability | 0.22 | 0.35 |
| Positive-label probability per day | 0.07 | 0.16 |
| Screen / language / mixed cluster probabilities, given positive | 0.65 / 0.10 / 0.25 | 0.15 / 0.55 / 0.30 |
| Additional daily screen / language outage probabilities outside blocks | 0.05 / 0.02 | 0.02 / 0.07 |

Profile rates combine lognormal source variation with quiet, steady and variable multipliers (0.60, 1.00, 1.35). Consecutive three-day windows share a Gamma(2, scale 0.5) multiplier, producing correlated daily activity and changes in both event types. Baseline counts are Poisson conditional on these factors. The additional clusters generate 1–4 screen incidents where relevant and 1–3 language flags where relevant. The label is the scheduled cluster indicator regardless of whether its channel remains visible. Injected events have different confidence/frame or risk distributions. Normal language mixes vary across profiles; language app IDs, source mix and C1/C2 event assumptions differ from the original generator.

Each profile gets independently positioned two-day screen and three-day language outages, with extra per-source missing days. The existing daily builder aggregates normalized events. Post-aggregation redaction drops unavailable source features and combined totals, explicitly distinguishing observed zero activity from missing coverage. Input simulation starts at the normalized event boundary, so this protocol exercises daily aggregation and downstream XAI. It does not simulate raw sensor behavior or validate an upstream detector's alerts.

Report the entire cohort. The eight nearest assessable days on each side of the saved cutoff are reported per scenario and overall, ordered by absolute margin, child ID and date. Also report every assessable case on each side within the frozen v3 background span. These panels are descriptive slices of the generated cohort, with overlap between scenario and overall results. Record a shortfall or zero cases if a slice lacks rows. Do not regenerate, select a roving seed, alter labels or force a target score.

## Metrics and denominators fixed before the run

- Precision, recall, F1 and TP/FP/FN/TN use assessable days and the pre-generated synthetic cluster labels. Report missing positive labels and TP divided by all positive labels separately. Undefined ratios are null.
- Report assessment coverage and insufficient-data abstention over all days. Report specific and neutral rates both over all days and among flagged days, plus overlapping reasons among flagged days and specific wording by synthetic label.
- On flagged cases with four full vectors, report unanimity, mean pairwise leading-group and sign agreement and lead changes after restricting donor sets. Report reason counts for sparse target keys and insufficient retained explanation/deletion donors.
- Report returned full and restricted gate reconstructions, maximum absolute error, counts within tolerance, score and threshold agreement, and structural violations. Attribute unflagged and insufficient days only if a later separately frozen research protocol requires it. In this run, the coverage/flag branches intentionally skip attribution.
- Report median, empirical 95th percentile (higher order statistic), maximum wall time and sampled peak RSS and RSS increase over all decisions and among flagged decisions. Time the loaded parent gate with scoring, audits, four reference explanations, deletion and support-restricted checks where applicable. Report gate construction time separately. RSS sampled at about 1 ms includes resident dependencies and model memory and can miss transient allocations.
- Report each metric overall and per scenario. Child days share profiles and activity windows, so pooled rates are descriptive. There is no post-result acceptance threshold or policy adjustment. This evaluation grounds statements about the model on synthetic observations.

## Execution

Use the frozen Python and package versions in the manifest, with Kernel SHAP dependencies available from an ignored virtual environment. Use a new output directory. The runner writes a standalone synthetic CSV, case-level JSON and metrics JSON with artifact fingerprints. Keep this data separate from earlier datasets and never use it as a source of donors. The protocol, runner and manifest remain independent of the output files.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python -m component4_profiling_xai.evaluate_behavioral_shap_independent component4_profiling_xai/outputs/shap_independent_v1
```

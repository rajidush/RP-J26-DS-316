# Independent synthetic SHAP validation results v1

The [locked protocol](behavioral_shap_independent_protocol_v1.md) and [reproducibility manifest](behavioral_shap_policy_freeze_manifest_v1.json) were written before generating or scoring the cohort. The existing v3 gate, exact group-aware explainer, model, threshold, training donors and support index were used without changes. The runner checked the same 28 file fingerprints before and after execution. The [synthetic data](shap_independent_v1/synthetic_daily.csv), [case evidence](shap_independent_v1/cases.json) and [metrics](shap_independent_v1/metrics.json) are separate from v1–v3 development evidence. Regenerating the CSV reproduced its exact SHA-256.

## Cohort and detector behavior

There were 672 synthetic child-days spanning 24 unseen IDs and two 336-day scenarios. Among assessable days, the observed screen/language count means were 1.87/0.85 in the screen-heavy scenario and 0.75/2.16 in the language-heavy scenario. Scheduled positive labels occurred on 21 and 53 days, respectively. Monitoring outages affected 160 days (62 screen-only, 93 language-only, five both), including 16 positive labels.

| Metric | All days | Screen-heavy weekends | Language-heavy bursts |
|---|---:|---:|---:|
| Assessable days | 512/672 (76.2%) | 261/336 (77.7%) | 251/336 (74.7%) |
| TP / FP / FN / TN among assessable | 25 / 39 / 33 / 415 | 3 / 13 / 13 / 232 | 22 / 26 / 20 / 183 |
| Precision on assessable days | 0.391 | 0.188 | 0.458 |
| Recall on assessable days | 0.431 | 0.188 | 0.524 |
| F1 on assessable days | 0.410 | 0.188 | 0.489 |
| Detected / all positive labels, including unassessed days | 25/74 (33.8%) | 3/21 (14.3%) | 22/53 (41.5%) |

The gate's scores and flags matched the existing prediction path for all 672 cases, including null scores and flags on the 160 insufficient-data days. Synthetic cluster labels describe generated activity only. The scenario difference is descriptive and reflects several jointly changed assumptions.

## Gate outcomes and fidelity

| Wording outcome | All days | Screen-heavy weekends | Language-heavy bursts |
|---|---:|---:|---:|
| Specific | 3 | 1 | 2 |
| Neutral fallback | 61 | 15 | 46 |
| Suppressed because unflagged | 448 | 245 | 203 |
| Suppressed because insufficient data | 160 | 75 | 85 |
| Specific rate among flagged days | 4.7% | 6.2% | 4.2% |
| Neutral rate among flagged days | 95.3% | 93.8% | 95.8% |

Specific wording was produced on two synthetic true positives and one synthetic false positive. The two cases naming language came from the language-heavy scenario, and the case naming screen activity came from the screen-heavy scenario. All 61 neutral cases failed at least one target support key. In 59 cases, an explanation reference also retained fewer than two distinct donor children, and deletion donors fell below this minimum for the same count of cases. Three flagged days were within the frozen expected-background score span of 0.04546. All three received neutral wording. Six unflagged days fell within that span and all remained suppressed. All eight of the overall nearest flagged days received neutral wording. The screen-heavy scenario had no flagged case within the frozen span, although it supplied all eight nearest flagged cases requested for its scenario slice.

All 64 flagged days had a unanimous leader across the four full references, with mean pairwise sign agreement of 0.997 across groups. All three cases with support-restricted references retained their leading group. There were no execution, numerical or structural audit failures. All 256 full-reference and 12 support-restricted reconstructions preserved the frozen score and threshold decision within tolerance. Both had maximum reconstruction error 1.78 × 10⁻¹⁵ and zero deterministic mask violations. Full-reference stability and accurate reconstruction therefore coexisted with substantial empirical support failure.

## Desktop cost

Intervals cover one full gate decision, including donor-support checks and any required restricted attributions. On Windows 11 x64, gate and model construction took 1.212 seconds. Median gate wall time was 25.8 ms across all days and 795 ms for flagged days. Flagged p95 was 1.229 seconds and the maximum was 2.597 seconds. Sampled peak process RSS was 172.7 MiB across all decisions. The largest measured per-decision RSS increase was 0.051 MiB. Sampling includes prior resident allocations and may miss transient peaks. These timings do not establish latency, energy, package or memory requirements on a target device.

## Decision and limitations

**NO-GO for parent dashboard integration.** The stable model attributions were usually outside the gate's empirical support requirements under these two scenarios. The protocol exposed that limitation without changing model decisions or relaxing the policy. Specific wording on a synthetic false positive also demonstrates why score fidelity and real-world validity are separate.

This cohort was generated after the freeze using different assumptions and new IDs. It remains a single synthetic cohort at the normalized-event level, with repeated days per profile and shared aggregation semantics. The results do not validate an upstream sensor, the occurrence of real harm, or parents' interpretation of the wording. Before reconsidering integration, collect further independent evidence about model precision and explanation support under representative observation patterns, evaluate comprehension of both specific and neutral wording, and profile the full gate on target devices. Continue to keep urgent-rule reasons separate.

## Regression verification

The Component 4 and existing Streamlit regression suite reported **184 passed in 113.57 seconds**, exit code 0, including four new independent-protocol tests. The manifest and all 28 fingerprints were rechecked. `git diff --check` returned exit code 0 with no output. A separate no-index whitespace check of all nine added files returned zero diagnostics.

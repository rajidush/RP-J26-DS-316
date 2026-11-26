# Step 05 · Temporal behavioral change detection optimization

## Deployed detector audit

Function 02 scores model-assessable daily records with the fixed Global One-Class SVM. The deployed temporal input is `clip((score − center) / scale, −10, 10)`, where the bundle center is −14.7539442 and scale is 0.2867126. The score uses the negative of the SVM's sample score. The recurrence initializes to zero on a child's first valid date and after a calendar gap or unavailable score. Dates are processed separately per child.

EWMA updates the statistic with weight 0.30 on the current value and 0.70 on the previous state. CUSUM adds the current value minus drift 0.50 and floors at zero. Their strict cutoffs are 3 and 6, respectively. Each requires two consecutive threshold exceedances. There is no extra burn-in requirement. An onset fires on entry to the active state. A below-threshold statistic clears the active and run indicators, while the finite accumulated state can persist. Unavailable observations clear the state and report no statistic. Thus recovery on continuously scored days may lag the return to normal activity. The dashboard's current-change interpretation adds an existing recent-evidence check.

I reproduced the original report using its consumed test cohort and historical planned-date-range matching. EWMA detected **7/22** with mean delay **1.43 days** among detections, and CUSUM detected **5/22** with **1.40 days**. Both emitted two normal-day onsets on 627 assessable normal days, or **0.32%**. Earlier PP2's separately tuned challenger reported 16/40 EWMA and 15/40 CUSUM exact-observed-day detections. Those results used a different underlying scorer and thresholds. This study keeps the deployed score and thresholds intact.

## Candidate methodology and lock

- The original chronological validation dates select the champion, with earlier records supplying history. The consumed original test dates are used only for the reference audit. The fresh generator draw uses seed **2026112627** after lock verification. Its final 34 dates supply the primary outcome, with the preceding dates used for warm-up. Step 04's cohort is excluded.
- Score candidates vary EWMA weight and cutoff and CUSUM drift and cutoff. Persistence candidates require either two or three elevated observations in the last five calendar days with contiguous coverage. Alerts fire on entry. A missing observation or calendar gap resets the sequence.
- Feature candidates use the maximum of positive-direction standardized evidence across screen incident counts, nighttime screen incidents, language flags and bullying flags. The fixed baseline is the original 14-day historical profile. The guarded baseline replays the existing Step 03 admission engine and its prior-only deviations. The short/long candidate uses a complete three-day mean ending on the prediction day, compared with the earlier 14-day median and robust scale. The blended score uses the greater of the standardized SVM score and the guarded feature deviation minus one. It can use the score alone when guarded history is insufficient. Every arm respects current model assessability, and reference shortages do not become normal observations.
- A preselected **2% normal-day onset budget** applies to selection. Highest episode recall wins within that budget, then episode F1, lower false alert rate, and deterministic configuration name. The development lock records parameters, cutoffs, matching, recovery, uncertainty and the recommendation gate. The holdout command verifies hashes of the deployed bundle, development sources and both research scripts before generating outcomes.
- A separate controlled cohort reuses earlier normal anchors from the fresh draw. It tests gradual, abrupt, sustained and subtle language changes, recurring flagged days, isolated nonconcerning deviations, legitimate screen-volume shifts, return to usual activity and incomplete monitoring. The cases are controlled stress evidence from shared synthetic anchors. The untouched generator cohort is the primary independent benchmark. No label, category or episode field is used to produce predictor signals.

## Episode and recovery metrics

An onset is matched once to the episode owning that assessable labeled injected day. Scheduled dates without observed injection, early onsets carried into an episode and alert dates after the episode do not receive detection credit. The deployed union deduplicates EWMA and CUSUM onsets on the same day. Each daily SVM flag is an onset for the descriptive comparator. Precision is matched episodes divided by all onsets, so repeat onsets do not raise precision. Recall uses all scheduled episodes, even where a monitoring gap hides every labeled positive. False-alert rates use common assessable normal-labeled days. A day without an injected event can still contain ordinary generator signals.

Delay is measured from planned onset. Counts within one, three and seven days use all episodes as their denominator. Recovery is the first later assessable inactive normal day before the next episode, conditional on detection, and is censored when unobserved. The paired child bootstrap uses 2,000 draws. Controlled streams sharing anchors are correlated, so their bootstrap is descriptive and the generator interval is primary.

## Development result

The search compared 42 candidates on 24 validation episodes and 631 assessable normal days. The deployed union detected 5/24 with a 1.11% false-alert rate. The fixed-feature persistence candidate with the highest recall detected 14/24 but exceeded the budget at 5.55%. The locked combined rule detected 7/24 at 1.27%, using at least two elevated days in five with a cutoff of 1.5. Its comparator is equivalent to elevated SVM evidence at or above 1.5 or a guarded feature deviation at or above 2.5. The nuisance parameter recorded for persistence does not affect this rule.

## Fresh generator holdout

The benchmark replayed 1,360 assessable days across 40 generated children, including 1,247 normal-labeled days. One of the 47 scheduled episodes had no emitted labeled assessable day. It remains in the recall denominator.

| Measure | EWMA | CUSUM | Deployed union | Locked challenger |
| --- | ---: | ---: | ---: | ---: |
| Detected / missed episodes | 15 / 32 | 11 / 36 | 16 / 31 | 20 / 27 |
| Episode precision | 93.75% | 64.71% | 61.54% | 62.50% |
| Episode recall | 31.91% | 23.40% | 34.04% | 42.55% |
| Episode F1 | 0.476 | 0.344 | 0.438 | 0.506 |
| False onsets | 1 | 6 | 7 | 12 |
| Normal-day false-alert rate | 0.08% | 0.48% | 0.56% | 0.96% |
| False onsets per child | 0.025 | 0.150 | 0.175 | 0.300 |
| Mean delay among detected episodes | 1.60 d | 1.55 d | 1.56 d | 0.85 d |
| Detected within 1 / 3 / 7 days | 6 / 15 / 15 | 7 / 10 / 11 | 9 / 15 / 16 | 20 / 20 / 20 |
| Detected episodes with an observed recovery | 10 | 1 | 1 | 13 |
| Median recovery among those observed | 2.5 d | 1 d | 1 d | 4 d |

The paired child-bootstrap recall difference interval versus the deployed union was **+1.85 to +18.42 percentage points**. The paired false-alert-rate difference interval was **−0.16 to +0.97 points**, with a point increase of **0.40 points**. Four children gained an episode detection and 36 tied. The challenger met the numerical recall and alert-budget gates on this cohort. The union's recovery median uses one observed recovery, so it cannot establish that the union recovers faster. The descriptive daily SVM comparator detected 31/47 with 17 normal-day flags. That includes single-day detections and repeated flags rather than a persistence episode policy.

| Category | Total | Deployed union | Challenger | Daily SVM |
| --- | ---: | ---: | ---: | ---: |
| Bullying runs | 9 | 0 | 0 | 1 |
| Grooming bursts | 15 | 13 | 14 | 15 |
| Late-night violence | 7 | 3 | 4 | 7 |
| Self-harm days | 6 | 0 | 1 | 6 |
| Volume spikes | 10 | 0 | 1 | 2 |

**Score versus detector limitations.** Only one of 30 labeled bullying days was flagged by the daily model. The highest standardized bullying-day score was 2.59, below the deployed EWMA cutoff, and guarded feature evidence was available on only three such days. The alternatives did not recover a bullying episode. Daily flag coverage was much stronger for self-harm and late-night violence than temporal detections. That gap is consistent with timing and persistence requirements on concentrated signals. It shows why daily flags and independent urgent alerts must stay available.

## Controlled stress and baseline protection

The challenger detected **20/28** labeled episodes while the deployed union and daily comparator both detected **0/28**. It detected all four episodes in each of gradual, abrupt, sustained, recurring and recovery scenarios. It missed all four subtle bullying and all four incomplete-monitoring episodes. In these deliberately weak controlled changes the daily SVM emitted no positive-day flags. The feature path contributed useful evidence when behavioral contrasts were larger. Only 16/28 were detected within one day, and the mean delay among detections was 2.60 days. All 20 detected episodes had an observed return to inactive after their ends.

There were four challenger false onsets among 832 assessable normal-labeled days, and all occurred on **legitimate shifts**, one on each controlled legitimate stream. The deployed union emitted five false onsets across the controlled cohort, including one on an isolated-spike stream and one on a legitimate-shift stream. The challenger emitted none on the isolated-spike streams. The incomplete-monitoring scenario withheld 20 controlled dates from temporal scoring, and four independent urgent matches on those dates were still reported. The gap tests verify that a missing row or unavailable value resets state and cannot bridge a persistence run.

Under a matched two-of-five, 2.5-cutoff recipe, fixed versus guarded feature evidence detected **16 versus 20** controlled episodes with the same four normal-day false onsets. The guard excludes explicit concerns from admission, which can preserve differences that the ordinary median begins to absorb. On the generator cohort the same comparison detected **18 versus 2**, with false-alert rates of **2.09% versus 0.72%**. Only **425/1,360** generator dates had enough eligible guarded evidence. This is a coverage cost of using the guard alone. The combined candidate remains score-assessable when that history ages out. Contamination and recovery protection therefore do not guarantee better detection across cohorts.

## Temporal XAI and efficiency

Step 01 and Step 02 were replayed on the unchanged SVM score history at eight challenger-onset dates. Three score explanations were supported and five used limited wording. The largest window-score reconstruction error was **2.66e−15**. Three sampled onsets coincided with an existing temporal onset. These are explanations of group contributions to the **SVM score**. The challenger uses an additional statistical signal and its own persistence rule. Feature-leading columns in the diagnostics describe magnitude in that signal and are not SHAP attributions. The tests verify that replaying a candidate leaves the existing score explanation and review unchanged.

For one 90-day history, median precomputed-signal replay took **12.1 ms** for the challenger and **41.5 ms** for the deployed-equivalent EWMA replay. Peak traced Python allocation was **59,842** and **80,983 bytes**, respectively. Preparing all signals, including fixed scoring and guarded history, took **1,034 ms** and **451,622 bytes** for the same stream. The eight score explanation probes had median latency **1,051 ms**. These wall-clock measurements ran during local testing. They cover Python allocations and a research pipeline that builds all diagnostic signals, with no mobile, dashboard-startup or process-memory benchmark.

## Recommendation and limitations

**Retain the deployed detectors.** The combined rule is a useful research challenger. It improved generator episode recall within budget, but it failed every subtle bullying and incomplete-monitoring control episode and flagged all four legitimate shifts. Its additional profile preparation and the small explanation sample leave deployment readiness unverified. A future evaluation should predeclare acceptable false-alert rates for legitimate changes, measure incremental preparation cost, and test richer monitoring histories on a new independent cohort.

The primary holdout is one fresh draw from the existing synthetic generator with full nominal coverage. Missingness and legitimate shifts are imposed only in separate controls with correlated donors. Sparse category counts, planned and emitted episode differences, and observed-recovery censoring limit interpretation. This study does not establish real-world detection performance. The deployed models, original temporal logic, Step 01–04 artifacts, independent urgent rules and seven-section dashboard are preserved.

## Files and verification

New source files are `evaluate_temporal_detection.py` and `src/behavioral_temporal_challengers.py`. The focused tests are in `tests/test_function02_step05.py`. This report and the following artifacts are new.

| Evidence | Files under `outputs/temporal_detection_optimization_step05/` |
| --- | --- |
| Development and lock | `reference_audit.json`, `validation_reference.csv`, `validation_search.csv`, `selection_lock.json` |
| Cohort inputs and labels | `fresh_synthetic.csv`, `controlled_inputs.csv`, `controlled_labels.csv` |
| Generator outcomes | `generator_episodes.csv`, `generator_per_child.csv`, `generator_categories.csv`, `generator_positive_diagnostics.csv` |
| Controlled outcomes | `controlled_episodes.csv`, `controlled_per_child.csv`, `controlled_categories.csv`, `controlled_positive_diagnostics.csv`, `controlled_daily_trace.csv` |
| Summary and explanation evidence | `holdout_metrics.json`, `temporal_xai_compatibility.csv`, `episode_comparison.png` |

The focused suite passed 58 tests, including seven new Step 05 cases. The existing application and shared-service suite passed 230. The profiling suite passed 125 with two skips. Repository-wide collection reproduced the ten known missing dependency, cross-component import, and duplicate-test module errors.

## Reproduction

```cmd
cd /d E:\RP-J26-DS-316
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
python -m component4_profiling_xai.evaluate_temporal_detection validate
python -m component4_profiling_xai.evaluate_temporal_detection holdout
python -m pytest -q tests/test_function02_step05.py tests/test_function02_step04.py tests/test_function02_optimization.py tests/test_function02_upgrade.py tests/test_function02_temporal_explain.py tests/test_function02_explanation_reliability.py tests/test_function02_adaptive_profile.py --basetemp .venv/step05-focused
python -m pytest -q tests --basetemp .venv/step05-app
python -m pytest -q component4_profiling_xai/tests --basetemp .venv/step05-profiling
```

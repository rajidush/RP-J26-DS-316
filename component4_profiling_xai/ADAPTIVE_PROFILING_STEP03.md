# Step 03 · Safety-aware adaptive behavioral profiling

## Audit and objective

Function 02's existing personalized profile in `behavioral_profile.py` looks backwards from the selected day over a 14-day calendar window. Its median, mean, standard deviation, median absolute deviation (MAD), robust Z-scores and seven-day activity frequencies already update as observations enter and leave their windows. It keeps observed zeros and propagates unavailable sources as missing. The median is resistant to single spikes. Repeated unusual or concerning records can still shift that median because the window has no separate admission rule.

The deployed Global One-Class SVM uses the existing fused input design and its fixed model and threshold. The original `profiles` matrix continues to feed model scoring, observed feature handling and guidance. EWMA and CUSUM remain downstream of the same score history. Step 01 compares signed grouped SHAP in two prior-available seven-day windows, and Step 02 checks reconstruction, evidence stability and held-out LIME fit. Their logic and research outputs are preserved. Step 03 adds a separate statistical comparison for **Has Anything Changed?**, so its deviations describe a historical baseline rather than model attributions.

## Candidate methods and selection

A fixed trailing median was the reference. A robust median with gated admission and a bounded, variable set of eligible days was the adaptive candidate. Six policies compared three maximum calendar horizons (14, 21 and 28 days) and confirmation lengths of two or three consecutive records. ADWIN was considered as a window-length detector, but an unconditional change detector cannot distinguish safety-related records from a legitimate distribution change. Adding it would still require an admission policy and a separate multivariate or per-feature change definition. No ADWIN benchmark was run. The evaluated policy combines robust summary statistics and an explicit confirmation rule with bounded per-child state.

The development ordering prioritizes fewer labeled concerning admissions, fewer false drift transitions, more detected legitimate drifts, faster median adaptation, then more coverage. The last tie breaks prefer the shorter maximum horizon and longer confirmation. Four anchors from the original development dataset were selected before the validation split. Controlled histories ended at distinct dates within validation. Each of 12 scenarios ran once per anchor. Configuration and the deployed model's SHA-256 were written to the lock before the separate holdout run.

All six candidates detected 8/8 controlled developmental drifts, admitted 0/384 labeled rows, and raised no false drift transitions in the development scenarios. The 14-day/two-observation policy adapted in a median of 10 days. The 21- and 28-day policies with two observations adapted in 13.5 and 17 days, respectively. The 14-day limit won on adaptation time, with 68.6% usable profile coverage against 74.5% and 78.1% for those longer variants. The winning configuration is responsive because admission responds to observed evidence, although its maximum horizon matches the reference. The five-day minimum and count-scale floor follow the original profile. Recovery length and the two deviation thresholds were fixed design choices across all candidates.

## Method and safeguards

For each child and assessed date *t*, the adaptive snapshot uses only previously admitted records dated in `[t − H, t)`. Histories replay in chronological order, and a gap breaks confirmation and recovery. The six original anchors are screen, nighttime screen and escalated screen incident counts, and total, nighttime and bullying-related language flag counts. Each anchor uses the following summary once at least five jointly usable observations are available.

```text
m_g(t) = median of admitted feature values
MAD_g(t) = median of |x_g − m_g(t)| in admitted records
z_g(t) = (x_g(t) − m_g(t)) / max(1, 1.4826 × MAD_g(t))
```

The baseline comparison and admission rules are separate. A comparison uses the snapshot before the assessed record is considered for admission. The resulting audit records how the assessed record will be treated in later snapshots.

1. **Hard exclusions.** An independent urgent hit or a positive concerning count prevents admission and clears the tentative run. The signals include escalated screen incidents and bullying, threat and hate-language counts. Optional recorded screen violence, explicit visual and sexual harassment counters strengthen the guard when present. Missing optional categories do not establish that the activity was safe. The upload path keeps valid optional counts alongside the existing model inputs. It does not add those columns to the model.
2. **Unavailable records.** The replay requires both sources to be assessable and requires observed anchor and core concern counts. Unavailable days never enter as zeros and interrupt confirmation and recovery. The existing urgent rules can still act on recorded positive signals.
3. **Freeze and recover.** A concerning record freezes updates. Admission can resume on the second consecutive available day without a veto. The first recovery observation stays excluded. During a long freeze, old admissible records can age out and the profile can become insufficient. The admitted history is not padded with concerning or missing observations.
4. **Quarantine and confirm.** A deployed anomaly flag or an absolute robust deviation of at least three starts or continues a tentative batch. Consecutive batch members must remain within two events of its first observation on every anchor. A gap, veto, or incompatible observation discards the batch. Two compatible tentative days with no veto become available to future snapshots. This permits coherent flagged activity to be admitted after confirmation, while isolated spikes and inconsistent flagged runs remain excluded. Lack of a coded harm indicator is a limitation of the guard.
5. **Detect a shift.** The replay compares the current admitted median vector to the last established median vector and records an onset when any anchor moves by at least two counts. The admitted median then becomes the reference for subsequent shifts. This descriptive transition is separate from the deployed EWMA and CUSUM.

The audit includes the decision, feature-level robust deviations and medians, recorded risk fields, coverage, freeze state, admitted indices, and transition marker. The working admission buffer is bounded by the calendar horizon, with a separate pending queue bounded by the confirmation requirement. Returned audit history grows with the number of supplied days.

The fixed baseline remains reproducible. On a guarded comparison failure, the dashboard can show the fixed comparison if it verifies coverage, checks for recorded vetoes and sees only unflagged records in the relevant fixed window. Otherwise it shows an unavailable message. A normal insufficient guarded history gets a neutral message, including after exclusions. This avoids falling back to a contaminated baseline just because the guarded profile aged out.

## Holdout design and measurements

The holdout uses four earlier anchors from the separate existing `step4a_heldout_seed20261020` dataset and ends controlled histories in its test split. Its change sizes differ from development. The 48 cases cover stable activity, gradual and abrupt volume changes, isolated spikes, persistent screen and language indicators, alternating repeated anomalies, missing dates and unreliable coverage, cold start, simulated false-positive flags, recovery and different child activity levels. The analysis uses actual model scores, with explicit flag overrides in the false-positive and repeated-anomaly scenarios solely to test admission. The production model and its decisions were never fitted or replaced.

The reference retains the original fixed rolling median; for research it uses the same median-shift threshold so the detection comparison isolates baseline updating. An admission contamination is a known labeled concerning or alternating abnormal record entering the history. Normalization means that on a labeled day the prior median moved at least halfway from the original level toward the injected increase. Detection delay is measured from the controlled onset; adaptation is the first prior median within one count of the final ordinary volume. Recovery delay measures when the old ordinary center is usable again, and update recovery separately measures when new records are admitted. Coverage is the fraction of supplied records with usable medians, including observed records in cold-start cases. Timing and Python allocation peaks measure the incremental comparison layer after the common feature matrix and fixed rolling statistics are built. They exclude page rendering, model inference and native allocator memory.

### Actual results

| Holdout measure | Fixed rolling profile | Guarded adaptive profile |
| --- | ---: | ---: |
| Labeled concerning or abnormal rows admitted | 384/384 | 0/384 |
| Concerning histories with a normalized median | 16/16 | 0/16 |
| Legitimate drifts detected | 8/8 | 8/8 |
| False drift transitions outside legitimate change episodes | 56 | 0 |
| Median legitimate-drift detection delay | 8.5 days | 9 days |
| Median adaptation delay within one count | 11 days | 11 days |
| Median recovery to an available ordinary baseline | 8 days | 6 days |
| Median delay before admitting new observations after a concern episode | 0 days | 1 day |
| Mean available baseline coverage across cases | 81.6% | 68.6% |
| Median incremental comparison time for one history | 8.3 ms | 46.4 ms |
| Peak traced Python allocation for a 46-day history | 60.9 KiB | 166.9 KiB |

False drift transitions count descriptive median shifts outside the legitimate-volume benchmark episodes. They are not false deployed anomaly alerts. The measured gains concern resistance to normalization and fewer misleading profile shifts. Legitimate adaptation did not speed up on holdout. Detection was half a day later, and coverage declined by 12.9 percentage points because the guard leaves gaps in eligibility. Both methods eventually adapted to all eight legitimate shifts. No claim is made about improved deployed anomaly classification.

In the missing-record scenario, mean profile coverage was 85.0% guarded and 87.5% fixed, and neither method emitted a shift. Both approaches withheld all cold-start medians. Both produced zero shifts for the isolated spike or injected false-positive flag cases. The guarded audit shows that an isolated large spike stayed outside the buffer and matching flagged days could still enter after confirmation.

The [holdout metrics](outputs/adaptive_profiling_step03/metrics.json), [per-case comparisons](outputs/adaptive_profiling_step03/holdout_comparisons.csv), [daily audit](outputs/adaptive_profiling_step03/holdout_audit.csv), [validation grid](outputs/adaptive_profiling_step03/validation_grid.csv), [locked policy](outputs/adaptive_profiling_step03/policy_lock.json) and [baseline plot](outputs/adaptive_profiling_step03/holdout_baselines.png) record the experiment.

## Research contribution and limitations

The research contribution is a prior-only robust baseline whose admission decisions can be audited separately from the deployed anomaly model. In these controlled comparisons it resisted explicitly labeled contamination without sacrificing eventual adaptation to the injected ordinary-volume changes.

All examples are synthetic. The controlled histories repeat four anchor records per split, and labeled harm has clearly specified observed indicators. The safeguards cannot detect unavailable or unmeasured harmful context. Coherent flagged activity can be admitted after confirmation when the records contain no veto. The deliberately alternating anomaly case therefore does not establish the same protection for every form of persistent anomaly. Low-volume incidental signals may also exclude legitimate records, while incomplete categories reduce what can be screened. Long freezes and missing records can reduce baseline coverage. The results do not establish behavior on a mobile device or parent interpretation.

## Parent dashboard

The same seven sections are retained. The adaptive wording appears only in **Has Anything Changed?**, with the existing note about comparing earlier records. The anomaly overview, recent detector summary, explanation, guidance and independent safety alerts use their original paths. **Data Availability** still reports raw historical coverage, so the guarded comparison can have too few eligible records even when the original view has enough recorded days.

The selected-day formatter produced the following messages from actual holdout replay snapshots. The offline dashboard regression also checks the concern and unavailable-record messages in Streamlit.

| Holdout snapshot | Output |
| --- | --- |
| Stable | “This day’s recorded activity was similar to your child’s recent usual pattern.” |
| Accepted distribution shift | “Your child’s usual recorded activity has changed over recent days.” |
| Sustained screen concern | “Recent records that raised a concern were kept out of your child’s usual activity comparison.” |
| Unavailable selected day | “There isn’t enough recorded activity to compare this day with earlier days.” |
| Cold start | “There aren’t enough earlier records to establish a usual activity pattern.” |

## Regression verification

The focused run passed 40 tests, including the new guarded replay cases, the Step 01 and Step 02 tests, and the Streamlit child/date regression with DNS and outbound sockets blocked. The isolated profiling component suite passed 125 tests with two skips. The app and shared-service suite passed 214 tests at baseline and 225 on the final run.

Repository-wide pytest still encountered the ten known collection errors and two skips. These are the missing Torch and ONNX modules, cross-component `src` imports and the duplicate `test_pipeline` test module. The initial Step 03 run exposed a cold-start tuple handling error in the test and renderer. Both now check the six values individually, and the regression confirms the neutral cold-start message.

## Reproduction

```cmd
cd /d E:\RP-J26-DS-316
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
python -m component4_profiling_xai.evaluate_adaptive_profile validate
python -m component4_profiling_xai.evaluate_adaptive_profile holdout
python -m pytest -q tests/test_function02_adaptive_profile.py tests/test_function02_upgrade.py tests/test_function02_temporal_explain.py tests/test_function02_explanation_reliability.py --basetemp .venv/adapt-focused
python -m pytest -q tests --basetemp .venv/adapt-final
python -m pytest -q component4_profiling_xai/tests --basetemp .venv/adapt-component
python -m streamlit run ui/app.py --server.address 127.0.0.1
```

Use the existing dependencies. Open [the local Parent Dashboard](http://127.0.0.1:8501), choose **Behavioral Anomaly**, and select a child and date.

## File inventory

| File | Change |
| --- | --- |
| `component4_profiling_xai/src/behavioral_adaptive_profile.py` | New deterministic, guarded profile and audit. |
| `ui/pages/_behavioral_upgrade.py` | Qualitative wording and screened fixed fallback in the existing comparison card. |
| `ui/services.py` | Keeps optional concerning counters when provided by an existing daily export. |
| `component4_profiling_xai/evaluate_adaptive_profile.py` | Synthetic development, policy lock, final holdout and plots. |
| `tests/test_function02_adaptive_profile.py` | Guard, recovery, adaptation and causal replay tests. |
| `tests/test_function02_upgrade.py` | Extends the existing offline dashboard regression. |
| `component4_profiling_xai/ADAPTIVE_PROFILING_STEP03.md` | This report. |
| `component4_profiling_xai/outputs/adaptive_profiling_step03/` | `validation_grid.csv`, `validation_summary.json`, `policy_lock.json`, `holdout_comparisons.csv`, `holdout_audit.csv`, `metrics.json` and `holdout_baselines.png`. |

## Viva explanations

- **What was already adaptive?** The original median rolled forward as the calendar advanced. Step 03 adds decisions about which observations can enter that window.
- **How can legitimate activity change the baseline?** Ordinary records enter normally. Unusually large or flagged changes enter after two comparable observations without a recorded veto.
- **What happens when concerning observations recur?** Each observation with a documented concerning signal stays out, and recent history can become too thin for a comparison. Consecutive eligible observations allow updates to resume.
- **Does this change the anomaly model?** The adaptive history powers only the parent comparison. Model, explanation and detector paths still receive their original inputs.
- **Where is the main trade-off?** Shorter retention responds faster. It can also leave too little history after a long freeze or gap.

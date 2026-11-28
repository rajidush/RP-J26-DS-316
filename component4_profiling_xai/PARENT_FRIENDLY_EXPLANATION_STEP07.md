# Step 07 · Parent-friendly and evidence-grounded explanations

## Initial audit

The earlier audit found no hardcoded child or date outcomes. Post-audit changes already preserved incomplete safety coverage, routed guidance through the reviewed explanation, tied the admission caption to actual admission, and separated the guarded profile from raw history. Step 07 keeps these conditions.

The selected-day overview repeated the recent change conclusion. The recent card could repeat an anomaly flag for the selected day. The personal comparison gave generic difference wording even though Step 03 already supplied a signed feature deviation. The daily explanation's neutral flagged branch restated the daily prediction, and its helper could name raw attributions if called without a review. Technical labels such as “eligible earlier records” and “contributing factor” added reading burden. Parent guidance needed shorter actions and questions. Coverage and safety statements could be shortened while retaining independent warnings.

## Evidence and wording

| Section | Evidence used |
| --- | --- |
| Behavior Overview | The selected day's unchanged assessment flag, including its unassessed state. Recent changes are described in the recent card. |
| Has Anything Changed? | The existing prior-only guarded comparison. If its abrupt condition is true, the largest absolute robust deviation supplies the recorded feature and increase or decrease. Shift, similarity, and insufficient evidence retain their original state conditions. Captions describe actual exclusion, admission or probation. |
| Why Did the AI Say This? | A supported Step 02 review supplies the group. Its existing positive/negative direction rule determines more or less unusual. Corroboration appears only when the review supplies it. Missing or rejected reviews use neutral wording. |
| Recent Behavioral Pattern | Existing date-scoped EWMA/CUSUM recency evidence supplies continuing-change wording. Coverage supplies the insufficient-history branch. Supported Step 01/02 evidence can describe a one-day or multi-day effect on the AI assessment. Historical dates retain their existing explicit context. The explanation makes no claim about what drove EWMA or CUSUM. |
| How Can You Help? | Actual urgent matches take priority. Other specific wording uses supported AI evidence, independently observed counts or continuing change. General advice and an open question apply when none support specificity. |
| Safety Alerts | Existing independent rule matches and coverage indicators, including selected/latest labels and the earlier alerts expander. |
| Data Availability | Original historical counts, the selected day's assessable inputs, and guarded history sufficiency are separate checks. Assessment input coverage and earlier record gaps are described without repeating a generic missing message for the selected day. |

A personal comparison uses the child's own earlier records. A signed model contribution describes how a factor affected the global AI assessment. An increase in recorded language activity can coexist with language records making the day look less unusual to the AI. Neither statement establishes safety, a cause, or a diagnosis.

## Evaluation protocol

The harness loads the original renderer from the recorded pre-Step-07 Git revision. It runs both renderers with the same model, data and selected date. A capture adapter records the seven displayed sections and front-page alerts without running the browser. Genuine backend computations still run. It separately replays the existing evidence for each case to audit the selected statements.

Development uses the existing synthetic records. The final stage verifies source and model hashes and generates a new cohort with seed **2026112641**. The predefined strata include selected flagged and unflagged records, short and long histories, controlled spikes, sustained changes, ordinary shifts, guarded profile insufficiency, incomplete monitoring, and an urgent match during incomplete monitoring. A capped search for same-source opposite directions uses computed baseline deviation and reviewed model evidence. It does not search wording outcomes.

The automated checks count evidence violations in selected branches, explicit scope for opposite directions, duplicated continuing-change claims, a fixed dictionary of technical terms, word count, and words per sentence. Guidance checks verify supported AI evidence, recorded observations and urgent priority. Each paired case takes five alternated stopwatch measurements and separate instrumentation for scoring calls and traced Python peak memory. Counts do not quantify actual comprehension or usability. Measurements exclude browser rendering and omit native allocations where the tracer cannot observe them.

## Actual before-and-after examples

These are excerpts from paired captured outputs on fresh synthetic cases.

| Section | Before | After |
| --- | --- | --- |
| Behavior Overview | “An unusual activity pattern was detected on this day.” | “The AI detected an unusual pattern in this day’s records.” |
| Has Anything Changed? | “This day’s recorded activity differs from your child’s established pattern.” | “More language-related concerns were recorded than usual for your child.” |
| Why Did the AI Say This? | “The AI assessment found that the language monitoring records made this day look less unusual.” | “In the AI’s assessment, the language monitoring records made the recorded activity look less unusual.” |
| Recent Behavioral Pattern | “Across several consecutive days, the language monitoring records contributed more strongly to the AI’s unusual-activity assessment. A change alert was also active on the selected day.” | “Across several days, the language monitoring records made the activity look more unusual to the AI than before.” The recent card’s continuing-change headline already states the change evidence. |
| How Can You Help? | “Was there anything someone said online that you’d like to talk about?” | “How have your conversations online been going?” |
| Safety Alerts | “No urgent alert was found in the available records for this day.” | “The available records contained no urgent alert for this day.” Both views warn when monitoring is incomplete. A separate test case retains the actual urgent match despite incomplete monitoring. |
| Data Availability | “The usual activity profile has too few eligible earlier records for a reliable comparison.” | “Too few earlier records could be used to establish the usual pattern.” The same case explicitly says earlier activity records and inputs for the daily AI check are available. |

The personal and daily AI excerpts come from the **same selected day**. Both same-source opposite-direction cases now state the personal direction and keep the supported model direction. In the example, the existing guarded comparison found an increase relative to the child's admitted earlier records. The supported negative SHAP contribution described the model’s assessment, even though the personal count increased.

## Locked holdout results

| Engineering measure, 17 paired cases | Before | After |
| --- | ---: | ---: |
| Violations in the prespecified evidence checks | 0 | 0 |
| Mean captured words per case | 206.9 | 181.9 |
| Mean words per sentence | 10.95 | 11.02 |
| Mean matches from the fixed jargon list | 4.41 | 0 |
| Cases repeating the continuing-change claim in two cards | 3 | 0 |
| Cases with a directional personal comparison | 0 | 3 |
| Opposite-direction cases with both directions and scopes stated | 0/2 | 2/2 |
| Cases with evidence-linked topic guidance | 12 | 12 |
| Total predictor calls across all cases | 188 | 188 |
| Median renderer computation | 200.1 ms | 202.3 ms |
| P95 renderer computation | 308.1 ms | 309.4 ms |
| Median traced Python peak | 833,532 bytes | 804,930 bytes |

The average display became approximately **12% shorter** and the tracked jargon and repeated trend claim disappeared in this sample. Mean sentence length was nearly unchanged. These indicators cannot establish parent readability or usability. The baseline already passed the tested evidence conditions, so the study does not claim a drop in unsupported claims.

Both arms rendered identical front-page urgent warning text and date labels in every paired case. The checks preserved warnings for incomplete monitoring, priority urgent guidance and neutral advice when specific evidence was absent. Nine evaluated daily cases had LIME R² below the existing corroboration cutoff, and those cases retained the original corroboration decision. A limited daily case and two unassessable days supplied uncertainty examples. The paired capture required identical scoring call counts for each case, not just in aggregate. The measured median latency rose about 2.1 ms and P95 rose 1.4 ms. Median traced peak fell about 28.6 KB. The small timing differences are descriptive and do not establish a reproducible runtime regression.

The final development rerun using the same code recorded 17 paired cases, 176 predictor calls in each arm and zero evidence-check violations. It reduced the repeated continuing-change claim from 11 cases to zero. Development peak memory moved in the opposite direction to holdout, from 864,265 to 881,461 bytes, which further limits interpretation of small memory differences.

## Verification

The final focused suite passed **76** tests, including eight new presentation cases. The final app and shared-service suite passed **255** tests. The profiling suite passed **125** tests and skipped two. Offline coverage, child/date switching, historical causality and unchanged Step 06 scoring and explanation behavior are covered by the combined runs. An initial run found one stale probation-caption assertion, which was corrected before rerunning. Repository-wide collection reproduced the ten known errors involving missing dependencies and conflicting test imports.

## Files and limitations

- `ui/pages/_behavioral_upgrade.py` contains the presentation changes. The model checksum, scoring, adaptive update policy, detector policies, reliability review, urgent rules, and Step 06 request implementation are preserved.
- `tests/test_function02_step07.py` adds presentation tests. `tests/test_function02_upgrade.py`, `tests/test_function02_adaptive_profile.py`, `tests/test_function02_explanation_reliability.py` and `tests/test_function02_temporal_explain.py` update wording expectations and retain their backend checks.
- `evaluate_parent_explanations.py` and this report are new. The output directory `outputs/parent_friendly_explanation_step07/` contains `development_messages.json`, `development_runtime.csv`, `development_summary.json`, `selection_lock.json`, `holdout_fingerprint.json`, `holdout_messages.json`, `holdout_runtime.csv` and `holdout_summary.json`.

The holdout uses a fresh draw from the same synthetic generator family plus engineered cases. The word, jargon and repetition rules cover prespecified checks. They do not certify all possible text or parent interpretation. The capture adapter times local model-side computation. It supplies no mobile or parent usability evidence, and no SUS or satisfaction score was measured.

## Reproduction

```cmd
cd /d E:\RP-J26-DS-316
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
python -m component4_profiling_xai.evaluate_parent_explanations validate
python -m component4_profiling_xai.evaluate_parent_explanations lock
python -m component4_profiling_xai.evaluate_parent_explanations holdout
python -m pytest -q tests -k function02 --basetemp .venv/step07-verified-focused
python -m pytest -q tests --basetemp .venv/step07-final-app
python -m pytest -q component4_profiling_xai/tests --basetemp .venv/step07-profiling
```

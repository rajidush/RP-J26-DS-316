# Expanded behavioral SHAP validation v2

## Scope

This synthetic evaluation extends [the v1 pilot](behavioral_shap_report_v1.md). It runs the saved median-imputer, scaler and One-Class SVM with the saved 27-column order. The score is `−pipeline.score_samples(X)`, and the flag is set when that score is strictly above `−12.515804087687126`. The method evaluates eight exact coalitions for screen-related recorded activity (9 inputs), language-related recorded signals (15), and weekday or weekend context (1). It keeps each source's features together and rebuilds both cross-source event totals before scoring. SHAP values describe this model and reference sample. They do not identify real-world causes or substantiate harm.

The study uses the original synthetic test split and the test split from the independently seeded Step 4a synthetic dataset (seed 20261020). Both use generator version 1.0.0. The second source was already used in a previous model evaluation, so this is a repeat across simulated samples, with shared generator assumptions. Dataset identity was verified against the model metadata and Step 4a freeze manifest, taking the Windows line-ending conversion into account. Input hashes before and after evaluation are recorded in the output JSON.

## Selection and checks

- **Main panel.** Use the existing coverage gate. Exclude the ten original pilot days from the original split. For each source, reserve the six smallest absolute margins separately for flagged and unflagged rows, with ties broken by child and date. Draw up to 12 more per TP, FP, FN and TN stratum without replacement using seed 22026. Supplement anomaly categories with fewer than four selected days where additional examples exist. Selection is verified after input row reordering. Balanced labels and boundary sampling make the panel unsuitable for estimating population prevalence.
- **Leading-group challenge.** Outside the main panel, sample candidates with screen evidence and zero language flags, and quiet days with zero combined events. Use seed 22027, screen up to 32 per pattern per source against the 32/17 reference, and retain up to two confirmed screen leaders or calendar leaders. Keep this deliberately enriched panel separate when interpreting leading-group frequency.
- **Attribution and stability.** Compute all eight grouped coalitions with four original normal-training backgrounds (32/17, 32/29, 64/17 and 64/29, where the second number is the seed). For each case, compare the six pairs of runs for leading absolute group, top-two set overlap, signs, group-attribution span and expected-score span. Require reconstructed score error at most `1e-8` and agreement with the strict cutoff.
- **Fidelity and masking.** Compare group ranks and the leading attribution's sign against whole-block deletion with a disjoint 64-row original normal-training donor set. This checks the response to that replacement intervention, which can differ from an average over Shapley coalitions. Run the existing deterministic audit on all scored coalition rows. For mixed coalitions, also measure original training support for (i) screen present, language present and weekend and (ii) exact screen count, language count and weekend, with fewer than five training occurrences treated as sparse. Sparse support does not prove a mask is impossible. Passing these checks does not verify the joint distribution of confidence, severity, apps or other cross-source signals.
- **Parent wording.** Independently verify each named group and direction against all four runs and the donor deletion, using the formatter's top-two and 0.01 score-unit materiality rules. For flagged cases, withhold donor evidence and test a single background to exercise the neutral fallback, and mark language coverage absent to verify suppression. These stress tests modify copies in memory.
- **Synthetic dominance diagnostic.** Inventory screen and language event presence by category across both eligible test splits, then compare each category's leading group in the main panel. Pair observed screen-only one-event days with language-only one-event days without replacement within dataset and weekend status, using seed 22028. Compare frozen scores relative to the mean score of quiet original training days in the same weekend stratum. This controls combined event count and weekend status. Timing, severity, event subtypes, child, baseline rates and model feature geometry can differ.

## Results

The [case-level results](behavioral_shap_expanded_v2.json) contain 130 days across two datasets, 65 per source, covering 29 synthetic child profiles and 18 labelled episodes per source. Each source contributed 63 main-panel and two screen-led challenge cases. The combined sample included 29 TP, 35 FP, 26 FN and 40 TN. It included 12 flagged and 12 unflagged boundary days. Flagged margins ranged from +0.01605 to +0.27729, and unflagged margins from −0.04727 to −0.00945. The distance of the sixth flagged original day from the cutoff reflects the available observations after excluding the pilot.

| Measure | Result |
|---|---:|
| Frozen score and strict threshold agreement | 520/520 |
| Maximum absolute reconstruction error | 3.55 × 10⁻¹⁵ |
| Explanation execution or reconstruction failures | 0 |
| Cases with the same leading group across all backgrounds | 122/130 (93.8%) |
| Mean pairwise leading-group agreement | 0.964 |
| Mean pairwise top-two group Jaccard | 0.995 |
| Mean pairwise group-sign agreement | 0.995 |
| Leading group agreed with independent donor deletion | 498/520 (95.8%) |
| Sign of leading attribution agreed with deletion for that group | 520/520 |
| Mean absolute group rank Spearman versus donor deletion | 0.953 |
| Scored coalition rows violating deterministic relationship checks | 0/199,680 |
| Mixed masks with unseen presence/weekend combination | 0/149,760 |
| Mixed masks with exact counts/weekend observed fewer than five times | 31,611/149,760 (21.1%) |

The expected score varied by 0.04546 across the four backgrounds. Median contribution spans across the four configurations were 0.02346 for screen, 0.04664 for language and 0.000544 for calendar. The largest spans were 0.03356, 0.05671 and 0.000827 respectively. A baseline difference affects attributions, while the saved score and decision stay fixed.

Donor deletion disagreed with the leading group in 22 configurations across ten days. For example, on the original C002 day of 2026-09-18 (TN), the same group led across all four backgrounds, but a different group had the larger deletion effect in every configuration. On independent C007, 2026-09-13 (FP, margin +0.02553), the leading group split two-to-two across backgrounds. Both contributions remained directionally verified, so the formatter named both. The wording made no explicit ranking claim. Among near-threshold days, deletion agreed with the attribution leader in 94/96 runs. Among screen-led challenge days, it agreed in 14/16. These misses limit any claim that the leading group is reliably the most influential in every case.

The structural mask audit checks arithmetic totals, subcount bounds, and confidence/risk summaries. The added support check was stricter in a different way. Among flagged cases, 22,951/73,728 mixed rows (31.1%) had a sparse exact count/context tuple. The highest case/configuration rate was 78.1% on original C031, 2026-09-19. The formatter's existing gate does not incorporate these empirical support rates. Matching or conditioning donor selection on compatible context warrants a separate check before showing explanations to parents.

### Language-related dominance

At the primary 32/17 setting, language led in 91/126 main-panel days (72.2%) and screen in 35/126 (27.8%). The four challenge additions were screen-led under that setting. Two independent screen challenges changed leader across backgrounds. No calendar-led case occurred in the main panel or in 64 screened quiet candidates. The calendar group's contribution moved little across configurations and these results give limited evidence about it as a leading group.

| Synthetic label | Main days | Language leads | Screen leads |
|---|---:|---:|---:|
| Bullying run | 19 | 15 | 4 |
| Grooming burst | 13 | 13 | 0 |
| Late-night violence | 8 | 1 | 7 |
| Self-harm day | 8 | 8 | 0 |
| Volume spike | 7 | 5 | 2 |
| No injected anomaly label | 71 | 49 | 22 |

The generator injects language records for grooming, bullying and self-harm scenarios. These three types make up 84 of 108 positive-labelled test days in the two full split inventories, and all 84 had language flags. The late-night violence pattern injects screen activity with optional language records, and seven of eight selected days were screen-led. The volume-spike pattern scales normal activity and produced both lead groups. Even among the 1,252 negative-labelled days in the two full inventories, language flags were present on 739 days (59.0%) and screen incidents on 290 (23.2%). These frequencies and the category breakdown support a generator and feature-distribution contribution to the pilot's language dominance.

The matched comparison yielded 50 pairs (24 original, 26 independent). Median score shifts relative to quiet training days were −0.205 versus −0.295 for screen-only and language-only in the original split, and −0.203 versus −0.306 in the independent split. The paired median language-minus-screen difference was −0.085 and −0.098 respectively. Language yielded the higher anomaly score in only 4/50 pairs. Restricting the comparison to pairs with two negative labels gave the same result in 4/49 pairs. Ordinary one-event language days therefore did not show a blanket upward score bias versus matched one-event screen days in this diagnostic. Both sources' typical one-event days scored below the quiet reference.

The model still has different groups of correlated inputs, rare language categories, and redundant totals. Its fitted scaling and support vectors may respond differently to specific signals. Matching one event and weekend status does not equate rarity, night timing, category, confidence or severity, and this frozen-model design cannot independently identify a generator effect and a feature-design effect. Replication across different generators or appropriate real observations, and pre-registered source and subtype comparisons, would give stronger evidence.

### Parent grounding

All 130 outputs passed the independent wording check. All 64 flagged days had at least one material group verified against the four backgrounds and donor deletion. Forty-seven named two groups, 17 named one, and 16 included score-decreasing wording. No natural fallback occurred. Withholding the donor effects or keeping only one background triggered the neutral fallback in each of the 64 flagged cases. Marking language coverage absent suppressed explanations for all 64. The 66 unflagged days generated no parent text.

Actual output for the original C033 day of 2026-09-25 (flagged, negative synthetic label):

> The model identified an unusual pattern in the available recorded activity. Compared with typical recorded days, screen-related recorded activity increased the model's score for unusual activity. Compared with typical recorded days, language-related recorded signals decreased the model's score for unusual activity.

That wording describes the score even on a synthetic false positive. It does not establish that the statistical difference corresponds to a meaningful change for a child. Automated checks verify mechanical grounding and fallback behavior. These checks do not establish parent comprehension or the suitability of the underlying donor substitutions.

### Desktop cost

Windows 11 x64, Python 3.13.7, NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.6.1 and psutil 7.1.3. OpenBLAS/OMP thread limits were set to one. Timings cover a loaded explainer, the exact masking work, the deterministic audit and pipeline scoring. They exclude the independent deletion checks, empirical-support audit, parent formatting and cold start. Peak process RSS was sampled at 1 ms, so brief peaks may be missed. RSS includes the loaded Python research process and retained allocations.

| Background size | Runs | Median / 95th percentile / max seconds | Peak RSS / max RSS increase | Model calls / rows per explanation |
|---|---:|---:|---:|---:|
| 32 | 260 | 0.085 / 0.139 / 0.189 | 170.91 / 0.043 MiB | 9 / 257 |
| 64 | 260 | 0.087 / 0.136 / 0.287 | 170.91 / 0.063 MiB | 9 / 513 |

The parent evidence gate used four explanations per day, plus a separate donor check. The table measures single runs. Device latency, memory, battery use and native package compatibility remain unmeasured. The grouped path uses the existing NumPy/pandas/scikit-learn stack. SHAP and its dependencies are still required for the previous Kernel SHAP benchmark and its tests.

## Decision

**NO-GO for parent-facing dashboard integration at this stage.** The expanded study supports mathematical correctness and mostly stable model attribution on these synthetic records. It exposes plausible-looking masks with sparse training support, rank reversals near the cutoff, and a formatter that can return a specific explanation despite both conditions. The test set still uses two seeds of a shared generator. Its false positives illustrate how faithful score explanations can nevertheless be read as evidence of a real safety problem.

Before reconsidering, test a donor-support policy with criteria fixed in advance and compare how often it abstains or changes named groups. Add cases with different generator assumptions or suitable representative records, verify that unstable or weakly supported wording is suppressed, and test parent comprehension of score-increasing and score-decreasing language. Profile periodic calculations on representative devices before any on-device roll-out. The expanded evaluator and results can support offline review while the dashboard decision is pending.

## Reproduction

From the repository root, use the requirements declared in `component4_profiling_xai/requirements-shap.txt` and choose a new output path.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python -m component4_profiling_xai.evaluate_behavioral_shap_expanded component4_profiling_xai/outputs/behavioral_shap_expanded_rerun.json
python -m pytest -q --basetemp=.venv/pytest-shap-expanded component4_profiling_xai/tests tests/test_ui_f2_services.py tests/test_ui_pages.py
git diff --check
```

## Regression verification

Component 4 and the existing Function 1 and Function 2 Streamlit regressions reported **169 passed in 74.02 seconds**, exit code 0, with SHAP 0.49.1 installed. `git diff --check` exited 0 with no output. A no-index check of the three new files found zero whitespace diagnostics. The original pilot results and protected input fingerprints matched before and after the run.

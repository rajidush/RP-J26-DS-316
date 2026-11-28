# Function 2 behavioral SHAP research pilot v1

## Frozen detector and existing application

The tested artifact is `models/behavioral_anomaly_pipeline.pkl`, with its feature order, threshold and provenance in `models/behavioral_anomaly_metadata.json`. It applies median imputation, standard scaling, and a One-Class SVM (`nu=0.01`, `gamma=0.001`). The explained function is

```text
f(x) = −pipeline.score_samples(x)
t = −12.515804087687126
margin(x) = f(x) − t
flag(x) = f(x) > t
```

The existing `score_observed_days` path gates on observed counts, screen duration, calendar input and available coverage indicators. Confidence/risk statistics can be imputed. The offline API uses the same gate and returns no explanation for unassessable rows. `baseline_ready` applies to the background selection because it was part of the training filter. In the Streamlit page, `ui/services.py` prepares the behavioral report and calls `explain_day` for standardized feature deviations. Those deviations describe inputs, while SHAP explains the continuous score. The urgent-rule path calls `urgent_alerts.py` and is independent of the model score and SHAP. The page and services were left untouched.

The saved metadata reports validation F1 0.645 and PR-AUC 0.595, and original test F1 0.615 and PR-AUC 0.597 (32 TP, 19 FP, 21 FN, 608 TN). A later synthetic holdout compared several detectors and recorded legacy SVM F1 0.234 across 2,926 rows. This pilot explains the frozen legacy SVM from the dashboard path. It uses the original test split for explanation analysis only. No model selection or training was performed.

## Attribution games

For background B and feature subset S, Kernel SHAP estimates Shapley values of the empirical donor-replacement game

```text
v_K(S) = (1/|B|) Σ_b f(x_S, b_{−S})
f(x) ≈ v_K(∅) + Σ_i φ_i
kernel(M, k) = (M−1) / [C(M,k) k (M−k)]   for 0 < k < M
```

The identity link leaves contributions in original score units. SHAP 0.49.1's KernelExplainer uses sampled coalitions and an unregularized weighted fit in this experiment. Each omitted subset comes from a complete donor row, preserving relationships among the omitted columns but potentially splitting dependencies across the observed/omitted boundary. Positive values raise the score relative to the background score and negative values lower it. They do not measure the real-life effect of changing an input.

**Aggregated Kernel SHAP** sums the fitted individual attributions by the display buckets below. It has the same reconstruction as the feature attribution vector. The combined totals bucket is available only for analysis.

| Bucket | Inputs and dependencies |
|---|---|
| Screen-related recorded activity (9) | Incidents, flagged frames and seconds, maximum and mean confidence, nighttime incidents, blocked, escalated and dismissed incidents |
| Language-related recorded signals (15) | Flag count, threat, hate/identity, bullying, self-harm, grooming, profanity, urgent and block-rung flags, maximum and mean risk, nighttime and audio flags, distinct apps and app categories |
| Weekday or weekend context (1) | Weekend indicator |
| Combined recorded event totals (2) | Total and nighttime risk events. Each is the sum of the corresponding screen and language counts. |

**Group-aware Shapley** uses the first three buckets as players. For each coalition the method copies the included blocks from the target and the other blocks from one donor. It recomputes both totals before scoring, so each of the 27 model inputs is consistently supplied and neither total becomes a separate parent-facing explanation. Night counts, response signals, and language app/channel context stay with their respective source counts and summaries. A standalone night or safety group would split those relationships. For G groups, the exact empirical game is

```text
v_G(S) = (1/|B|) Σ_b f(rebuild_totals(x_S, b_{−S}))
ψ_g = Σ_(S ⊆ G∖{g}) [v_G(S∪{g}) − v_G(S)] / [|G| C(|G|−1, |S|)]
f(x) = v_G(∅) + Σ_g ψ_g
```

Eight coalitions suffice. This calculates group-level values for a different, constrained replacement game. Partition SHAP produces within-hierarchy Owen values. A feature tree alone does not express the two overlapping arithmetic dependencies or make masked values obey the equations. A custom masker could recompute totals, but exact enumeration is simpler for this three-player comparison. Within-group Owen values were outside this pilot.

## Evaluation protocol

- Training pool consists of 1,894 baseline-ready, assessable normal training rows. Random seeds 17 and 29 select 32 and 64 observations without replacement. The selector rotates through children, cycling through within-child event-volume terciles with seeded starting offsets. The JSON stores every child/date donor pair, selection seed and observed training correlations. The perturbation donor set uses seed 101, 64 normal training rows and excludes every row in any explanatory background.
- The fixed pilot has two observations per TP, FP, FN and TN stratum chosen with seed 42, plus the two distinct observations closest to the strict cutoff. The comparison pilot uses 256 Kernel masks (seed 137) per background and repeats the 32/17 setting with 512 masks. Group-aware runs enumerate all masks for each background.
- Reconstruction succeeds with absolute error at most `1e-8` and an identical strict threshold decision. Each score is also checked against the existing scoring path. A controlled linear scorer tests positive and negative attribution directions independently of reconstruction.
- Stability uses pairwise top-three absolute feature Jaccard, top-two absolute group Jaccard and all-group sign agreement. Score differences between reference expectations are also recorded. Jaccard of one means identical membership. A top-two overlap among three groups has a higher chance baseline than among four.
- Fidelity uses a disjoint training donor set to replace one source block at a time, recomputing derived totals. For each group, compare its signed attribution with the score drop `f(x) − mean(f(masked x))`. Compare the largest absolute contribution with the largest absolute deletion effect across the three shared groups. Group deletion starts from the full target, while Shapley averages across contexts. Interactions can cause disagreement even with exact values. Shared perturbation semantics make this check more directly aligned with the group-aware game.
- The audit identifies broken sums, source subcounts exceeding totals, mean exceeding maximum, and nonzero confidence/risk with zero events. It measures violations of these rules. Groupwise substitutions can still break cross-source and weekday dependencies that the audit does not cover.
- The parent formatter is restricted to flagged, assessable days. A group must remain among the top two across all supplied runs, with stable sign and at least 0.01 absolute score units in every attribution and the disjoint donor deletion result. All runs must reconstruct and pass masking checks. This heuristic is an offline screening rule without a validated operating point on real records.

## Pilot results

The [run data](behavioral_shap_pilot_v1.json) stores the ten fixed observations, all 90 explanations, the individual attributions, timings, deletion effects and pairwise stability statistics. There were no explanation failures. All stored scores exactly matched the existing scoring path, and all reconstructed scores agreed with the strict model decision. The closest pilot day was 0.00577 score units below the threshold. Both nearest-cutoff observations were below it, so this stratum tests one side of the boundary. Artifact, metadata and dataset hashes matched before and after the run.

| Measure | Kernel SHAP / grouped sums | Exact group-aware Shapley |
|---|---:|---:|
| Explanations | 50 | 40 |
| Maximum absolute reconstruction error | 2.66 × 10⁻¹⁵ | 1.78 × 10⁻¹⁵ |
| Audited rows violating at least one rule | 381,275 / 657,650 (58.0%) | 0 / 15,360 |
| Mean top-three feature Jaccard (median, min) | 0.915 (1.000, 0.500) | Feature ranks unavailable |
| Mean top-two group Jaccard (median, min) | 0.907 (1.000, 0.333) | 0.967 (1.000, 0.333) |
| Mean group sign agreement | 0.965 | 0.978 |
| Largest shared-group attribution matches largest donor-deletion effect | 50 / 50 | 40 / 40 |
| Sign agreement for top shared-group attribution and deletion | 50 / 50 | 40 / 40 |
| Mean Spearman correlation of absolute shared-group ranks versus deletion | 0.950 | 0.938 |

Across backgrounds, the expected score differed by up to 0.0455. Kernel SHAP's largest absolute individual attributions most often belonged to total risk events (14/50), grooming flags (10/50) and profanity flags (7/50). These are model inputs that vary with synthetic records. They do not establish that an event occurred as labeled or that a child was harmed. The two methods ranked language-related recorded signals first among the three shared groups in 40/40 paired runs. Repeated configurations of a day are dependent observations, and results for the leading group should be read in light of the limited case mix.

The training pool had 40 children and no constant or exactly duplicate features. Absolute Pearson correlations included flagged frames with seconds (0.997), screen maximum with mean confidence (0.996), language maximum with mean risk (0.996), and distinct apps with app categories (0.973). Kernel SHAP broke the total-event equation in 209,012 scored rows and the night-total equation in 62,699. Other common violations included app counts above language flag counts and mean language risk above its maximum. The constrained game resolves these audited failures at the cost of within-block detail. Its higher top-two overlap also reflects the smaller number of ranked groups. Deletion results offer mixed evidence for an overall improvement because both methods matched the leading group and direction, while the aggregate Kernel ranking had a slightly higher mean correlation with deletion. Additivity is a mathematical check and gives no evidence about masking plausibility on its own.

### Computational cost

Windows 11 x64, Python 3.13.7, SHAP 0.49.1, scikit-learn 1.6.1, NumPy 2.4.6, pandas 3.0.3, psutil 7.1.3. OpenBLAS and OMP thread counts were set to one. Timings include masking audits, scoring and attribution after dependency import, model load and background selection. They exclude the separate donor-deletion validation, which scores another 193 rows in four calls per record. Peak process RSS was sampled at 1 ms and includes imports, the loaded pipeline and retained allocations. The increase column is the maximum observed increase from an explanation's starting RSS. Runs shared the desktop with regression work, so timings should be treated as descriptive.

| Method | Background / mask budget | Runs | Median / max seconds | Peak RSS / max increase (MiB) | Pipeline calls / scored rows per run |
|---|---|---:|---:|---:|---:|
| Kernel | 32 / 256 | 20 | 0.338 / 1.099 | 259.5 / 7.00 | 4 / 8,226 |
| Kernel | 64 / 256 | 20 | 0.577 / 1.328 | 266.3 / 14.39 | 4 / 16,450 |
| Kernel | 32 / 512 | 10 | 0.502 / 1.162 | 266.3 / 13.56 | 4 / 16,418 |
| Group-aware | 32 / 8 exact | 20 | 0.516 / 1.516 | 252.7 / 0.02 | 9 / 257 |
| Group-aware | 64 / 8 exact | 20 | 0.449 / 1.634 | 252.7 / 0.02 | 9 / 513 |

Doubling the Kernel background at 256 masks approximately doubled the scored rows and raised median runtime by 71%. The group-aware method scored about 32× fewer rows at either size. Its eight separate coalition batches carry fixed DataFrame and audit costs, so the row-count advantage did not yield a consistently faster desktop result. RSS deltas can be small after earlier allocations and should not be interpreted as standalone device memory requirements.

### Parent text generated from the pilot

A flagged synthetic true positive (C002, 2026-09-12):

> The model identified an unusual pattern in the available recorded activity. Compared with typical recorded days, language-related recorded signals and screen-related recorded activity increased the model's score for unusual activity.

A flagged synthetic false positive (C015, 2026-09-12):

> The model identified an unusual pattern in the available recorded activity. Compared with typical recorded days, language-related recorded signals increased the model's score for unusual activity.

All four flagged pilot days had qualifying evidence and all reported contributions were score-increasing. The six unflagged days produced no parent text. The formatter's decreasing-contribution wording and fallback for insufficient explanation evidence are covered by tests. A false-positive example can describe model behavior accurately even when the simulated anomaly label is negative.

## Limitations and deployment recommendation

Mathematical reconstruction verifies agreement with the frozen continuous score. Mask audits and donor deletions provide limited evidence of model-level fidelity. This synthetic, small, retrospective exercise does not establish real-world child-safety validity. The audit does not establish that the joint distribution of mixed source blocks is realistic. The broad groups trade away within-source detail. Near-cutoff observations from both sides, more children and cases with a wider range of leading groups are needed to test robustness. Future work should validate the evidence gate, including score-decreasing and fallback coverage, and assess whether parents interpret the wording as intended.

Keep the dashboard integration pending review. The group-aware method is a candidate for periodic explanation of already-flagged daily records after evaluation on a broader sample. Profile standalone startup, peak memory, latency, energy, and score parity on representative devices. Kernel SHAP adds SHAP, Numba, llvmlite and other scientific Python dependencies. Group-aware attribution uses the existing NumPy, pandas and scikit-learn scoring stack, and psutil is used for the evaluator. Desktop timing and RSS do not establish suitability for per-event or on-device execution. If target-device budgets rule out periodic exact group attribution, benchmark a smaller background or approximation and validate score parity, stability, and parent-text agreement before use.

## Reproduction

From the repository root, using a dedicated Python environment:

```powershell
python -m pip install -r component4_profiling_xai/requirements-shap.txt
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python -m component4_profiling_xai.evaluate_behavioral_shap component4_profiling_xai/outputs/behavioral_shap_rerun.json
python -m pytest -q component4_profiling_xai/tests tests/test_ui_f2_services.py tests/test_ui_pages.py
git diff --check
```

## Regression verification

`python -m pytest -q --basetemp=.shap-test-tmp component4_profiling_xai/tests tests/test_ui_f2_services.py tests/test_ui_pages.py` reported **169 passed in 213.08 seconds**, with exit code 0. This includes Function 1 tests, the urgent-rule and coverage regressions, the existing Streamlit page tests, and eight new SHAP test cases. The initial system-temp run hit Windows sandbox fixture setup errors. The workspace-temp rerun passed all tests and emitted a separate sandbox permission warning while cleaning up a library's system-temp directory at interpreter shutdown.

`git diff --check` returned exit code 0 with no output. A separate whitespace check of all six untracked deliverables returned no diagnostics.

## References

- Lundberg, S. M., and Lee, S.-I. (2017). A unified approach to interpreting model predictions. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- Aas, K., Jullum, M., and Løland, A. (2021). Explaining individual predictions when features are dependent. [Artificial Intelligence](https://doi.org/10.1016/j.artint.2021.103502)
- Jullum, M., Redelmeier, A., and Aas, K. (2021). groupShapley: Efficient prediction explanation with Shapley values for feature groups. [Preprint](https://arxiv.org/abs/2106.12228)
- Janzing, D., Minorics, L., and Blöbaum, P. (2020). Feature relevance quantification in explainable AI: A causal problem. [AISTATS proceedings](https://proceedings.mlr.press/v108/janzing20a.html)
- SHAP API documentation. [KernelExplainer](https://shap.readthedocs.io/en/stable/generated/shap.KernelExplainer.html) and [PartitionExplainer](https://shap.readthedocs.io/en/latest/generated/shap.PartitionExplainer.html)

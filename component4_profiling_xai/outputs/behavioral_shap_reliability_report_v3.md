# Behavioral SHAP reliability gate v3

## Why v2 named groups on every flagged day

The [v1 pilot](behavioral_shap_report_v1.md) used agreement on the top two absolute group contributions as a screen for wording. The [v2 evaluation](behavioral_shap_expanded_report_v2.md) used four backgrounds and an independent deletion donor set with the same formatter. The old gate required inputs to match, score reconstruction and threshold status to agree, and zero deterministic mask violations. For each candidate in the primary run's top two, the gate then required it to remain in the top two, keep the same sign, and reach 0.01 absolute score units in every attribution and in its deletion effect. One such group was sufficient. All 64 flagged cases met that condition. Forty-seven qualified for two groups and 17 for one.

A swap of the first and second groups kept both candidates eligible. The gate also accepted a stable candidate when a different group had the largest deletion effect. Near-threshold status and empirical donor support were not inputs to the wording decision. In v2, 21.1% of mixed masks had exact screen-count, language-count and calendar combinations that occurred fewer than five times in training, even though every scored mask satisfied the recorded feature relationships. The old formatter was preserved to reproduce historical results. The new parent-facing offline entry point is `ReliabilityGate.review`.

## Decision policy

The gate reuses the frozen saved preprocessing, feature order and score `−pipeline.score_samples(X)`. It calls the existing observation gate. Missing required observations return `insufficient_data`, and a score at or below the saved strict cutoff returns `not_flagged`. Both suppress parent wording and skip SHAP. An assessable flagged case either receives `reliable` with one named group or `uncertain` with the neutral fallback. Urgent rules and their reasons are outside this entry point.

The policy below was specified before replaying the v2 panel. It responds to issues observed in v2, so the replay measures the consequences of that policy on known cases. It supplies no independent estimate of the gate's performance.

### Stability and near-threshold checks

- Use the four fixed normal-training references from v2 (32 and 64 observations, each with seeds 17 and 29). Check reconstruction and the original strict flag using the existing numerical tolerance of 1e-8. Compare the complete rank orders and record sign agreement for all groups.
- The named group must lead under all full and support-restricted reference samples, with a positive contribution. Use a separation condition over the observed attribution ranges. For leader L, require `min_r |phi_L(r)| − max_(r,g≠L) |phi_g(r)| > 1e-8`. This rejects cases where ranks are constant within each run but the envelopes still overlap across backgrounds. Zero and tie handling use the existing numerical tolerance. The intervals are empirical ranges over fixed references, with no confidence-level interpretation.
- Require both the original and support-restricted disjoint donor deletion to assign the greatest absolute score effect to the same group and in the positive direction. The deletion effect is measured from the complete target, while a Shapley value averages over coalitions. Requiring agreement here is an abstention rule for wording, even when disagreement may reflect legitimate interactions.
- If the frozen score exceeds the threshold by no more than the range of the original backgrounds' expected scores (plus numerical tolerance), report the day as flagged and withhold a specific factor. The background range is an observed scale for instability in the score decomposition. The frozen score and threshold do not vary with the background. This rule does not estimate prediction uncertainty or alter the cutoff.

### Empirical support

The support index uses eligible normal days from the original training split. An exact key needs matched records from at least two distinct training children. Two witnesses are the minimum for cross-child replication. Multiple days from one child alone are insufficient. This rule supplies no calibrated risk level.

| Key | Fields |
|---|---|
| Joint count and calendar | Screen incident count, language flag count, weekend indicator |
| Screen activity and calendar | Screen incident count, screen nighttime incident count, weekend indicator |
| Language activity and calendar | Language flag count, language nighttime flag count, weekend indicator |

The target and retained donor endpoints must pass all three keys. For each reference, the same paired subset of donors is retained across the six mixed coalitions. A donor is removed if any of its mixed rows fails any key. The disjoint deletion reference uses the same logic across its three deletion masks. Each donor set must retain at least two distinct children. Recompute the attribution/deletion on the retained sets and compare them against the full-reference results. This checks whether the result survives removing poorly supported substitutions. It changes the empirical reference distribution, so it is a sensitivity test.

Features such as confidence, severity, apps, and finer time dependence are outside the support keys. Structural checks keep source blocks intact, enforce the two derived sums and check known count and score bounds. Those checks and exact-key support address different questions. Neither establishes that a counterfactual reflects observed behavior in the real world.

## Replay protocol

Replay exactly the 130 synthetic days from v2, 65 per dataset, using the saved identifiers. Compare the gate's score and flag against the existing prediction function and the v2 evidence. The research evaluator also reconstructs scores for unflagged days in a separate offline path. Each case gets a missing-coverage control by changing a copy of its coverage indicator. The evaluator checks model, metadata, panel and dataset identities before and after the replay.

The independent-seed Step 4a records are a second synthetic sample, created by the same generator and already seen in v2. There is no available independent dataset generated under materially different scenario assumptions. This report makes no out-of-scenario validation claim.

## Results

The [case-level replay](behavioral_shap_reliability_v3.json) contains 130 unchanged primary cases and 130 separately labelled missing-coverage controls. All 64 originally flagged cases previously produced specific text.

| Source | Cases | Specific / reliable | Neutral / uncertain | Suppressed / not flagged |
|---|---:|---:|---:|---:|
| Original test | 65 | 10 | 21 | 34 |
| Independently seeded Step 4a test | 65 | 16 | 17 | 32 |
| Total | 130 | 26 | 38 | 66 |

The 26 specific explanations name 24 language leaders and two screen leaders. Fifteen came from synthetic true positives and 11 from synthetic false positives. The other 38 flagged days retained the model flag and received this exact fallback:

> The system identified an unusual pattern in the available recorded activity, but cannot reliably identify which recorded factors contributed most.

The added controls returned `insufficient_data` in all 130 cases, with no text or model flag. None of the original 130 cases lacked required observations.

### Reasons for withholding

Reasons overlap; counts are flagged cases.

| Reason | Cases |
|---|---:|
| Target failed at least one empirical support key | 36 |
| An explanation reference retained fewer than two distinct donor children | 35 |
| The deletion reference retained fewer than two distinct donor children | 35 |
| Score margin at or below background expected-score span | 3 |
| Leading group changed | 1 |
| Attribution envelopes overlapped | 1 |

Thirty-six of the 38 fallbacks included a support failure. Of the 64 flagged targets, 28 failed the joint event-count and calendar key, 18 failed the screen nighttime key, and 17 failed the language nighttime key. Across the original four reference sets for flagged cases, out of 73,728 mixed rows, 17,391 failed the joint key, 11,220 the screen nighttime key and 9,676 the language nighttime key. A row can fail several keys. The training index held 57 replicated joint keys, 21 screen keys and 28 language keys.

All 28 flagged targets that passed all keys also retained enough donors for the support-restricted attribution and deletion. Restricting donors did not flip the leader in a case whose full-background results had a unanimous leader. This replay therefore shows the value of the coverage check more directly than it shows a benefit from the attribution sensitivity check. Controlled tests cover rank reversal, sign reversal, overlapping envelopes despite a common within-run leader, and deletion disagreement.

### Boundary and fidelity checks

The expected-score span was 0.04546. Previously all 12 flagged boundary days received specific wording. Under v3, 11 received the neutral fallback and one received specific wording. All 12 unflagged boundary days remained suppressed. The three cases inside the expected-score band remained flagged and all received fallback. Independent C007 on 2026-09-13 had a score margin of +0.02553 and swapped its leading group across references. The new reasons included cutoff proximity, lead change and envelope overlap. Independent C040 on 2026-09-15 failed only the cutoff-margin check. The one specifically explained boundary case had a margin of +0.19353 and passed the support, rank and deletion checks. It belonged to the v2 nearest-six stratum and lay outside the background-span band.

All 130 scores and flags agreed with the existing prediction path and prior evaluation within tolerance. All 520 full-reference reconstructions agreed with the saved decision, with maximum error 3.55 × 10⁻¹⁵. All 112 support-restricted reconstructions agreed, with maximum error 7.11 × 10⁻¹⁵. Both sets had zero deterministic mask violations and no execution or reconstruction failures. For unflagged observations, reconstruction was measured in the research replay only. The parent path skipped SHAP.

## Limitations and recommendation

**NO-GO for parent dashboard integration.** The new gate withheld specific wording in the known sparse and unstable cases while preserving the detector output. Its two-child rule, separation envelope, and background-span cutoff band are conservative research policies. Their error and abstention rates on independent scenarios or representative records remain unknown. Replication of exact keys gives evidence only about the chosen discrete summaries, and named groups still appeared on 11 synthetic false positives. The label `reliable` is scoped to the explanation evidence for the model score. It carries no claim about harm or actual changes in a child's behavior.

Freeze the policy before evaluating new scenario assumptions, including rarer profiles and partial observation patterns. Assess abstention and retention by group, case type and child, and test whether parents understand specific and neutral wording as descriptions of the model. Evaluate the end-to-end cost of full and support-restricted references before considering periodic on-device use. Any future integration should call the new gate and preserve separate urgent-rule presentation.

## Reproduction

From the repository root, with the offline SHAP requirements installed, select an unused output path.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python -m component4_profiling_xai.evaluate_behavioral_shap_reliability component4_profiling_xai/outputs/behavioral_shap_reliability_rerun.json
python -m pytest -q --basetemp=.venv/pytest-shap-reliability-final component4_profiling_xai/tests tests/test_ui_f2_services.py tests/test_ui_pages.py
git diff --check
```

The final regression run reported **180 passed in 80.18 seconds**, exit code 0, including 11 reliability test cases, Component 4, and the existing Function 1 and Function 2 dashboard regressions. `git diff --check` returned exit code 0 with no output. The separate whitespace check of all five new files found zero diagnostics.

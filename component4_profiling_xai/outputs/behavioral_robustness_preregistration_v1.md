# Controlled frozen-model robustness replication preregistration v1

This is a prospective synthetic research protocol. It is locked before generating or scoring any replication rows. The earlier independent synthetic study motivated the factors and expected directions. It contributes no rows, labels, donors, or tuning signals to this replication.

## Questions and hypotheses

1. Increasing ordinary screen or language volume at fixed anomaly labels will increase false positives. The prior shifted scenarios had higher activity counts on false-positive days.
2. Reassigning ordinary events to nighttime will increase false positives and empirical donor-support failures. Nighttime counts were sparse in the original training pool.
3. Shifting ordinary events from language to screen at fixed total volume will change detector errors or the dominant attribution source. The direction is unspecified because the model weighs correlated source blocks and derived totals jointly.
4. Changing only the calendar field will have less impact than changing channel volumes. This comparison is descriptive and does not assert that weekend and weekday behavior are exchangeable in real records.
5. Replacing the seen profile's ordinary-activity parameters with an age-matched new profile will change performance. The direction is unspecified. This comparison bundles a profile's rate, language mix and nighttime pattern.
6. Marking one or both monitors unavailable will suppress scoring and explanations on those days. Outcomes measured on the remaining assessable days may differ through selection.

The primary model endpoints are the changes in false-positive rate (FPR) and false-negative rate (FNR) for each of the six complete-coverage interventions, paired against reference. Other comparisons, including the hypotheses concerning explanations and calendar versus volume magnitude, are descriptive.

## Paired generation design

Twenty profiles are sampled without replacement from the original seed-42 training generator. Pair each with an unused profile of the same age generated using a new seed. “Seen” means that the profile ID contributes normal rows to the original model training. The reference uses newly simulated observations for that profile. The model receives no identifier, so this is a comparison of known versus newly drawn behavioral parameters, not a test of identity recognition or memorization. The new profile uses a fresh ID outside all earlier cohorts.

For each matched pair generate 12 latent child-days. A random permutation assigns exactly nine ordinary days and one each of screen, language, and mixed cluster labels, yielding 180 ordinary and 60 injected days per arm. Every arm reuses the label and the same injected events. Inject two events per affected channel, with normalized feature distributions from the independently implemented synthetic event samplers. The screen and language exposures allow a direct comparison at the daily feature boundary. Upstream sensor or event-classifier performance is outside scope.

For ordinary activity, use shared daily Gamma(2, scale 0.5) intensity and shared inverse-Poisson uniforms. Screen expected count is 1.8 times the profile's screen-session rate times intensity; language expected count is its candidate language rate times intensity. This converts legacy profile parameters to simulated normalized event rates. The language mixture uses the profile's four common modeled categories, normalized to sum to one. Pre-draw matched event banks and timing uniforms; use consistent nested selections and common random numbers. Ordinary baseline timing uses each profile's nighttime share (capped at 0.40) and injected timing uses probability 0.10 in all arms. Calendar contrast holds event counts and event properties fixed rather than reapplying the original profile's weekend multiplier.

| Arm | Change from the same reference day |
|---|---|
| Reference | Seen profile, weekday, complete coverage |
| Screen volume | Add two ordinary screen events, holding the language block fixed |
| Language volume | Add two ordinary language events, holding the screen block fixed |
| Nighttime | Set ordinary event nighttime assignment probability to 0.40 using the same timing uniforms; hold injected events and other inputs fixed |
| Channel balance | Replace up to two ordinary language events with screen events at fixed total count. A day with no ordinary language event has no treatment exposure. |
| Calendar | Keep all events intact and move the date to a weekend, changing only the saved weekend feature |
| Unseen profile | Substitute a fresh age-matched profile's ordinary rate, category mixture and nighttime parameters with shared latent draws; preserve the anomaly event bank and calendar |
| Coverage | On three randomly chosen days per pair, mark screen only, language only, and both unavailable, one each. Remove unavailable observations and derived totals. Keep all other days unchanged. |

Calendar dates begin at the new period 2026-11-02 and all are later than the previous independent evaluation. Factor arms reuse the same pair/day keys and include 240 rows each (1,920 total). A fixed-size feasibility design uses all rows without optional stopping or seed search. Complete coverage is intentional for the six primary contrasts. Coverage arm outages are assigned by the layout stream independently of scores; their overlap with labels is reported.

Seeds: legacy seen-profile 42; event/layout root 2026112302; unseen profiles 2026112303; matched pair sampling 2026112304; bootstrap 2026112305. Each pair/day draws independent spawned streams as implemented in the hashed generator. Generation and analysis save to a new results directory. The runner verifies pairing, fixed labels, channel-volume invariants, the exact calendar contrast, monitoring controls, and structurally valid model inputs.

## Metrics and uncertainty

- For each arm, report precision, recall, F1, FPR and FNR among assessable days, plus numerical confusion counts. Report assessment coverage and insufficient-data abstention over all 240 days. For each anomaly category, report recall among assessed and detected fraction among all scheduled positives (including missing). The fixed 25% positive-label share makes precision and F1 conditional on this design.
- Estimate uncertainty by resampling the 20 matched profile pairs with replacement, keeping all paired repeated days and interventions together. Use 10,000 multinomial-weighted cluster bootstrap replicates, percentile intervals and the shared bootstrap seeds. Report 95% intervals for arm summaries and paired deltas. The 12 primary paired differences also receive two-sided Bonferroni-adjusted intervals with nominal 95% family coverage. Undefined metrics retain null values; report the number of interpretable replicates for arm estimates. This small-cluster percentile bootstrap gives approximate intervals and can be degenerate for all-zero or all-success outcomes.
- Record specific and neutral rates among flagged days, donor-support failure reasons and exact target-key failures. Record full-reference leading-group and direction stability, restricted leaders where available, and reconstruction errors. Report score/threshold parity, plus coverage suppression. Cache identical numerical feature and coverage inputs only to avoid repeated evaluation of the unchanged gate. Count all scheduled observations in summaries. Counts across arms are paired and cannot be treated as independent evidence.
- Plot paired FPR and FNR deltas with the adjusted intervals, alongside a separate plot of specific/neutral counts by arm. No model, threshold, SHAP attribution, or reliability setting is fit or changed.

## Acceptance criteria for this experiment

- Reference FPR and FNR point estimates must each be at or below their historical seed-42 test anchors (19/627 and 21/53). This is a descriptive comparability screen using unpaired historical results.
- For robustness under each complete-coverage change, the upper endpoint of the adjusted paired interval must be at most **+5 percentage points** for both FPR and FNR. The margin allows up to one additional error per twenty applicable days for this research noninferiority screen. It is not a clinical or deployment tolerance. Failure includes intervals too wide to rule out harm at that margin.
- For fidelity, require no score/flag discrepancy from the saved prediction path, no reconstruction or structural failures from the frozen explanations, and exact suppression for the 60 designed outage days while assessable cases retain normal processing. Rates of support and specific wording are descriptive because no validated deployment minimum exists. A pass on fidelity does not establish detector robustness.
- No result in this simulation is sufficient for dashboard readiness. The experiment measures frozen-model behavior under engineered normalized-event conditions. UI decisions need independent representative observations, intact reliability evidence, wording comprehension and target-device measurements.

## Reproduce

Use the unchanged environment and model files from the [previous freeze manifest](behavioral_shap_policy_freeze_manifest_v1.json), with its SHAP dependencies available from an ignored environment and Matplotlib 3.10.7 for static plots. The preregistration lock pins this protocol, generator, runner and referenced input identities before generation.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python -m component4_profiling_xai.evaluate_behavioral_shap_robustness component4_profiling_xai/outputs/shap_robustness_v1
```

# Reliability-aware XAI for Function 02

## Objective and audit

The objective is to qualify what the existing Parent Dashboard says about an explanation while preserving model decisions, the temporal detectors and independent safety alerts.

| Existing path | Finding |
| --- | --- |
| Deployed daily SHAP | `behavioral_explain.py` enumerates exact group coalitions for the saved Global One-Class SVM and its 24 training donor rows. The parent view previously named the strongest contribution with a direction consistent with the assessment. |
| Deployed daily LIME | Ridge regression on binary group presence splits donor rows between training and local evaluation. The earlier renderer used coefficient sign agreement to mention a second explanation. It did not consider held-out local fit for that wording. |
| Step 01 | It already checked score reconstruction and observation coverage, required signed evidence for isolated and sustained patterns, and abstained on ties or unclear patterns. It used two adjacent seven-day windows and kept detector overlap separate. Its report, experimental outputs and decision rules are retained. |
| Archived reliability research | `behavioral_shap_reliability.py` already implements reference-envelope, empirical-support and group-deletion checks for a separate SHAP research path with different cohorts and donor selection. It remains in place. The deployed implementation reuses the common ranking convention and exercises the production explainer with its own frozen donor set. |

## Method and decision policy

For each model input, the score is the negative of the deployed pipeline's `score_samples` output. A fixed expected group set excludes cross-source total columns that are recomputed during grouped coalition masking. The reference is calculated independently by scoring the frozen background after recomputing those totals. A daily explanation must have finite signed values and satisfy both `full coalition score = score` and `reference + sum(group SHAP) = score` within the Step 01 absolute tolerance of `1e-8`. Replaying the historical records uses the same check. The Step 01 reconstruction check for the difference between window averages still applies.

Held-out LIME diagnostics are computed over donors excluded from the local Ridge fit, using the existing local kernel weights. With predicted score `ŷᵢ`, actual coalition score `yᵢ`, residual `eᵢ = ŷᵢ − yᵢ`, weight `wᵢ` and weighted target mean `ȳʷ`, the metrics are

```text
RMSE_w = sqrt(sum(wᵢ eᵢ²) / sum(wᵢ))
MAE_w  = sum(wᵢ |eᵢ|) / sum(wᵢ)
R²_w   = 1 − sum(wᵢ eᵢ²) / sum(wᵢ (yᵢ − ȳʷ)²)
```

The surrogate must have finite coefficients and errors, a nondegenerate target variance, and adequate weighted R² to corroborate a result. SHAP measures change relative to its background score. LIME coefficients describe changes when a group switches from absent to present in the surrogate and have their own intercept. The comparison uses score direction and the leading compatible group, without comparing intercepts or equating magnitudes. A stable, verified SHAP explanation can stand when LIME has low fidelity or disagrees.

Stability uses two small affine probability recalibrations around the recorded values. For a positive source count with valid confidence aggregates, the upward probe applies `p ↦ 0.98p + 0.02` and the downward probe applies `p ↦ 0.98p`. Both transform the mean and maximum together and preserve bounds, mean ≤ maximum, and equality for a single observation. Counts, dates, totals and missing values stay fixed. The engine checks the leading compatible group and its direction. It also checks whether the probes cross the original decision threshold. A crossing represents local model sensitivity. A leading-group change without a decision crossing represents sensitivity in the wording of the explanation. Neither changes the original assessment.

Temporal reliability replays the same probes on scored inputs in the two Step 01 windows for the selected child and date. Step 01 still decides coverage, positive evidence, stability, isolated versus sustained change, and score reconstruction. The gate checks that the pattern and group survive the probes. The original detector annotation uses actual historical signals. Values outside the selected horizon, including other children's records, are excluded.

| Outcome | Parent effect |
| --- | --- |
| Supported | Show the verified daily factor or the Step 01 pattern. Mention daily corroboration only when the LIME fidelity and agreement checks pass. |
| Limited evidence | Use neutral wording when the leading group, direction, model decision, or temporal pattern is sensitive or unclear. |
| Conflicting evidence | Use neutral wording when SHAP evidence is already ambiguous and usable LIME favors a different candidate. Stable SHAP remains eligible on its own. |
| Insufficient data | Use the data-availability message for the selected day or the temporal coverage fallback. |
| Invalid explanation | Use the unavailable message after failed reconstruction or absent explanation output. |

The daily score classification, overview, guidance and urgent alerts retain their existing paths. Only **Why Did the AI Say This?** and **Recent Behavioral Pattern** use the reliability decision for their wording. The dashboard stays qualitative and uses the existing sections.

## Experimental protocol

The development run uses the original synthetic validation split, and the final run uses the test split from the separate `step4a_heldout_seed20261020` synthetic dataset. Both stages generate controlled examples from earlier normal anchors using the unchanged Step 01 scenario generator. They include stable activity, gradually changing and sustained counts for each source, single spikes, missing dates and insufficient coverage. Each stage also uses the last available endpoint per child and the scored day closest to the frozen threshold per child. Only records through each case date are made available. Near-boundary cases are used for daily review.

For the coverage comparison, a larger, separate ±0.05 probability probe supplies a robustness proxy. A named explanation fails the proxy when the larger probes fail the original daily group/direction or classification checks, or the temporal pattern/group checks. The proxy shares the same perturbation family, so it measures repeatability under that stress. It is not independent ground truth about a child. The daily and temporal old-system comparators apply the original naming rules with the reliability gate bypassed.

Development compares 0.5 versus 1.0 required stability votes and weighted LIME R² floors of 0, 0.5, 0.7 and 0.9. The selection first minimizes named cases that fail the robustness proxy, then maximizes retained stable cases. LIME selection prioritizes agreement with a second donor split, whose target is positive held-out R² and replication of the leading compatible LIME group. Remaining ties favor retained corroboration, then the stricter stability requirement and R² floor. The validation command writes the policy and deployed model hash. The holdout command requires the application policy to match that lock.

A separate diagnostic stress table starts from an actual scored, SHAP-verified flagged example and injects a low-fit surrogate, conflicting coefficients, or a rank-changing SHAP replay whose sum still reconstructs the score. These cases exercise decision branches and are kept separate from cohort statistics. They represent simulated explanation faults, not estimates of their real incidence.

## Threshold selection and sensitivity

Both vote rules retained the same 90 named daily and temporal explanations that passed the larger probe, with zero retained failures on development. The tie rule required both probes to agree. Across the 58 evaluable method pairs, a LIME floor of 0.50 reached 50/58 agreement with the second donor split, compared with 46/58 at 0, 41/58 at 0.70, and 37/58 at 0.90. At 0.50, the first split accepted 39 of the replicated leaders and one unreplicated leader. At 0.70 it accepted 29 replicated leaders and none of the unreplicated leaders. The locked policy uses two agreeing probes and a LIME R² floor of 0.50. The LIME floor affects corroboration only. The stability comparison did not establish that one vote rule was empirically better on this development set.

## Actual holdout results

| Outcome | Existing | Reliability-aware |
| --- | ---: | ---: |
| Daily named explanations | 71/112 | 71/112 |
| Daily responses without a named group | 41/112 | 41/112 |
| Temporal named explanations | 32/72 | 32/72 |
| Temporal responses without a named group | 40/72 | 40/72 |
| Named daily explanations failing the larger probe | 0/71 | 0/71 |
| Named temporal explanations failing the larger probe | 0/32 | 0/32 |
| Daily LIME corroboration under the respective wording rules | 71 | 50 |
| Daily median explanation-only time | 14.3 ms | 58.9 ms |
| Temporal median explanation-only time | 247.0 ms | 574.1 ms |

- **Fidelity.** Maximum absolute daily reconstruction error was `1.78e-15` across 112 daily explanations. Maximum error for reconstructed differences between window means was `4.30e-15` across 68 eligible temporal cases. The held-out local LIME median R² was 0.486, median weighted RMSE 0.149 and median weighted MAE 0.089 in score units. SHAP and LIME had the same direction and leading compatible group in all 71 named daily cases. The fit requirement removed the corroboration claim in 21 of those cases.
- **Ranking, direction and model sensitivity.** Full daily ranking was unchanged across deployment probes in all 112 cases. All 71 explainable daily leaders kept their direction. In 73 cases a deployment probe moved the model score by more than the numerical tolerance, including 69 named cases. None crossed the threshold. The 40 closest-per-child cases had varying margins and produced no deployed decision flip. The focused test exercises an injected crossing separately.
- **Controlled cases and fallbacks.** Both approaches matched the injected group in all 24 changed cases for both daily and temporal explanations. Neither named a wrong injected group. The temporal gate retained the 16 gradual or sustained results, eight controlled spikes, four stable results and four insufficient-history results. The complete temporal breakdown was 42 supported, 26 limited and four insufficient. Ten supported cases described stable patterns without naming a group.
- **The 26 Step 01 unclear results.** The unmodified cohort reproduced all 26. In 17 cases there was an out-of-band departure without a positive group crossing the elevation rule. In nine, elevation occurred on multiple observed dates, but no group met the complete sustained rule ending on the selected date. None of the 26 was attributed to missing coverage, numerical reconstruction failure, or the explicit tie branches. The gate preserved these abstentions.
- **Simulated diagnostic stress.** A low-fit LIME result and a high-fit conflicting LIME result each removed corroboration while retaining verified stable SHAP. A staged SHAP rank swap yielded limited evidence, and combining it with opposing usable LIME yielded conflicting evidence. Both withheld a factor. These outcomes verify the policy branches under the staged conditions.

**Before and after examples.** For synthetic child C010 on September 28, 2026, the original wording named language monitoring records as making the day look less unusual and added “A second explanation also pointed in that direction.” Step 02 keeps the signed SHAP statement and removes the corroboration sentence because the local LIME fit was below the locked floor. In the staged unstable-rank condition on a flagged example, the old wording names an activity group and Step 02 instead says the available information does not clearly explain which activity contributed most.

### Research contribution and limitations

The measured gain is better support for corroboration wording. The filter removed 21 corroboration claims that had low held-out surrogate fit while retaining the SHAP factor. The evaluation did not show a reduction in unstable named factors or an increase in attribution coverage. The separate stress tests show that the policy can abstain when instability or conflict is present. There is additional local computation on each selection.

The examples are synthetic and controlled cases repeat a small number of anchor rows. The two donor splits reuse the same small background pool. Both sets of model probes use affine changes to source confidence, leaving counts and other assumptions untested. The larger probe is a model-level robustness check within the same perturbation family. Group SHAP accuracy says how the deployed model scored a record, and the local LIME tests measure how well a surrogate approximates those outputs. The results establish neither behavioral causes nor safety or medical judgments. They include no mobile or participant validation. The holdout seed was previously used for Step 01, although Step 02 selected its thresholds from separate development records before its holdout run.

The research artifacts include [holdout metrics](outputs/xai_reliability_step02/metrics.json), [holdout cases](outputs/xai_reliability_step02/holdout.csv), [sensitivity results](outputs/xai_reliability_step02/threshold_sensitivity.csv), [diagnostic stress cases](outputs/xai_reliability_step02/diagnostic_stress.csv) and the [coverage plot](outputs/xai_reliability_step02/coverage.png).

## Regression results

| Test run | Baseline | Step 02 |
| --- | ---: | ---: |
| App and shared-service regressions | 203 passed | 214 passed |
| Profiling component regressions | 125 passed, 2 skipped | 125 passed, 2 skipped |
| Focused Function 02 run | 18 Step 01 cases | 29 passed with the locked policy |
| Repository-wide pytest collection | 10 pre-existing errors, 2 skipped | Same 10 errors, 2 skipped |

The repository-wide blockers are missing Torch and ONNX, cross-component `src` import collisions and duplicate integration test module names. The focused run covers actual score reconstruction, invalid or missing evidence, LIME agreement, unreliable LIME, unstable SHAP, a cutoff crossing, temporal checks, child and date switching, as-of replay, unchanged decisions, and the urgent safety path with outbound sockets and DNS blocked.

## Reproduction

```cmd
cd /d E:\RP-J26-DS-316
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
python -m component4_profiling_xai.evaluate_explanation_reliability validate
python -m component4_profiling_xai.evaluate_explanation_reliability holdout
python -m pytest -q tests/test_function02_explanation_reliability.py tests/test_function02_upgrade.py tests/test_function02_temporal_explain.py --basetemp .venv/rel-focus
python -m pytest -q tests --basetemp .venv/rel-after
python -m pytest -q component4_profiling_xai/tests --basetemp .venv/rel-component-after
python -m streamlit run ui/app.py --server.address 127.0.0.1
```

Install the existing dependencies with `python -m pip install -r requirements.txt` if needed. Open [the local dashboard](http://127.0.0.1:8501), select **Behavioral Anomaly**, and choose a child and date.

## File inventory

| File | Change |
| --- | --- |
| `component4_profiling_xai/src/behavioral_explanation_reliability.py` | Daily and temporal review, bounded local probes, method-specific decisions. |
| `component4_profiling_xai/src/behavioral_explain.py` | Shared SHAP verification, reference score computation and held-out LIME error diagnostics. |
| `component4_profiling_xai/src/behavioral_temporal_explain.py` | Shared verifier and audit reasons for Step 01 unclear results. |
| `ui/pages/_behavioral_upgrade.py` | Qualitative reliability wording in both explanation sections. |
| `tests/test_function02_explanation_reliability.py` | Focused policy, scoring, as-of replay and feature-constraint tests. |
| `tests/test_function02_upgrade.py` | Updated offline parent regression for reliability-aware temporal captions. |
| `component4_profiling_xai/evaluate_explanation_reliability.py` | Chronological development and locked holdout evaluation, diagnostics and plot. |
| `component4_profiling_xai/XAI_RELIABILITY_STEP02.md` | This report. |
| `component4_profiling_xai/outputs/xai_reliability_step02/` | New `validation.csv`, `threshold_sensitivity.csv`, `policy_lock.json`, `holdout.csv`, `diagnostic_stress.csv`, `metrics.json` and `coverage.png`. |

## Viva explanations

- **What does the gate measure?** It checks whether the explanation faithfully matches the model score and whether its wording survives small, plausible input changes.
- **Why check LIME and SHAP separately?** SHAP reconstructs the score relative to its reference. LIME approximates the score locally. Its held-out errors tell us whether it adds credible corroboration.
- **What happens near the model cutoff?** A perturbation can expose a sensitive explanation. The dashboard keeps the original decision and can use neutral wording about contributing factors.
- **What does temporal reliability mean?** The validated Step 01 pattern and leading group survive the probes within the coverage available through the selected day.
- **Can reliability lower an urgent alert?** The urgent rules run independently of attribution review.

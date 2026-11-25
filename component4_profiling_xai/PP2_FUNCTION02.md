# Function 02 upgrade for PP2

The parent dashboard uses the validation-selected novelty detector, shows a causal child profile and score trends, explains the selected scorer locally, and checks urgent rules independently. The original Function 02 view and saved artifacts remain available through `C4_F2_LEGACY_VIEW=1`. Function 01 keeps its original entry point and artifacts.

## Windows Command Prompt

The final Parent Dashboard uses the shipped models. Install the dependencies once with `python -m pip install -r requirements.txt`. This needs local wheels or package-index access if dependencies are missing.

```cmd
cd /d E:\RP-J26-DS-316
python -m streamlit run ui/app.py --server.address 127.0.0.1
```

In the parent view, choose a child and activity date together. The day's AI result, differences
from earlier records, recent patterns, supportive guidance, and safety alerts are shown in words.
The view withholds a current-change message when a temporal statistic is carrying an older alert
without a recent start or repeated flagged days in that same run. The underlying detector and
temporal calculations are preserved. The technical research files contain the numeric results.

Both functions run locally after setup. The historical training command `python -m component4_profiling_xai.train_function02` reproduces the earlier experiment and writes to its model output directory. It is unnecessary for launching the parent app. Optional training baseline flags are `--baseline-days`, `--frequency-days`, and `--min-history`.

The evaluation tables and charts remain in `outputs/function02_upgrade/` and `outputs/function02_optimization/`. The experiment viewer source is archived in `experiments/archived_research_function02.py`. Use the files and [optimization report](PP2_FUNCTION02_OPTIMIZATION.md) for PP2.

Run regressions from the repository root with `python -m pytest tests component4_profiling_xai/tests -q --basetemp .venv/f2-pp2-tests`.

## Files

```text
.streamlit/config.toml                                 modified, localhost and no usage analytics
requirements.txt                                       modified, measurement and chart dependencies
component4_profiling_xai/
  requirements.txt                                     modified
  README.md                                           modified, links to current workflow
  PP2_FUNCTION02.md                                    new
  experiments/archived_research_function02.py          archived PP2 experiment viewer
  train_function02.py                                  new, fit, calibrate and evaluate
  src/
    behavioral_profile.py                              new, calendar profiles, score replay and prioritization
    behavioral_explain.py                              new, group-wise SHAP and regression LIME
  outputs/function02_upgrade/
    selected.joblib                                    selected pipeline, feature names and local reference rows
    candidate_models.joblib                            comparison pipelines and validation metadata
    evaluation.json                                    split dates, thresholds, temporal, XAI and resource results
    model_comparison.csv, model_comparison.png           detector and ablation results
    training_correlations.csv                          training-only overlap audit
    xai_evaluation.csv, global_shap.png                  XAI measurements and global group attribution
ui/
  theme.py                                             modified, remove remote font fetch
  pages/behavioral_anomaly.py                           modified, route to upgrade or legacy view
  pages/_behavioral_upgrade.py                          new, parent dashboard sections
tests/
  test_function02_upgrade.py                           new
  test_ui_pages.py                                     modified, explicit legacy regression mode
```

The clean pre-upgrade Git snapshot is stored at `.venv/function02-pre-upgrade-b5eb7e3.zip`. The managed workspace disallowed creating a new Git ref. The original datasets, notebooks and model directories retain their existing files.

## Viva algorithm notes

| Algorithm | What it computes |
|---|---|
| Personalized profile | Within each child, a left-closed calendar window includes dates from day −14 through day −1. For each recorded feature it computes median, mean, population standard deviation, count and observed-minus-median. Six selected count features also receive median absolute deviation and a robust Z-score `(observed − median) / max(1.4826 × MAD, 1)`, clipped to ±12. The median and MAD protect the comparison against a single extreme record. Reported means and standard deviations are descriptive and remain sensitive to extremes. |
| Cold start and missingness | Personal deviations are unavailable below five observations. Median imputation for these model columns is learned from normal training days. Seven-day frequency counts prior dates with a positive recorded count, retaining missing values when no dates were observed. Missing screen or language coverage blocks combined scoring, while actual monitored zeroes remain valid. |
| Isolation Forest | Random trees partition normal training records. Unusual records tend to require fewer splits to isolate. The anomaly score negates scikit-learn’s normality-oriented score. |
| One-Class SVM | An RBF decision function learns the support of normal training records. The negative of the fitted pipeline’s `score_samples` is used, so higher means more unusual. Both model families fit imputation and scaling on normal training days. Thresholds come from normal training-score quantiles chosen by validation F1. |
| EWMA and CUSUM | EWMA applies weight 0.3 to the current standardized score and 0.7 to the previous estimate. One-sided CUSUM adds the standardized score minus drift 0.5 and floors at zero. Two consecutive threshold exceedances activate a change. An alert fires only on entry to the active state. A missing day breaks the run. Each threshold maximizes validation episode detection minus five times the false-alert rate. |
| Alert priority | Urgent rules are applied first. A missing required score produces insufficient data. An active change or a flagged day with enough prior flagged observations raises priority. The repeat threshold was selected by validation F1. Prior frequency and consecutive-flag statistics use earlier predictions for prioritization only. A same-day output is never fed back to the detector. |
| Group SHAP | Enumerates all coalitions of screen, language and calendar context. Omitted groups use 24 randomly sampled normal training donors. Overlapping totals are recomputed from their source features. Shapley weights average the marginal change across possible group orders. Signed values sum to the actual score minus the training-reference mean. |
| Regression LIME | Uses the same detector scores under empirical source-group substitutions. Binary group presence is the interpretable representation. Exponential distance weights emphasize masks near the original row. A weighted Ridge surrogate reports its signed coefficients and weighted R² on a held-out donor partition. |
| Recommendations | Guidance combines actual positive group attribution with recorded incident types or sustained change. The text invites supportive conversations and avoids causal or clinical conclusions. |

## Component input contract

The upgraded page defaults to the project’s bundled synthetic daily CSV. To load a local daily aggregate, set `C4_F2_DAILY_CSV` before starting Streamlit.

```cmd
set "C4_F2_DAILY_CSV=E:\RP-J26-DS-316\private\behavior_daily.csv"
python -m streamlit run ui/app.py --server.address 127.0.0.1
```

Use `data/behavioral_daily.csv` as the schema reference. The required columns are a stable local `child_id`, an Asia/Colombo calendar `date` formatted YYYY-MM-DD, and the 27 named daily fields from `DAILY_MODEL_FEATURES`. Supply the screen and language monitoring indicators and urgent-rule counts whenever known. Leave unobserved fields blank. One monitored day with zero recorded events uses numeric zero. Every child/day key must be unique.

For raw Component 1 and Component 2 data, use existing `behavioral_dataset.c1_events`, `c2_events`, then `build_daily_features` with verified per-source coverage. The adapters validate timestamps, resolve child IDs, deduplicate within each source, and retain detection confidence and risk aggregates. Apply `to_descriptive_names` and `add_calendar_features` before exporting. Cross-source incident IDs lack a verified shared key. Keep source-specific aggregates and treat the legacy combined totals as counts of detection records. The proposed feature set omits those redundant totals and the report lists high correlations in training.

For future Component 3 ingestion, resolve the contract’s `session_id` to a verified local child key via the session registry. Normalize `timestamp` to the same local calendar, deduplicate by session ID, and join aggregated emotional-state, self-regulation and escalation observations by child and day. Keep absent session data unknown. No dialogue text belongs in the daily detector input. The existing separately generated emotional and behavioral exports reuse the C001–C040 labels. Shared keys alone do not establish proven linkage and are excluded from model fitting and emotional-only ablation.

## Experimental results and limits

- The existing 40-child synthetic benchmark was split chronologically with 2,240 scored training days, 680 validation days and 680 test days. The historical data and benchmark were previously used by this project. These results measure in-project reproducibility and are unsuitable as independent external validation.
- The primary comparison prespecified the global and full personalized designs. The ablations were measured separately. Validation selected the global One-Class SVM (F1 0.621 versus 0.607 for the full personalized SVM). On test it achieved precision 0.660, recall 0.585, F1 0.620 and FPR 2.55%. The full personalized SVM achieved F1 0.577 and FPR 3.35%. The frozen original achieved F1 0.615 and FPR 3.03%. F1 missed the 0.80 target. The combined-raw and no-temporal ablations remain informative follow-up candidates. Source-only variants use the same fully observed test days as the combined designs for a matched comparison.
- EWMA detected 7 of 22 test episodes (31.8%) and CUSUM detected 5 (22.7%). Mean delay among detections was 1.43 and 1.40 calendar days respectively. Both false change-point rates were 0.32% of scored normal days. An already-active alert does not count as a new detection when a later episode begins. Rising-score and sudden-change wording describe observed score behavior and are not separately labeled scenario classifications.
- On 30 sampled flagged test days, LIME mean weighted held-out R² was 0.947, above the 0.90 surrogate-fidelity target for this local neighborhood. SHAP mean absolute reconstruction error was about 3.3 × 10⁻¹⁶ score units. Reconstruction is an additivity check. Top absolute group agreement was 100% in one alternate-background run. Substituting the top group reduced the score by a mean of 4.37 score units. These are grouped, model-specific measurements. Individual feature values and deviations, rather than individual SHAP/LIME coefficients, are shown in the parent view.
- See `evaluation.json` and `model_comparison.csv` for actual Windows inference and explanation timings, CPU time, process RSS and artifact size. Inference includes preprocessing of a warm one-row request. RSS is measured before and after the XAI batch and does not represent peak allocation. The network-blocked AppTest covers the real Streamlit entry point, Function 02 navigation and local Python execution. A separate localhost HTTP health check returned 200. Mobile hardware, browser-level network isolation, sourced real-child accuracy, emotional ablation and participant SUS were unevaluated.

## Verification

The combined root UI and Component 4 regression run passed 308 tests and skipped two legacy tests that require the optional external SHAP package. The upgraded exact group SHAP and regression LIME paths use the installed NumPy and scikit-learn stack. Five focused integration and unit tests passed after the final dashboard changes, including the network-blocked app entry point, child/date switching, missing-history and zero-MAD behavior, SHAP additivity, LIME output, an urgent rule on an unflagged day, and the separate research interface. The original data, notebooks, model directories and Function 01 source files passed the Git diff check.

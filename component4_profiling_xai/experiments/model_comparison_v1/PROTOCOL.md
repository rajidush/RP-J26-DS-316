# Function 2 experimental model comparison v1

This preregistration implements Step 1 of the approved plan. The experiment asks whether either challenger improves detection of isolated screen anomalies while limiting false alerts and preserving recall for the other anomaly categories. The cohort will be synthetic. A successful detector may advance to a separate offline XAI review. Dashboard integration remains blocked.

## Frozen comparator and shared inputs

| Arm | Pipeline and search |
|---|---|
| A. Frozen One-Class SVM | Load the saved median-imputer, standard-scaler and RBF SVM pipeline read-only. Keep nu 0.01, gamma 0.001, and the exact stored cutoff −12.515804087687126. |
| B. Experimental Isolation Forest | Use the same imputer and scaler. Fix 300 estimators, contamination auto, random_state 1729 and n_jobs 1. Cross max_samples `[128, 256, 512]` with max_features `[0.75, 1.0]`. |
| C. Experimental tuned One-Class SVM | Use the same imputer and scaler. Cross RBF nu `[0.005, 0.01, 0.025, 0.05]` with gamma `[0.0003, 0.001, 0.003, 0.01, 0.03]`. Include the original parameter pair as a reproduction check. |

Use the saved, ordered 27-feature `A_daily` set, verbatim in [config.json](config.json). Fit each challenger and its preprocessing on exactly the same **1,894** original records and **40** training profiles. Reuse the existing normal training selector: original split train, baseline ready, label 0, and assessable under the existing coverage rule. Use a canonical child/date sort to check row identity and pass the rows in the original CSV order for fitting. No extra features, model-specific row exclusion, feature selection, new imputation rule or refit after validation selection. Use scikit-learn 1.6.1. Record all runtime package versions before generation.

| Frozen input | SHA-256 |
|---|---|
| `models/behavioral_anomaly_pipeline.pkl` | `25634fbafabd133495bc66b9ced5921f27ddaac222902d72f38a5fe1515774b7` |
| `models/behavioral_anomaly_metadata.json` | `3f5f0fa72eb5ac1cb19644c36d9a96e063a4aa2855515028c97293b406f375a4` |
| `data/behavioral_daily.csv` | `2b353eab1b715900458ddfb973e665c8206370b564f15f3b2fdbd307dbd32563` |
| Sorted eligible training keys | `dde2ba0ac9ea157078901bdf0a1a2125c597cf2f6a80e6dd3f1620d54352d7f8` |

Input paths are relative to Component 4. The key digest uses UTF-8 CSV with header `child_id,date`, no index, rows sorted by child ID and ISO date, and LF line endings. Verify all four hashes and the model score and coverage contract before fitting and final evaluation.

## Cohort specification

| Cohort | Generation seed | New child profiles | Days | Assessable days | Assessable positives per category |
|---|---:|---:|---:|---:|---:|
| Validation | 2026112401 | 60, 20 per scenario | 1,680 | 1,500 | 100 |
| Final test | 2026112402 | 90, 30 per scenario | 2,520 | 2,250 | 150 |

Use 28 consecutive local dates per child, starting 2027-01-04 for validation and 2027-03-01 for final test, in Asia/Colombo. Use the split-specific namespaces in the config. Generate new latent profiles and source event IDs. A seeded NumPy Generator with separate SeedSequence child streams for profile, ordinary activity, scenario labels and coverage must make the data reproducible. Record the generator code and dependency fingerprints, seed-to-stream mapping, and full dataset hashes in cohort manifests. Define and test the seal before any candidate fitting.

Each scenario uses the normalized-event boundary and the existing daily feature builder. Use the existing independent simulator's screen and language event samplers. Ordinary screen and language counts are Poisson conditional on the scenario rate, a shared daily Gamma(2, 0.5-scale) shock, source-specific lognormal profile multipliers, the profile tier, and the source-specific weekend multiplier. Cycle quiet, steady and variable tiers by child index within scenario. Draw the four-category ordinary language mixture from the specified Dirichlet distribution. Assign ordinary events to night using the scenario probabilities. Draw injected events through the sampler's injected branch with night probability 0.10. The feature builder derives totals and calendar fields. Positive labels come from scheduled injections, independent of scores or coverage. These records represent selected normalized events and cover neither upstream classifier accuracy nor complete device activity.

| Scenario | Mean screen / language rate before modifiers | Weekend multiplier screen / language | Ordinary night probability |
|---|---:|---:|---:|
| Reference-like | 0.45 / 1.10 | 1.15 / 1.40 | 0.07 |
| Screen-heavy weekend | 1.40 / 0.65 | 2.20 / 2.20 | 0.22 |
| Language/night shift | 0.45 / 2.30 | 0.70 / 0.70 | 0.35 |

Shuffle a 28-day layout per child: 22 ordinary days, plus one low- and one high-intensity day each of screen-only, language-only, and mixed anomalies. Add one or four events per affected channel. Ordinary high-volume days retain their negative label. Shift strata jointly vary several factors, so they support a robustness comparison and do not isolate individual effects.

Schedule three outage days per child: two ordinary and one positive. Rotate the six positive outage category/intensity combinations across a seeded permutation of children within each scenario, starting at offsets 0, 2 and 4 for the three scenarios in listed order. This balances category and intensity within one child per cell in each scenario and exactly across the cohort. Randomly assign screen-only, language-only and both-source outages, one of each per child. Remove the affected source fields and cross-source derived totals and set the coverage flags. Preserve scheduled labels. Verify that the existing gate suppresses every planned outage and that the remaining synthetic records satisfy required coverage and feature relationships. An observed zero event count represents a covered day.

Before generating data, perform a development-only precision planning check for the proposed profile counts and error margins. Any sample-size or scenario revision requires a versioned amendment before cohort generation. Do not adapt the sample, remove cases, reroll seeds, or expand the cohort after inspecting results. Wide intervals will leave a criterion inconclusive.

## Leakage controls and selection lock

1. The original seed-42 validation and test partitions, the Step 4A seed-20261020 holdout, and all SHAP pilot, expanded, reliability, independent and robustness cases are consumed development evidence. Use them to motivate this protocol. Candidate fitting uses only the exact original normal training keys above. Historical validation, test and stress cases will not be used to choose the new candidates.
2. Require child-disjoint fitting, new validation and new final test. Check profile draw provenance and source event IDs against both new partitions and prior cohorts. Different namespaces alone do not establish separation. Block overlap, duplicated child/date keys, wrong hashes and metadata leakage. Labels, splits, identities and scenario fields are excluded from model inputs. Check the ordered feature schema and coverage mask for all three arms.
3. Seal the final cohort separately before tuning. A preparer or restricted process verifies its structural and quota invariants and publishes a manifest. The tuning process receives validation inputs and the final manifest only. Keep final rows, labels, scores and distributions inaccessible to tuning. No candidate selection or debugging uses the final cohort.
4. For each fitted challenger compute the linear empirical quantiles `[0.95, 0.97, 0.98, 0.99, 0.995]` of its own training-normal scores. For all arms use `-pipeline.score_samples(X)`; flag only when the raw score strictly exceeds the relevant cutoff. Ignore estimator `predict` and offsets for comparison decisions. Validate exact ties as unflagged. Missing coverage yields no score or flag.
5. On validation reject configurations with assessable FPR above 5% overall or more than 3 percentage points above A in any scenario. Rank the remaining configurations by higher screen-only recall, higher F1, then lower FPR. Break ties by family B before C where applicable, then Cartesian grid order and quantile order from the config. Keep one candidate per feasible family and designate the highest-ranked candidate as primary. If no family qualifies, stop before opening final test. If one family is infeasible, report it and carry the other only. No relaxed fallback search.
6. Before unlocking test access, write an exclusive selection lock with protocol/config and manifest hashes, validation trials and deterministic ranking, selected parameters, exact cutoffs, pipeline and metadata hashes, training key fingerprint, primary challenger ID and any infeasible arm. The final runner verifies the lock and scores each locked model once on the same cases. Secondary results are descriptive. A failed primary cannot be replaced using the same test. A later model or policy change starts a new version with a fresh untouched evaluation cohort.

## Metrics, uncertainty and pass criteria

Report confusion counts, precision, recall, F1, FPR and FNR on assessable days. Category recall uses assessed positives of that specific category. Report overall and scenario-wise scores. For monitoring, report assessment coverage and abstention over all days, unassessed positives and detection over all scheduled positives. False alerts per 100 assessable normal days equal FPR times 100. Include counts and denominators with rates. Undefined metrics are null and cannot pass a criterion. Report the synthetic prevalence alongside precision and F1.

Use seed 2026112403 and 10,000 paired percentile child-cluster bootstrap replicates, stratified by scenario. Resample entire children within scenario and reuse the same resampled children for all arms. Report nominal 95% intervals for descriptive summaries. For the primary challenger, use two-sided Bonferroni intervals with tail probability 0.05/(2×8) for the eight prespecified differences. They are screen recall, overall FPR, overall F1, language recall, mixed recall, and FPR in each of the three scenarios. Report effective replicate counts. Mark any requirement inconclusive if its required bound cannot be estimated or its interval fails to establish the required margin. Cluster intervals are approximate and can be unstable with sparse positives or very few discordant children.

| Final gate | Requirement |
|---|---|
| Screen-only recall | Point improvement at least +10 percentage points versus A, with adjusted lower bound strictly above zero. |
| Overall FPR | Challenger point FPR at most 5%, with adjusted upper bound on increase versus A at most +1 point. |
| Shift-specific FPR | Adjusted upper bound on increase versus A at most +3 points in each scenario. |
| Overall F1 | Adjusted lower bound on difference at least −2 points. |
| Language-only and mixed recall | Adjusted lower bound on each difference at least −5 points. |
| Coverage and preservation | Identical assessment mask across arms, no flag on insufficient-data days, and frozen A hashes and prediction parity. |

The ten-point screen target addresses the previously observed screen misses. The pooled FPR rules bound absolute burden and incremental false alerts. The somewhat wider scenario margin makes subgroup guardrails explicit where fewer children are available. The two- and five-point margins limit losses in other categories and overall F1. These are research screening values. Their usefulness depends on cohort composition and interval precision. Synthetic labels and prevalence do not establish real-world child-safety validity, and passing the screen advances the candidate only to separate offline XAI validation.

## Next implementation stage

Implement `experiments/model_comparison_v1/prepare_cohorts.py` and its split, schema, coverage and seal checks before writing any training entry point. Store validation and reserved final outputs in distinct new directories under `data/model_comparison_v1/`. This preregistration does not generate cohort records, fit models, score the final test, or change the frozen SHAP gate, urgent rules, Function 1 or dashboard.

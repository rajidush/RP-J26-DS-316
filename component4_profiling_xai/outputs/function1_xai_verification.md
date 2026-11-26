# Function 1 XAI correctness verification

The saved Function 1 model and its cluster-distance explanation agreed with an independent numerical reference on the audited synthetic cases. The audit found defects in input validation, session counting, and parent-facing wording. The corrections leave the model, scaler, feature order, contribution formula, severity mapping, and trend thresholds intact.

## Cluster assignment and evidence

Function 1 aggregates each selected group of sessions into six features: four emotion shares, the share with the upstream self-regulation flag, and mean dialogue turns. The saved StandardScaler transforms that vector in metadata feature order. The saved K-Means model selects the assigned cluster. The explanation selects the closest remaining centroid by squared Euclidean distance, using the lower cluster ID when alternative distances tie.

For standardized feature `z[j]`, assigned centroid `a`, and alternative centroid `b`, the contribution is `(z[j] - b[j])² - (z[j] - a[j])²`. A positive term favors the assigned centroid in this comparison, a negative term favors the alternative, and a zero term contributes equally to the two distances. Summing across the six features gives the alternative squared distance minus the assigned squared distance.

The reference test counted the raw features from session rows, applied the saved scaler means and scales directly, and calculated scalar centroid distances. It used the difference-of-squares identity `(a[j] - b[j]) × (2z[j] - a[j] - b[j])` to check each term independently. It compared the manual assignment with saved-model predictions and checked both distances, all contributions, and their sum to an absolute tolerance of `1e-12`. Repeated reports for the named educator cases produced identical XAI results after input shuffling. Controlled centroids checked exact ties, near ties, and deterministic selection between equally close alternatives.

## Parent explanation and input findings

- The explanation describes one grouped session summary and the actual K-Means assignment. A saved-model example with a calm majority matched a different pattern.
- Parent wording pairs a positive contribution with the actual recorded proportion. A contribution can support a low proportion or an observed zero. High observed values with negative contributions, zero terms, and negligible positive terms are excluded as support. A tied match or missing usable evidence produces neutral wording.
- The upstream self-regulation flag is triggered by language cues. The revised wording describes cues that can suggest thinking about what to do differently. The explanation makes no inference about whether the child acted on a thought, why they behaved a certain way, or any diagnosis. The distance margin stays in the research output as a distance difference.
- Missing timestamps and missing, non-finite, or non-whole turn counts now fail validation before aggregation. The report counts unique cleaned sessions for the requested child before applying the minimum. The one-child explanation API now rejects mixed or empty input explicitly.

## Trend findings

Windows are 14-day UTC intervals anchored at midnight on the child's first recorded day. At least three sessions qualify a window for classification. The current label uses the latest qualifying window, with an all-sessions prediction when the total meets the minimum and no individual window qualifies. At least three reliable windows are needed for a trend. The earlier mean uses the first two reliable windows, and the recent mean uses the last two. With exactly three qualifying windows, the middle window is included in both means.

The reference calculated each window, both means, and the difference independently. A difference of at least +1 is Worsening, a difference of at most −1 is Improving, and values between the thresholds are Stable. Fewer than three reliable windows gives Insufficient Data. All 16 combinations of current pattern and trend status were checked with actual saved-model predictions. A +0.5 change with a different latest pattern stayed Stable. A valid pattern comparison can also appear with Insufficient Data for the trend. Stable wording now describes the absence of a clear overall shift and allows for changes between periods.

| Synthetic source | Child | Current pattern | Reliable windows | Earlier mean | Recent mean | Difference | Trend |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Educator export | C001 | Calm and reflective | 3 | 2.0 | 1.0 | −1.0 | Improving |
| Educator export | C004 | Guarded responses | 5 | 0.0 | 1.0 | +1.0 | Worsening |
| Educator export | C007 | Calm and reflective | 5 | 1.0 | 0.0 | −1.0 | Improving |
| Educator export | C031 | Harder to read | 4 | 1.0 | 1.0 | 0.0 | Stable |
| Dashboard demo | C001 | Calm and reflective | 5 | 0.0 | 0.0 | 0.0 | Stable |
| Dashboard demo | C004 | Guarded responses | 5 | 0.0 | 2.0 | +2.0 | Worsening |

C007 and C031 are absent from the dashboard demo. Both were verified using the educator export.

## Verification status and research limits

The arithmetic, window selection, trend classification, and evidence-to-wording path are verified within the audited synthetic cases. Automated tests check the UI text and fallbacks. Parent comprehension and usefulness in practice remain unmeasured.

The regression run passed 331 distinct test cases with zero failures across the Streamlit pages, UI services, Component 4, and Component 3-to-4 integration. The audit adds 35 cases in the dedicated test module and two cases for near ties. A final rerun passed all 103 audit, parent-view, and page cases after the language-cue wording was clarified. `git diff --check` passed.

The educator export is synthetic training data. These checks establish implementation consistency and provide no estimate of generalization to real children. The explanation is relative to one alternative centroid, so it cannot establish causation or clinical validity. Emotion labels and the self-regulation flag inherit the limits of the recorded session data. The ordinal severity mapping and window thresholds are existing research choices whose clinical significance was outside this audit.

Reference tests and reviewed code are available in the [independent audit tests](../../tests/test_function1_xai_audit.py), [Function 1 implementation](../src/emotional_pattern.py), [UI service](../../ui/services.py), [parent wording](../../ui/parent_guidance_view.py), and [dashboard](../../ui/pages/emotional_pattern.py).

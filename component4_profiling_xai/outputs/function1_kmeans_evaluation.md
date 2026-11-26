# Function 1 K-Means model evaluation

The saved four-cluster model is reasonable for grouping the original synthetic child profiles. The four groups are more distinct when each child is summarized over all sessions than when the same model is applied to individual 14-day windows. A fifth cluster improves several experimental scores, but the additional child-level group consists of three children from the existing calm cluster. These results support retaining four interpretable categories for the current research use, with greater caution about the separation and stability of window assignments.

## Dataset and preprocessing

The [Component 4 evaluation CSV](../data/educator_sessions.csv) is byte-identical to the Component 3 educator export. It contains 754 sessions from 40 simulated children. Their dialogue and behavior profiles were generated from predefined archetypes, then processed by the Component 3 session classifier. The dataset card describes the generation and labeling process. [Component 3 dataset card](../../component3_socratic_educator/out/DATASET_CARD.md)

| Data check | Result |
| --- | --- |
| Missing cells, exact duplicate rows, repeated session IDs | 0 each |
| Invalid timestamps, emotion values, turn counts, or self-regulation flags | 0 each |
| Sessions per child | 12–25, median 19, mean 18.85 |
| Session labels | Distressed 244 (32.4%), calm 214 (28.4%), defensive 198 (26.3%), unclear 98 (13.0%) |
| Nonempty 14-day windows | 195, including 147 reliable and 48 below the minimum |
| Empty windows between each child's first and last observed windows | 1 |
| Reliable windows per child | 1–5, median 4. All 40 children contributed. |
| Distinct valid aggregate vectors | 40 of 40 child vectors and 116 of 147 window vectors |

The existing cleaner normalizes UTC timestamps, categories, and boolean values, validates turn counts, and removes duplicate records. Each window starts on a 14-day boundary measured from the first session's UTC calendar day. At least three sessions qualify a window. The training unit is one six-feature vector per child over all sessions. The window view recomputes the six features for each qualifying window. Windows from the same child are repeated measurements. Children with more qualifying windows contribute more vectors to the descriptive window results.

| Original-scale feature | Child median (IQR) | Window median (IQR) | Tukey flags, child / window |
| --- | ---: | ---: | ---: |
| Calm-session share | 15.9% (8.3–56.3%) | 20.0% (0–50.0%) | 0 / 0 |
| Defensive-session share | 14.7% (7.7–58.6%) | 16.7% (0–50.0%) | 0 / 0 |
| Distressed-session share | 16.7% (12.5–44.8%) | 20.0% (0–50.0%) | 1 / 0 |
| Unclear-session share | 4.9% (0–13.0%) | 0% (0–25.0%) | 6 / 10 |
| Share with the self-regulation language flag | 4.8% (0–8.9%) | 0% (0–6.3%) | 6 / 36 |
| Average dialogue turns | 2.125 (2.062–2.223) | 2.000 (2.000–2.333) | 6 / 13 |

The 1.5×IQR rule flagged 119 session turn counts because both quartiles were two turns. Counts ranged from 2 to 4. The aggregated flags similarly describe tails of bounded or zero-heavy features. All features passed checks for finite values, valid proportions, emotion shares summing to one, and plausible turn counts. Flagged observations were retained.

## Saved model and quality

The loaded artifact is scikit-learn 1.6.1 K-Means with **four clusters**, Lloyd's algorithm, k-means++ initialization, 10 starts, seed 42, 300 maximum iterations, and tolerance 0.0001. The selected fit converged in two iterations. The saved StandardScaler centers features and scales each by its fitted standard deviation. The notebook fits the scaler and model on one vector per child and names each cluster from the largest mean emotion share. All metrics below use the saved feature order and standardized space. [Training notebook](../notebooks/Function_1___Emotional_Pattern_Clustering.ipynb)

| Evaluation set | Vectors | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Inertia | Cluster counts 0 / 1 / 2 / 3 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Training export, child aggregates | 40 | 0.6428 | 0.5214 | 68.137 | 35.938 | 10 / 12 / 12 / 6 |
| Training export, reliable windows | 147 | 0.3797 | 0.9485 | 75.056 | 521.399 | 38 / 45 / 43 / 21 |
| Separate dashboard demo, child aggregates | 5 | 0.4533 | 0.6074 | 6.089 | 12.378 | 0 / 3 / 0 / 2 |
| Separate dashboard demo, reliable windows | 25 | 0.6757 | 0.4773 | 70.798 | 75.343 | 2 / 10 / 6 / 7 |

The child-level Silhouette, Davies–Bouldin, and inertia reproduce the notebook and artifact results. Silhouette rewards cohesion and separation, Davies–Bouldin rewards lower within-cluster scatter relative to separation, and Calinski–Harabasz rewards a larger between- to within-cluster dispersion ratio. The child and window views differ in sample size and aggregation, which affects Calinski–Harabasz and total inertia. Inertia is the sum of squared distances to the assigned model centroids on each evaluation view. It typically falls as more clusters are fitted and does not establish an optimal k by itself. Partition metrics return null when there are fewer than two occupied clusters or when every point occupies its own cluster.

## Experimental cluster counts

Each experiment used the same saved scaling and 10 k-means++ starts. All k values from 2 to 8 were supported by the sample counts and distinct vectors. The tables show seed 42. Cluster counts are in experimental ID order and IDs need not correspond across k values.

| Child k | Silhouette | Davies–Bouldin | Calinski–Harabasz | Inertia | Cluster sizes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 0.4157 | 0.8121 | 17.634 | 163.929 | 6 / 34 |
| 3 | 0.4693 | 0.8726 | 29.216 | 93.050 | 12 / 22 / 6 |
| 4 | 0.6428 | 0.5214 | 68.137 | 35.938 | 10 / 12 / 12 / 6 |
| 5 | 0.6293 | 0.5182 | 79.033 | 23.923 | 10 / 9 / 12 / 6 / 3 |
| 6 | 0.5922 | 0.6226 | 75.467 | 19.838 | 10 / 9 / 12 / 4 / 3 / 2 |
| 7 | 0.5371 | 0.7914 | 71.454 | 17.153 | 3 / 4 / 12 / 10 / 5 / 3 / 3 |
| 8 | 0.4361 | 0.8574 | 69.707 | 14.771 | 9 / 10 / 4 / 5 / 4 / 2 / 3 / 3 |

| Window k | Silhouette | Davies–Bouldin | Calinski–Harabasz | Inertia | Cluster sizes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 0.3357 | 1.1162 | 54.325 | 960.884 | 121 / 26 |
| 3 | 0.3042 | 1.2826 | 60.219 | 719.288 | 61 / 62 / 24 |
| 4 | 0.3725 | 0.9695 | 80.215 | 492.346 | 20 / 40 / 64 / 23 |
| 5 | 0.4032 | 0.8987 | 92.532 | 366.246 | 35 / 38 / 19 / 38 / 17 |
| 6 | 0.3835 | 1.0167 | 85.458 | 327.728 | 31 / 38 / 17 / 17 / 26 / 18 |
| 7 | 0.3884 | 1.0316 | 81.955 | 292.726 | 16 / 13 / 16 / 10 / 25 / 30 / 37 |
| 8 | 0.3963 | 1.0295 | 78.698 | 266.134 | 29 / 10 / 25 / 12 / 13 / 37 / 14 / 7 |

Child-level k=4 has the best Silhouette and a pronounced reduction in inertia from k=3. The experimental child k=4 reproduces the saved assignments with ARI 1.000. Child k=5 slightly improves Davies–Bouldin and improves Calinski–Harabasz while splitting the 12 calm assignments into groups of 9 and 3. The window k=5 experiment leads the three separation metrics among window-fitted candidates, but repartitions recorded patterns, including splitting the saved calm group primarily across two clusters (25 and 17 of 45 windows). The window-fitted k=4 experiment agrees only moderately with saved window assignments (ARI 0.577). A window-specific model is a different research question and would require child-grouped validation on new data and evidence that any extra cluster has a reproducible interpretation.

## Stability

Seed stability uses 10 independent initializations of the experiment configuration with seeds `42, 0, 1, 2, 3, 4, 5, 6, 7, 8`, each with `n_init=10`. ARI compares all 45 pairs per k and is invariant to ID permutation. Cluster sizes are aligned by maximum assignment overlap. The full results include minimum, median, maximum, and standard deviation for the ARI and cluster sizes.

| k | Child ARI mean (min) | Window ARI mean (min) |
| ---: | ---: | ---: |
| 2 | 1.000 (1.000) | 0.966 (0.936) |
| 3 | 1.000 (1.000) | 0.758 (0.243) |
| 4 | 1.000 (1.000) | 0.891 (0.654) |
| 5 | 1.000 (1.000) | 1.000 (1.000) |
| 6 | 0.993 (0.980) | 0.961 (0.889) |
| 7 | 0.962 (0.916) | 0.904 (0.788) |
| 8 | 0.900 (0.816) | 0.862 (0.715) |

For k=4, child cluster sizes stayed at 10 / 12 / 12 / 6 with zero variability. Aligned window size ranges were 19–20 / 40–57 / 49–64 / 22–29, with population SDs of 0.30 / 4.92 / 4.28 / 1.83. Window seed ARI SD was 0.114.

The grouped analysis fixed a common set of 10 children with split seed 316. Each of 10 runs sampled 24 of the other 30 children without replacement, fitted a scaler and k=4 experimental model on those children only, and predicted on the common evaluation set. The K-Means seed stayed at 42 to isolate changes in the fit sample. All windows from a child stayed together.

| Common evaluation view | Pairwise ARI mean / min / max | ARI SD | Mean ARI vs saved |
| --- | --- | ---: | ---: |
| 10 child aggregates | 1.000 / 1.000 / 1.000 | 0.000 | 1.000 |
| 33 windows from those children | 0.896 / 0.762 / 1.000 | 0.066 | 0.918 |

The saved model assigned none of the common child aggregates to Distressed, so the perfect child-level ARI does not establish coverage of all four clusters. The aligned window count ranges on the common set were 1–2 / 14–16 / 13–15 / 2–3 in saved cluster order. Grouped ARI measures within-export reproducibility. The common children were excluded from the experimental fits but were included in the original saved model's training data. Agreement with saved predictions therefore measures consistency with that reference.

## Interpretation of the saved clusters

Each feature cell is **mean / median / sample SD**, calculated on the original scale across assigned windows. The first five features use percentages and percentage points. Mean turns uses dialogue turns.

| ID and existing label | Windows | Calm | Defensive | Distressed | Unclear | Self-regulation flag | Mean turns |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| 0 Distressed | 38 (25.9%) | 10.6 / 0.0 / 13.6 | 4.8 / 0.0 / 10.1 | 80.1 / 83.3 / 19.7 | 4.5 / 0.0 / 9.4 | 2.0 / 0.0 / 6.2 | 2.070 / 2.000 / 0.134 |
| 1 Calm / reflective | 45 (30.6%) | 69.4 / 66.7 / 20.2 | 11.2 / 0.0 / 15.8 | 13.6 / 14.3 / 15.8 | 5.7 / 0.0 / 12.5 | 24.8 / 20.0 / 24.4 | 2.089 / 2.000 / 0.188 |
| 2 Defensive | 43 (29.3%) | 11.4 / 0.0 / 16.0 | 70.0 / 66.7 / 23.2 | 11.8 / 10.0 / 13.0 | 6.8 / 0.0 / 12.8 | 0.5 / 0.0 / 3.0 | 2.202 / 2.250 / 0.207 |
| 3 Withdrawn / unclear | 21 (14.3%) | 15.4 / 16.7 / 15.8 | 10.6 / 0.0 / 15.0 | 17.9 / 14.3 / 20.9 | 56.0 / 62.5 / 22.1 | 3.6 / 0.0 / 9.3 | 2.910 / 3.000 / 0.317 |

The dominant recorded emotion share matches the emotion component of each label. Distressed and Defensive have high corresponding shares and window Silhouettes of 0.551 and 0.442. Calm / reflective has the highest mean self-regulation language-flag rate and the weakest window Silhouette (0.189). Seven of its 45 windows had negative Silhouette values, indicating greater average similarity to another assigned group under that metric. Withdrawn / unclear has one negative value, the largest unclear share, and longer average dialogues. Overlap is visible in the window projection, and the calm and distressed clusters have identical turn-count interquartile ranges of 2.0–2.0. The evidence supports **calm** and **unclear** as descriptions of recorded labels. Reflection and withdrawal go beyond what the language flag, unclear label, and turn counts establish.

## Visualizations

- [Silhouette vs k](function1_kmeans/silhouette_vs_k.png)
- [PCA of saved window assignments](function1_kmeans/saved_assignments_pca.png)
- [Window feature profiles](function1_kmeans/window_feature_profiles.png)

The [evaluation runner](../evaluate_function1.py) can regenerate the omitted elbow curve, Davies–Bouldin vs k plot, Calinski–Harabasz vs k plot, and saved cluster size plot using the reproduction command below.

PCA was fitted on the 147 standardized window vectors. PC1 explains 38.48% of variance and PC2 explains 26.48%, for 64.96% together. The projection is for visualization and can obscure separation along discarded dimensions. The heatmap colors represent standardized means, and the cell labels give original-scale means.

## Validity and conclusion

**Original training data.** The child metrics are in-sample. The windows come from the same sessions and children with a different aggregation. Generated archetypes can make synthetic clusters unusually distinct. The notebook's ARI against simulated profiles and its internal child split are provenance for the model, with the same dataset limitations.

**Separate synthetic data.** The dashboard demo uses 152 scripted sessions from five fictional trajectories that were outside fitting the saved artifact. It is a supplemental out-of-training-source check with overlapping ID strings but separately generated records. It was designed as a demo, yields only two occupied clusters at child level, and is too small and stylized to establish held-out benchmark performance. There is no dedicated external synthetic Function 1 benchmark in the reviewed data.

**Genuinely independent data.** None was available. External validation needs independently collected longitudinal sessions across children, coverage of all four patterns, and checks on how consistently the upstream session labels are applied. Clinical validity and real-world accuracy remain unmeasured.

The current configuration offers four useful descriptions of synthetic child aggregates and reproducible assignments between fits on that data. Window separation is weaker and window stability depends on the fit sample and initialization. The additional groups found at k=5 merit investigation, with evidence still needed to justify distinct emotional labels or a change to the saved model.

## Reproduction

From the repository root, install the repository dependencies and Matplotlib, then run the research module. The reference run used Python 3.13.7, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, scikit-learn 1.6.1, and Matplotlib 3.10.7. The runner limits numerical thread pools to one for the documented run.

```powershell
python -m pip install -r requirements.txt matplotlib==3.10.7
python -m component4_profiling_xai.evaluate_function1
```

The runner writes figures and machine-readable metrics, including split membership, artifact and data SHA-256 checksums, and full cluster profiles. It verifies that the four Function 1 artifact hashes are unchanged. Experimental fits exist in memory. [Evaluation runner](../evaluate_function1.py) · [Full metrics](function1_kmeans/metrics.json)

The regression suites passed 345 distinct cases with zero failures, including 14 new evaluation tests. The evaluation tests passed again after the final validation and plot-formatting changes. `git diff --check` passed.

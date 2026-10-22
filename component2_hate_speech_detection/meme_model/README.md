# Meme model (Step 8: image / meme path)

A meme hate classifier for C2's image path (proposal SO3, FR5). It ships in
the app as ONNX and scores dropped images, but nothing it outputs changes a
child's outcome until it passes the calibration gate in
[`src/engine/fusion.py`](../src/engine/fusion.py) (invariant I-1). It does not
pass yet (see Calibration), so today it is evidence only. Once it passes, it
can only *raise* a score (I-2).

## Why this model

Constraints from the proposal: runs on a CPU laptop with no network (RQ3),
inside a 2 GB RAM budget with the text system already using 1.2 GB (measured),
and gives a calibrated probability (I-1).

| Approach | Reported HMC AUROC | Fits the laptop? | Decision |
|---|---|---|---|
| Fine-tuned large multimodal models: LLaVA-13B 85.3, Flamingo-80B 86.6 (as reported by Mei et al., 2024) | 85–87 | No: billions of parameters | Rejected |
| Zero-shot LMMs (GPT-4V 71.7 % accuracy; open models 37–65 %; HateSieve, 2024) | – | No (cloud) or weak | Rejected |
| Pro-Cap / PromptHate (Cao et al., 2022, 2023): caption every meme with BLIP-2, then a text model | not checked | No: a captioning model per meme | Rejected |
| ISSUES (Burbi et al., 2023): CLIP + textual inversion | 85.5 | Needs an extra inversion network on ViT-L/14 | Later |
| **Hate-CLIPper** (Kumar & Nandakumar, 2022): frozen CLIP + small fusion head | 85.8 (test_seen), 81.2–82.6 (dev_seen) | **Yes**: only the head trains; align fusion has ~5M parameters | **Chosen: step 1** |
| **RGCL** (Mei et al., 2024, ACL): Hate-CLIPper + retrieval-guided contrastive loss + k-NN | 87.0 | **Yes**: same head, one extra loss | Tried as step 2; no gain on B/32 (see Results) |

So: **frozen CLIP + Hate-CLIPper align-fusion head.** RGCL was tried on top
and dropped (see Results).

- Hate-CLIPper's align fusion matches its 1.1-billion-parameter cross fusion
  with ~5M parameters. The authors say the trend holds with ViT-B/32.
- RGCL gained +4.4 AUROC on a weaker encoder (ALIGN, 73.2 → 77.6), which is
  the situation B/32 is in.
- RGCL's k-NN classifier can take new labelled memes without retraining,
  which suits a policy-pack update model.
- The encoder is CLIP ViT-B/32 (`openai/clip-vit-base-patch32`, already in the
  local cache). Larger encoders (B/16, L/14) run on a GPU as upper bounds only,
  to measure what the laptop budget costs (RQ3).

**Comparison caveat.** Most published HMC numbers are on `test_seen`, whose
labels were never released. Our test split is `dev_seen`, where Hate-CLIPper
(ViT-L/14) reports **81.2–82.6**. That is the fair reference, not 85.8.

## Protocol

| Split | Memes | Used for |
|---|---|---|
| train | 8,000 | gradient steps |
| val | 500 (caption-grouped, confounder-matched; see [DATASHEET](../annotation/DATASHEET.md#composition)) | early stopping, temperature, threshold, every hyper-parameter |
| test | 500 (dev_seen) | reported once per configuration, never used to choose |

- Hate-CLIPper/RGCL optimiser settings: AdamW, lr 1e-4, weight decay 1e-4,
  gradient clip 0.1, batch 64, up to 20 epochs, early stopping on val AUROC.
- Hyper-parameters are picked by `grid.py` on mean val AUROC over 3 seeds; the
  winner is reported with 5 seeds (mean ± sd) plus a bootstrap 95 % CI for the
  seed ensemble.
- Calibration: temperature scaling on val (Guo et al., 2017); ECE reported
  before and after. Whether the score may fuse is decided by the I-1 gate
  below.
- Ablations: image only, text only, concat vs align vs cross, with and without
  the 833 Kaggle/WOAH label conflicts, with and without RGCL.

## Results: CLIP ViT-B/32 (2026-10-11; laptop CPU, re-runs on a Colab T4)

Test = dev_seen (500 memes, 49 % hateful). Heads: 5 seeds, mean ± sd, plus the
seed ensemble with a bootstrap 95 % CI. Raw figures are in
`evaluation/results/meme_*.json`.

| Model | Val AUROC | Test AUROC | Test accuracy @0.5 |
|---|---|---|---|
| MVP logistic regression (1,200 memes) | – | 0.658 | – |
| Logistic regression, image + text (8,000 memes) | 0.782 | 0.675 (CI 0.631–0.722) | 0.594 |
| Head, image only | 0.725 | 0.652 ± 0.011 | 0.588 |
| Head, text only | 0.764 | 0.642 ± 0.004 | 0.564 |
| **Head, align: chosen on val** | **0.822** | **0.751 ± 0.005**; ensemble 0.754 (CI 0.713–0.795) | 0.673 |
| Head, align, T4 re-run (30-epoch cap) | 0.823 | 0.755 ± 0.007; ensemble 0.761 (CI 0.722–0.802) | 0.670 |
| Head, align + RGCL (20 epochs, CPU) | 0.813 | 0.744 ± 0.011 | 0.672 |
| Head, align + RGCL (30 epochs, T4; paper setting) | 0.814 | 0.747 ± 0.015 | 0.667 |
| RGCL k-NN classifier (30 epochs, T4) | 0.803 | 0.741 ± 0.019 | – |
| *Reference: Hate-CLIPper, ViT-L/14 (published)* | – | *0.812–0.826* | – |

Selected head: align fusion, 3 pre-output layers, dropout 0.4, 4.2M
trainable parameters.

- **Image and caption together add about +0.10 AUROC** over either alone
  (0.751 vs 0.652 / 0.642). This is the evidence for a multimodal path (SO3).
- **Fusion ranking held on test.** In the grid (`meme_grid_clip-b32.json`),
  align beat concat and cross on both val and test, matching Hate-CLIPper.
- **The gain over the baseline is beyond noise.** +0.08 AUROC over logistic
  regression; the ensemble CI (0.713–0.795) lies above 0.675.
- **We are 6–7 points below the published L/14 result.** That is the cost of
  a ~3× smaller encoder that fits the laptop budget, and it is the RQ3
  trade-off to report.
- **RGCL does not help B/32, even at the paper's 30 epochs.** It scores
  lower on val (0.814 vs 0.823, the deciding split) and on test (0.747 vs
  0.755), with twice the seed spread. Its best epochs (16–29) fall inside the
  cap, so it was not cut short. The plain align head is the final model.
  - Hypothesis, not tested: RGCL retrieves positives and hard negatives in
    the head's embedding space, and B/32's weaker features make those
    look-ups noisier than with the paper's L/14.
- **The GPU re-run reproduces the CPU result** within seed noise (0.755 vs
  0.751). A T4 trains one seed in ~10 s, against ~170 s on the laptop CPU.
- **Calibration does not pass the I-1 gate.** Temperature scaling halves
  ECE (0.22 → 0.11); see the next section for the gate.
- **A val-tuned threshold is unstable** (0.34–0.77 across seeds), so accuracy
  is reported at 0.5 and AUROC is the headline metric.

## Calibration: the I-1 gate (2026-10-15)

`calibrate.py` decides whether the meme score may move a decision. The rule
was set before the test result was read:

- **Gate:** consistency resampling (Bröcker & Smith, 2007), p ≥ 0.05, for the
  shipped temperature-scaled score, on val (5-fold, in distribution) **and**
  on test.
- **Why not an ECE cut-off:** on 500 memes a perfectly calibrated score still
  shows ECE ≈ 0.05 (95th percentile ≈ 0.067), so one number cannot separate
  noise from miscalibration.
- **Base rate:** a calibrated probability holds for one base rate. Train and
  val are 35 % hateful, dev_seen 49 %. Test is reweighted to the training base
  rate (label shift; Saerens et al., 2002), so the gate judges the score, not
  the base rate.

| Calibrator (fitted on val) | Val ECE (5-fold) | Val p | Test ECE, raw | Test ECE at 35 % | Test p |
|---|---|---|---|---|---|
| **Temperature (shipped)** | 0.049 | 0.43 | 0.123 | 0.076 | **0.008** |
| Platt | 0.044 | 0.64 | 0.145 | 0.074 | 0.008 |
| Isotonic | 0.030 | 0.92 | 0.136 | 0.081 | < 0.001 |

Reliability per decision band (temperature-scaled; test reweighted to 35 %):

| Score band | Test memes | Mean score | Observed hateful | Val (5-fold): mean / observed |
|---|---|---|---|---|
| 0–0.35 | 256 | 0.157 | 0.222 | 0.156 / 0.136 |
| 0.35–0.55: stage 1, fusion agreement band | 103 | 0.435 | **0.348** | 0.454 / 0.534 |
| 0.55–0.75: persona thresholds | 81 | 0.646 | 0.633 | 0.673 / **0.565** |
| 0.75–1 | 60 | 0.771 | 0.716 | 0.773 / 0.742 |

**Verdict: FAIL. The meme score stays evidence only (I-1).**

- On val the score is calibrated (p = 0.43), so temperature scaling works for
  the training distribution.
- On dev_seen it is not, even after the base-rate correction (p = 0.008).
  No calibrator fitted on val passes (all p < 0.01), so the cause is a shift
  between train and dev_seen beyond the base rate, not the choice of
  calibrator.
- The miss is worst at 0.35–0.55: the score says 0.44, but 35 % are hateful.
  That band feeds the fusion agreement bonus (+0.15), which can lift a fused
  score past the 0.55 threshold, so it would change outcomes.
- 0.55–0.75 looks right on test (0.65 vs 0.63) but is over-confident on val
  (0.67 vs 0.57), so it is not reliable either.
- Children's screens are a further shift from both splits. The gate should be
  re-run on the child-context test set (roadmap Step 9) before the meme score
  is allowed to fuse. Ranking (AUROC) is unaffected by any of this.

Figures: `evaluation/results/meme_clip-b32_align_calibration.json`.

## Shipping form and app wiring (2026-10-15)

The app never imports torch, so the model ships as ONNX
(`tools/export_meme_onnx.py`; figures in
`evaluation/results/meme_onnx_parity.json`). The runtime is
[`src/engine/meme_scorer.py`](../src/engine/meme_scorer.py): ONNX Runtime +
`tokenizers`, its own CLIP preprocessing, loaded on first use.

| Check | Result |
|---|---|
| fp32 ONNX vs torch, same inputs | max logit difference 0.000006 |
| Runtime preprocessing + tokeniser vs the reported head (val) | AUROC 0.8251 vs 0.8252; max score difference 0.003; 0 band changes |
| INT8, vision tower only (val gate: AUROC drop ≤ 0.005, mean shift ≤ 0.02) | AUROC +0.000, mean shift 0.012: **kept** |
| Shipped INT8, test, dataset captions | AUROC 0.750 (reference 0.752) |
| Size / speed (laptop CPU, 2 threads) | 351 MB (fp32 594 MB); 65 ms per meme (fp32 124 ms) |

INT8 on the text tower fails (val AUROC −0.035, mean shift 0.087): its
activations have outliers that dynamic quantisation clips. Only the vision
tower is quantised. One preprocessing bug was caught by the parity check:
CLIP floors the resized long side, and rounding it instead moved one val
meme's score by 0.19.

**Wiring.** A dropped image (`Agent.check_image`, the panel's "try it") is
scored as one meme, with its OCR text as the caption. The score reaches the
analyzer as a `vision` signal with `calibrated = False` (copied from the I-1
gate into `meme.json`), so it is shown as evidence ("meme model 0.82
(evidence only)") and never changes the score, rung or C3 payload. Live screen
frames are scored too, through the picture finder (next section).

**End to end: the 500 test memes through the app path** (real OCR, both text
heads, meme scorer, decision):

| | Result |
|---|---|
| Meme score AUROC (OCR captions, INT8) | **0.733** |
| Text path alone, verdict-score AUROC | 0.560 |
| Text path, age 9: hateful memes alerted | 32 % (79 / 246) |
| Text path, age 9: benign memes alerted | 21 % (52 / 254) |
| Hateful memes the text path misses (age 9) where the meme score is ≥ 0.55 | 67 of 167 (and 47 of 254 benign memes score ≥ 0.55) |
| Latency per image, median / p95 | 421 / 806 ms (OCR 240, meme 98, text 78) |
| Peak memory: OCR + both text heads + meme model, no torch | 1,508 MB (NFR2 budget 2 GB) |

- **On memes, the text path is close to chance** (0.56), as the dataset
  intends: benign confounders reuse the hateful captions with a harmless
  picture. The meme model sees the picture and ranks far better (0.73).
- **The meme model could add detections once calibrated**: it scores 67 of
  the 167 hateful memes the text path misses at or above 0.55, at the cost
  of 47 benign memes.
- **It could not remove the text path's false alerts** on benign
  confounders (21 %), even once calibrated: fusion may raise a score, never
  lower it (I-2). That is the design's safety trade-off, and it should be
  stated as such.

## Memes on a live screen (2026-10-17)

A meme on a screen is a small part of the frame. Scored as one image with all
the screen's text as its caption, the meme model falls to **0.565 AUROC**,
close to chance. So the screen path first finds the pictures
([`src/readers/pictures.py`](../src/readers/pictures.py)), then scores each
crop with the OCR text inside it as the caption.

**Finder.** Classical, no model, after Chen et al. (ESEC/FSE 2020) and UIED (Xie
et al., 2020): mark blocks that are not a flat UI fill (a flat block counts as
UI only in the page's own UI colours), drop thin text with a morphological
opening, cut at background-coloured gutters with recursive XY-cut (Nagy &
Seth, 1984), and reject text paragraphs by flatness and colourfulness (Hasler
& Suesstrunk, 2003). Measured with `evaluation/screen_memes.py` on built
1920x1080 screens with known boxes: thresholds set on val memes, test
reported once.

| Layout (test, 40 pages each) | Memes | Found at IoU >= 0.7 | False finds per page | Time per frame |
|---|---|---|---|---|
| Social feed, light | 40 | 100 % | 0 | 117 ms |
| Social feed, dark | 40 | 100 % | 0 | 117 ms |
| Chat with an attached meme | 40 | 100 % | 0 | 94 ms |
| Gallery, 2x2 or 3x2, 4-16 px gutters | 196 | 99 % | 0 | 115 ms |
| Article with an inline picture | 40 | 95 % | 0 | 94 ms |
| Busy chat, no picture | 0 | - | 0 | 87 ms |

The first finder ("busy texture") found 10-40 % of val memes: it missed the
smooth parts of photos. Treating only flat UI colours as background fixed it
(80-95 %, then 97-100 % on val).

**Scores.** Each of the 500 test memes on its own feed, chat or article page,
full-screen OCR, the shipped INT8 model; a meme the finder misses scores 0:

| How the meme is scored | AUROC |
|---|---|
| The meme alone, OCR caption (reference) | 0.733 |
| The whole screen as one image | 0.565 |
| **The finder's crop, OCR text inside it** | **0.713** (98.4 % found) |
| Crop, found memes only, vs those memes alone | 0.721 vs 0.736 |

**Wiring.** `Agent.screen_tick` runs the finder on every changed frame
(`AgentConfig.memes`, on by default) and scores each new picture once: a
16x16 thumbnail fingerprint caches its score, so a meme that stays on screen
is not scored again. The strongest picture goes to the analyzer as evidence
only. Driving the real agent over a feed: the frame where a meme appears costs
~0.1 s to find plus 65 ms to score (plus ~3.5 s once, to load the model), an
unchanged frame 17 ms, and a new chat line under the same meme 0.48 s in all
with the meme taken from the cache. A dropped screenshot is read the same way;
a dropped meme is still scored whole.

**Limits.** Built screens are cleaner than real ones (no compression, ads or
video). The next check is real page captures, such as C1's scraped pages with
memes pasted in. A meme on a white page with a white backdrop has no visible
edge, so the finder can cut it short.

## Head v2: trained on what C2 reads (2026-10-22, Thunder L40)

v1 was trained on the dataset's typed captions, which C2 never sees. v2 trains
the same head, with v1's settings and seeds, on three views of every meme:
the clean caption, C2's own OCR caption, and the meme cut out of a built
screenshot by the finder (24,000 training rows). The v1 recipe is retrained
alongside as the controlled baseline. The success rule was fixed before the
run (`train_views.py`): v2 must beat it on the OCR and screen views and lose
at most 0.01 on clean.

| Test view (500 memes, 5-seed ensemble, 95 % CI) | Shipped v1 | v1 recipe, retrained | v2, three views |
|---|---|---|---|
| Clean caption | 0.752 | 0.762 (0.723–0.802) | 0.753 (0.712–0.795) |
| C2's OCR caption | 0.735 | 0.746 (0.703–0.789) | 0.744 (0.699–0.787) |
| Cut from a screenshot | 0.718 | 0.731 (0.689–0.778) | 0.736 (0.695–0.783) |

- **Verdict: fail.** OCR −0.002, screen +0.006, clean −0.009, all well
  inside the confidence intervals. The shipped v1 head stays.
- A likely reason: CLIP is frozen, so words the OCR drops or garbles are lost
  before the head sees them; a head alone cannot learn them back. Adapting
  CLIP's last layers on these views (e.g. LoRA) is the next test.
- The retrained v1 recipe, as a 5-seed ensemble, beats the single shipped
  head by about +0.01 on every view and matches the earlier T4 re-run (0.761),
  so the recipe reproduces on new hardware.
- Screen-view numbers differ slightly from the section above (0.718 vs 0.713)
  because these pages use four layouts and one page per meme, and the head
  runs in fp32 rather than the shipped INT8 ONNX.

Evidence: [`TRAINING_CARD.md`](../evaluation/results/meme_v2/TRAINING_CARD.md)
(techniques, code commit, machine, packages, data fingerprints) and the logs
beside it. How to run it: [`thunder/README.md`](thunder/README.md).

## Run

```bash
python -m component2_hate_speech_detection.meme_model.embed --encoder clip-b32
python -m component2_hate_speech_detection.meme_model.train --logreg
python -m component2_hate_speech_detection.meme_model.grid --encoder clip-b32
python -m component2_hate_speech_detection.meme_model.train --fusion align --rgcl 1 --knn 10 --tag rgcl
python -m component2_hate_speech_detection.meme_model.calibrate --run meme_clip-b32_align
python -m component2_hate_speech_detection.tools.export_meme_onnx --run meme_clip-b32_align
python -m component2_hate_speech_detection.evaluation.screen_memes --split test --scores
```

Training runs on a GPU (`--device auto` picks it up; Colab or Thunder). Head
training needs only `data/embeddings/<encoder>.npz` and
`data/labels/finetune/*.jsonl`, not the images. The laptop runs the final
model for the RQ3 checks.

Embeddings, predictions and trained weights stay in git-ignored folders
(`data/`, `models/`). Only aggregate results (`evaluation/results/meme_*.json`)
are suitable to commit.

## References

- Bröcker, J. & Smith, L. A. (2007). Increasing the reliability of reliability diagrams. *Weather and Forecasting*, 22(3), 651–661.
- Burbi, G. et al. (2023). Mapping memes to words for multimodal hateful meme classification. *ICCV Workshops*.
- Cao, R. et al. (2022). Prompting for multimodal hateful meme classification. *EMNLP*.
- Cao, R. et al. (2023). Pro-Cap: Leveraging a frozen vision-language model for hateful meme detection. *ACM MM*.
- Chen, J. et al. (2020). Object detection for graphical user interface: old fashioned or deep learning or a combination? *ESEC/FSE*.
- Guo, C. et al. (2017). On calibration of modern neural networks. *ICML*.
- Hasler, D. & Suesstrunk, S. (2003). Measuring colourfulness in natural images. *Proc. SPIE 5007, Human Vision and Electronic Imaging VIII*.
- Kiela, D. et al. (2020). The Hateful Memes Challenge. *NeurIPS*.
- Kumar, G. K. & Nandakumar, K. (2022). Hate-CLIPper: Multimodal hateful meme classification based on cross-modal interaction of CLIP features. *NLP4PI Workshop*. arXiv:2210.05916.
- Mei, J. et al. (2024). Improving hateful meme detection through retrieval-guided contrastive learning. *ACL*, 5333–5347.
- Nagy, G. & Seth, S. (1984). Hierarchical representation of optically scanned documents. *Proc. 7th International Conference on Pattern Recognition*.
- Radford, A. et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.
- Saerens, M., Latinne, P. & Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Computation*, 14(1), 21–41.
- Xie, M. et al. (2020). UIED: a hybrid tool for GUI element detection. *ESEC/FSE Demonstrations*.

# Meme model (Step 8: image / meme path)

A meme hate classifier for C2's image path (proposal SO3, FR5). It is a
research build: nothing it outputs changes a child's outcome until it passes the
calibration gate in [`src/engine/fusion.py`](../src/engine/fusion.py)
(invariant I-1), and then it can only *raise* a score (I-2).

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
| **RGCL** (Mei et al., 2024, ACL): Hate-CLIPper + retrieval-guided contrastive loss + k-NN | 87.0 | **Yes**: same head, one extra loss | **Chosen: step 2** |

So: **frozen CLIP + Hate-CLIPper align-fusion head, improved with RGCL.**

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
  before and after.
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
- **Calibration is not yet good enough for the I-1 gate.** Temperature
  scaling halves ECE (0.22 → 0.11), but part of the residual comes from the
  prior shift (35 % hateful in train, 49 % in test).
- **A val-tuned threshold is unstable** (0.34–0.77 across seeds), so accuracy
  is reported at 0.5 and AUROC is the headline metric.

## Run

```bash
python -m component2_hate_speech_detection.meme_model.embed --encoder clip-b32
python -m component2_hate_speech_detection.meme_model.train --logreg
python -m component2_hate_speech_detection.meme_model.grid --encoder clip-b32
python -m component2_hate_speech_detection.meme_model.train --fusion align --rgcl 1 --knn 10 --tag rgcl
```

Training runs on a GPU (`--device auto` picks it up; Colab or Thunder). Head
training needs only `data/embeddings/<encoder>.npz` and
`data/labels/finetune/*.jsonl`, not the images. The laptop runs the final
model for the RQ3 checks.

Embeddings, predictions and trained weights stay in git-ignored folders
(`data/`, `models/`). Only aggregate results (`evaluation/results/meme_*.json`)
are suitable to commit.

## References

- Burbi, G. et al. (2023). Mapping memes to words for multimodal hateful meme classification. *ICCV Workshops*.
- Cao, R. et al. (2022). Prompting for multimodal hateful meme classification. *EMNLP*.
- Cao, R. et al. (2023). Pro-Cap: Leveraging a frozen vision-language model for hateful meme detection. *ACM MM*.
- Guo, C. et al. (2017). On calibration of modern neural networks. *ICML*.
- Kiela, D. et al. (2020). The Hateful Memes Challenge. *NeurIPS*.
- Kumar, G. K. & Nandakumar, K. (2022). Hate-CLIPper: Multimodal hateful meme classification based on cross-modal interaction of CLIP features. *NLP4PI Workshop*. arXiv:2210.05916.
- Mei, J. et al. (2024). Improving hateful meme detection through retrieval-guided contrastive learning. *ACL*, 5333–5347.
- Radford, A. et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.

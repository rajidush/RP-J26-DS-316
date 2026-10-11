# Datasheet — C2 Hateful Memes labels

Follows *Datasheets for Datasets* (Gebru et al., 2021). This describes the **label
set** C2 builds on top of the Hateful Memes images. It contains only aggregate
figures and meme ids. No images or captions are reproduced (dataset licence).

**Maintainer:** Dilnuka Liyanage (IT23209152), Component 2, SLIIT J26-DS-316
**Version:** 1.2 (2026-10-11). **Codebook:** [`CODEBOOK.md`](CODEBOOK.md)

---

## Motivation

- **Purpose.** To fine-tune and evaluate a meme hate detector for C2, an
  on-device harmful-content guard for children aged 8–15, and to measure where
  the harm sits (text, image or both) and how serious it is *for a child*.
- **Gap filled.** Published labels say whether a meme is hateful and, for some
  splits, which group is attacked and how. They do not say how severe it is for
  a child, whether the meme endorses or counters the hate, or whether the
  published label itself holds up. D6–D9 add those.
- **Funding.** None; undergraduate research project.

## Composition

| Split | Memes | Hateful (canonical) | Labels available |
|---|---|---|---|
| train | 8,500 | 3,007 | D1–D5 (gold + derived) |
| dev_seen | 500 | 246 | D1–D5, plus D6–D9 (draft; human verification pending) |
| test_seen | 1,000 | — | none (labels never released) |

- **Instances.** One meme = one image + its overlaid caption (`text`) + an id.
- **Fine-tuning files** (`data/labels/finetune/`): `train` 8,000; `val` 500;
  `test` = dev_seen 500; `unlabelled_test_seen` 1,000.
- **How val is cut (v1.2, 2026-10-11).** Val is taken from train in whole
  caption groups, with seed 316, so no caption appears in both train and val.
  It is also confounder-matched: 30 % of val memes sit in caption groups
  that hold both labels (dev_seen: 31 %), and its hateful share equals
  train's (35.4 %).
  - **Why.** The first split was a plain label-stratified random sample. It
    put 30 % of val captions in train and had only 2 % confounder pairs, so a
    text-only probe scored 0.80 AUROC on val but 0.64 on dev_seen. Val was
    rewarding caption memorisation, which dev_seen punishes. This is a
    leakage pattern (Kapoor & Narayanan, 2023).
  - **After the fix.** The text-only probe scores 0.75 on val. Val ranks the
    baselines in the same order as dev_seen. Val is still easier than
    dev_seen, so report test figures, never val.
- **Label noise (known).** The two releases disagree on 833 train memes (9.8 %)
  and 4 dev_seen memes; every conflict is flagged in `label_conflict`. A blind
  adjudication of 60 sampled conflicts found WOAH right in 75 % of decidable
  cases (CI 0.61–0.84), so WOAH is canonical. WOAH has 44 hateful memes with
  no target or attack label (`fine_grained_missing`).
- **Confounders.** The dataset deliberately contains benign twins: the same
  caption on a harmless image, or the same image with a harmless caption. 946
  hateful memes have `text_sufficient = false`.
- **Sensitive content.** Nearly all hateful memes attack people for religion,
  race, nationality, sex, sexual orientation, gender identity or disability.
  Many show real public figures. Several show graphic violence or reference
  sexual abuse of minors.

## Collection and labelling

- **Images and gold labels.** Kiela et al. (2020): memes rebuilt from Getty
  Images stock photos with trained annotators. Fine-grained labels: WOAH 2021
  shared task (Mathias et al., 2021).
- **D6–D9 draft.** An AI model (Claude Opus 5.5) drafted all 500 dev_seen memes
  from the full image and caption.
  - **Blind to gold.** The gold label was hidden while drafting.
  - **Blind agreement with gold.** On D1, Cohen's κ = 0.664 and accuracy 0.832
    (n = 494). Six memes whose gold label was seen before drafting are excluded.
  - **Reconciliation.** All 86 disagreements were re-examined at full resolution:
    46 were draft errors (gold upheld), 30 are `unsure` and 10 are `disagree`
    (suspected gold errors).
- **Human verification.** Pending (codebook section 6, steps 4–6). Until it is
  done, D6–D9 are **drafts and must not be reported as ground truth.**

## Draft label distribution (dev_seen, n = 500, pre-verification)

| Dimension | Distribution |
|---|---|
| D6 harm_locus | multimodal 235 · none 137 · text 100 · image 28 |
| D7 framing | commits 349 · none 132 · condemns 12 · reports 7 |
| D8 child_severity | 0: 137 · 1: 96 · 2: 175 · 3: 92 |
| D9 gold_check | agree 460 · unsure 30 · disagree 10 |

D8 by gold label:

| Gold | Severity 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| not_hateful | 135 | 79 | 30 | 10 |
| hateful | 2 | 17 | 145 | 82 |

47 % of *not-hateful* memes (119 of 254) still rate ≥ 1 for a child: crude, sexual or
violent content outside the hate definition. This is the "not hateful ≠
child-safe" gap that C2's severity scale exists to capture.

## Preprocessing

`build_labels.py` lower-cases and whitespace-normalises captions only when it
compares them for D5. Stored captions are unchanged. No images are altered.

## Uses

- **Intended.** Fine-tuning and evaluating C2's meme classifier, ablating text
  vs image paths (D6), testing counter-speech handling (D7), and age-band
  thresholds (D8).
- **Not intended.** Training generators of hateful content, profiling real
  people shown in the memes, or any use the dataset licence forbids.
- **Caveat.** A model trained on canonical labels inherits the 25 % of conflicts
  where WOAH appears wrong. Report results with and without `label_conflict`
  memes.

## Distribution and maintenance

- **Not redistributed.** Images, captions and all label files stay in the
  git-ignored `data/` folder (public repo, research-only licence). Others
  rebuild them with `fetch_data.py` → `build_labels.py` → `import_drafts.py`.
- **Errata.** Rows marked `disagree` are reported as suspected gold-label errors
  with a one-line reason; the gold label is never edited.

## References

Gebru et al. (2021); Kiela et al. (2020); Mathias et al. (2021). See
[`CODEBOOK.md`](CODEBOOK.md#8-references) for full citations.

Kapoor, S. & Narayanan, A. (2023). Leakage and the reproducibility crisis in
machine-learning-based science. *Patterns*, 4(9), 100804.

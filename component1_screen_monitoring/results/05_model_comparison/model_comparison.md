# Model comparison — A (current) vs B (screen fine-tune v2)

Generated 2026-10-21T03:21:15+00:00 by `src/compare_models.py`.

- **A** = `models/violence-vit-screen-ft-v1` (`config.MODEL_DIR`): `jaranohaal/vit-base-violence-detection` fine-tuned on full-frame RLVS images (Week 2).
- **B** = `models/violence-vit-screen-ft-v2`: the same pretrained checkpoint fine-tuned on screen composites + tiles with 40 % RLVS frames (notebook section 15; run u4_lr3e-5). Threshold 0.25 = B's max-F1 threshold on screen **val**.

## How to read this

- The **screen test set was never used for training or model selection** — B's run and its 0.25 threshold were chosen on screen val; these are the first and only test numbers.
- The **full test set** (520 images: 416 composites + 104 plain pages) reuses RLVS frames that were in the RLVS **training** split (203 of its 300 frames), i.e. frames A was trained on. B never saw them (its training excluded every frame near a test frame), so the full set slightly favours A.
- The **clean subset** (242 images: 138 composites with RLVS val/test frames + 104 plain pages) uses no training frames of either model — the fairer comparison, but small: one violent image ≈ 1.6 recall points.
- `tiles` = whole frame + 2×2 overlapping tiles + centre crop, flagged if any crop ≥ threshold (the live strategy); `full` = whole frame only. Missed = violent images not flagged; false alarms = safe images flagged. FPR = false alarms / safe images.
- ms/img = preprocess + forward pass on this laptop (mps), median over 30 runs on 1280×800 page screenshots; it does not depend on the threshold.

## Screen test set

| model | threshold | test set | strategy | accuracy | precision | recall | F1 | missed | false alarms | FPR | ms/img |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.50 | full test set (520) | full | 0.608 | 1.000 | 0.019 | 0.038 | 204 / 208 | 0 / 312 | 0.000 | 51.0 |
| B | 0.50 | full test set (520) | full | 0.763 | 0.797 | 0.548 | 0.650 | 94 / 208 | 29 / 312 | 0.093 | 51.3 |
| B | 0.25 | full test set (520) | full | 0.777 | 0.728 | 0.707 | 0.717 | 61 / 208 | 55 / 312 | 0.176 | 51.3 |
| A | 0.50 | full test set (520) | tiles | 0.715 | 0.833 | 0.361 | 0.503 | 133 / 208 | 15 / 312 | 0.048 | 243.4 |
| B | 0.50 | full test set (520) | tiles | 0.839 | 0.756 | 0.880 | 0.813 | 25 / 208 | 59 / 312 | 0.189 | 227.7 |
| B | 0.25 | full test set (520) | tiles | 0.792 | 0.670 | 0.947 | 0.785 | 11 / 208 | 97 / 312 | 0.311 | 227.7 |
| A | 0.50 | clean subset | full | 0.752 | 1.000 | 0.016 | 0.032 | 60 / 61 | 0 / 181 | 0.000 | 51.0 |
| B | 0.50 | clean subset | full | 0.822 | 0.705 | 0.508 | 0.591 | 30 / 61 | 13 / 181 | 0.072 | 51.3 |
| B | 0.25 | clean subset | full | 0.802 | 0.600 | 0.639 | 0.619 | 22 / 61 | 26 / 181 | 0.144 | 51.3 |
| A | 0.50 | clean subset | tiles | 0.769 | 0.609 | 0.230 | 0.333 | 47 / 61 | 9 / 181 | 0.050 | 243.4 |
| B | 0.50 | clean subset | tiles | 0.839 | 0.628 | 0.885 | 0.735 | 7 / 61 | 32 / 181 | 0.177 | 227.7 |
| B | 0.25 | clean subset | tiles | 0.781 | 0.538 | 0.934 | 0.683 | 4 / 61 | 49 / 181 | 0.271 | 227.7 |

## Recall by video size (violent composites)

Size = pasted video width / page width.

| model | threshold | test set | strategy | under 30% | 30–45% | 45%+ |
|---|---|---|---|---|---|---|
| A | 0.50 | full test set (520) | full | 0.011 (1/87) | 0.030 (3/99) | 0.000 (0/22) |
| B | 0.50 | full test set (520) | full | 0.356 (31/87) | 0.667 (66/99) | 0.773 (17/22) |
| B | 0.25 | full test set (520) | full | 0.563 (49/87) | 0.788 (78/99) | 0.909 (20/22) |
| A | 0.50 | full test set (520) | tiles | 0.149 (13/87) | 0.475 (47/99) | 0.682 (15/22) |
| B | 0.50 | full test set (520) | tiles | 0.816 (71/87) | 0.929 (92/99) | 0.909 (20/22) |
| B | 0.25 | full test set (520) | tiles | 0.908 (79/87) | 0.990 (98/99) | 0.909 (20/22) |
| A | 0.50 | clean subset | full | 0.037 (1/27) | 0.000 (0/26) | 0.000 (0/8) |
| B | 0.50 | clean subset | full | 0.296 (8/27) | 0.654 (17/26) | 0.750 (6/8) |
| B | 0.25 | clean subset | full | 0.481 (13/27) | 0.731 (19/26) | 0.875 (7/8) |
| A | 0.50 | clean subset | tiles | 0.111 (3/27) | 0.269 (7/26) | 0.500 (4/8) |
| B | 0.50 | clean subset | tiles | 0.815 (22/27) | 0.962 (25/26) | 0.875 (7/8) |
| B | 0.25 | clean subset | tiles | 0.889 (24/27) | 1.000 (26/26) | 0.875 (7/8) |

## RLVS test split (held-out full frames, n = 1661)

| model | threshold | accuracy | precision | recall | F1 | missed | false alarms |
|---|---|---|---|---|---|---|---|
| A | 0.50 | 0.9910 | 0.9988 | 0.9840 | 0.9914 | 14 | 1 |
| B | 0.50 | 0.9928 | 1.0000 | 0.9863 | 0.9931 | 12 | 0 |
| B | 0.25 | 0.9952 | 0.9977 | 0.9932 | 0.9954 | 6 | 2 |

## Speed on this laptop

| model | device | full: median / mean / p90 ms | tiles: median / mean / p90 ms |
|---|---|---|---|
| A | mps | 51.0 / 52.1 / 54.9 | 243.4 / 242.5 / 251.3 |
| B | mps | 51.3 / 51.5 / 53.0 | 227.7 / 228.4 / 239.4 |

## Decision

Rule: choose B only if, **with tiles**, its recall is higher than A's on **both** the full set and the clean subset, its false-alarm rate is at most 2× A's (on both), and its RLVS test accuracy is ≥ 98%. Otherwise keep A.

| B threshold | recall > A (full) | recall > A (clean) | FPR ≤ 2×A (full) | FPR ≤ 2×A (clean) | RLVS acc ≥ 98% | passes |
|---|---|---|---|---|---|---|
| 0.50 | yes | yes | **no** | **no** | yes | **no** |
| 0.25 | yes | yes | **no** | **no** | yes | **no** |

**Result: keep A.**

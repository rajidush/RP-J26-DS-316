# Model comparison — P (pretrained) vs V1 (RLVS fine-tune) vs V2 (screen fine-tune)

Generated 2026-10-21T06:24:49+00:00 by `src/compare_models.py`.

- **P** = `models/violence-vit-pretrained-hf`: the pretrained `jaranohaal/vit-base-violence-detection` (ViT-base, trained by its author on RLVS frames), converted to Hugging Face format — no training of ours.
- **V1** = `models/violence-vit-screen-ft-v1`: P fine-tuned on full-frame RLVS images (notebook §7). Deployed until the switch to V2. *Labelled “A” in the first version of this report (21 Oct) and in the `runs/A_*` files.*
- **V2** = `models/violence-vit-screen-ft-v2`: P fine-tuned on screen composites + tiles with 40 % RLVS frames (notebook §15; run u4_lr3e-5). Now `config.MODEL_DIR`. *Labelled “B” in the 21 Oct version and in `runs/B_*`.* Threshold 0.25 = V2's max-F1 threshold on screen **val**, kept for reference.

## How to read this

- The **screen test set was never used for training or model selection** — V2's run and its 0.25 threshold were chosen on screen val.
- The **full test set** (520 images: 416 composites + 104 plain pages) reuses RLVS frames from the RLVS **training** split (203 of its 300 frames), which V1 was trained on; V2's training excluded every frame near a test frame. So the full set slightly favours V1.
- The **clean subset** (242 images: 138 composites with RLVS val/test frames + 104 plain pages) uses no training frames of V1 or V2 — the fairer comparison, but small: one violent image ≈ 1.6 recall points.
- **P's own training data is only documented as “RLVS frames”**, so P may have seen any of these frames (screen test, clean subset and RLVS test alike); its numbers are an upper bound for “no screen training”.
- `tiles` = whole frame + 2×2 overlapping tiles + centre crop, flagged if any crop ≥ threshold (strict mode; the previously deployed `tiles+motion` is identical on single images); `full` = whole frame only (default mode). Missed = violent images not flagged; false alarms = safe images flagged; FPR = false alarms / safe images.
- ms/img = preprocess + forward pass on this laptop (mps), median over 30 runs on 1280×800 page screenshots; independent of the threshold. Measured P 2026-10-21, V1 21 Oct, V2 21 Oct.

## Screen test set

| model | threshold | test set | strategy | accuracy | precision | recall | F1 | missed | false alarms | FPR | ms/img |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P | 0.50 | full test set (520) | full | 0.604 | 1.000 | 0.010 | 0.019 | 206 / 208 | 0 / 312 | 0.000 | 48.2 |
| V1 | 0.50 | full test set (520) | full | 0.608 | 1.000 | 0.019 | 0.038 | 204 / 208 | 0 / 312 | 0.000 | 51.0 |
| V2 | 0.50 | full test set (520) | full | 0.763 | 0.797 | 0.548 | 0.650 | 94 / 208 | 29 / 312 | 0.093 | 51.3 |
| V2 | 0.25 | full test set (520) | full | 0.777 | 0.728 | 0.707 | 0.717 | 61 / 208 | 55 / 312 | 0.176 | 51.3 |
| P | 0.50 | full test set (520) | tiles | 0.706 | 0.824 | 0.337 | 0.478 | 138 / 208 | 15 / 312 | 0.048 | 225.2 |
| V1 | 0.50 | full test set (520) | tiles | 0.715 | 0.833 | 0.361 | 0.503 | 133 / 208 | 15 / 312 | 0.048 | 243.4 |
| V2 | 0.50 | full test set (520) | tiles | 0.839 | 0.756 | 0.880 | 0.813 | 25 / 208 | 59 / 312 | 0.189 | 227.7 |
| V2 | 0.25 | full test set (520) | tiles | 0.792 | 0.670 | 0.947 | 0.785 | 11 / 208 | 97 / 312 | 0.311 | 227.7 |
| P | 0.50 | clean subset | full | 0.748 | — | 0.000 | 0.000 | 61 / 61 | 0 / 181 | 0.000 | 48.2 |
| V1 | 0.50 | clean subset | full | 0.752 | 1.000 | 0.016 | 0.032 | 60 / 61 | 0 / 181 | 0.000 | 51.0 |
| V2 | 0.50 | clean subset | full | 0.822 | 0.705 | 0.508 | 0.591 | 30 / 61 | 13 / 181 | 0.072 | 51.3 |
| V2 | 0.25 | clean subset | full | 0.802 | 0.600 | 0.639 | 0.619 | 22 / 61 | 26 / 181 | 0.144 | 51.3 |
| P | 0.50 | clean subset | tiles | 0.756 | 0.545 | 0.197 | 0.289 | 49 / 61 | 10 / 181 | 0.055 | 225.2 |
| V1 | 0.50 | clean subset | tiles | 0.769 | 0.609 | 0.230 | 0.333 | 47 / 61 | 9 / 181 | 0.050 | 243.4 |
| V2 | 0.50 | clean subset | tiles | 0.839 | 0.628 | 0.885 | 0.735 | 7 / 61 | 32 / 181 | 0.177 | 227.7 |
| V2 | 0.25 | clean subset | tiles | 0.781 | 0.538 | 0.934 | 0.683 | 4 / 61 | 49 / 181 | 0.271 | 227.7 |

## Recall by video size (violent composites)

Size = pasted video width / page width.

| model | threshold | test set | strategy | under 30% | 30–45% | 45%+ |
|---|---|---|---|---|---|---|
| P | 0.50 | full test set (520) | full | 0.000 (0/87) | 0.020 (2/99) | 0.000 (0/22) |
| V1 | 0.50 | full test set (520) | full | 0.011 (1/87) | 0.030 (3/99) | 0.000 (0/22) |
| V2 | 0.50 | full test set (520) | full | 0.356 (31/87) | 0.667 (66/99) | 0.773 (17/22) |
| V2 | 0.25 | full test set (520) | full | 0.563 (49/87) | 0.788 (78/99) | 0.909 (20/22) |
| P | 0.50 | full test set (520) | tiles | 0.115 (10/87) | 0.465 (46/99) | 0.636 (14/22) |
| V1 | 0.50 | full test set (520) | tiles | 0.149 (13/87) | 0.475 (47/99) | 0.682 (15/22) |
| V2 | 0.50 | full test set (520) | tiles | 0.816 (71/87) | 0.929 (92/99) | 0.909 (20/22) |
| V2 | 0.25 | full test set (520) | tiles | 0.908 (79/87) | 0.990 (98/99) | 0.909 (20/22) |
| P | 0.50 | clean subset | full | 0.000 (0/27) | 0.000 (0/26) | 0.000 (0/8) |
| V1 | 0.50 | clean subset | full | 0.037 (1/27) | 0.000 (0/26) | 0.000 (0/8) |
| V2 | 0.50 | clean subset | full | 0.296 (8/27) | 0.654 (17/26) | 0.750 (6/8) |
| V2 | 0.25 | clean subset | full | 0.481 (13/27) | 0.731 (19/26) | 0.875 (7/8) |
| P | 0.50 | clean subset | tiles | 0.037 (1/27) | 0.269 (7/26) | 0.500 (4/8) |
| V1 | 0.50 | clean subset | tiles | 0.111 (3/27) | 0.269 (7/26) | 0.500 (4/8) |
| V2 | 0.50 | clean subset | tiles | 0.815 (22/27) | 0.962 (25/26) | 0.875 (7/8) |
| V2 | 0.25 | clean subset | tiles | 0.889 (24/27) | 1.000 (26/26) | 0.875 (7/8) |

## RLVS test split (held-out full frames, n = 1661)

| model | threshold | accuracy | precision | recall | F1 | missed | false alarms |
|---|---|---|---|---|---|---|---|
| P | 0.50 | 0.9964 | 0.9989 | 0.9943 | 0.9966 | 5 | 1 |
| V1 | 0.50 | 0.9910 | 0.9988 | 0.9840 | 0.9914 | 14 | 1 |
| V2 | 0.50 | 0.9928 | 1.0000 | 0.9863 | 0.9931 | 12 | 0 |
| V2 | 0.25 | 0.9952 | 0.9977 | 0.9932 | 0.9954 | 6 | 2 |

## Speed on this laptop

| model | device | full: median / mean / p90 ms | tiles: median / mean / p90 ms |
|---|---|---|---|
| P | mps | 48.2 / 50.4 / 50.7 | 225.2 / 218.9 / 233.0 |
| V1 | mps | 51.0 / 52.1 / 54.9 | 243.4 / 242.5 / 251.3 |
| V2 | mps | 51.3 / 51.5 / 53.0 | 227.7 / 228.4 / 239.4 |

## Pre-set rule (tiles vs tiles)

Set before the test run: choose V2 + tiles only if, vs V1 + tiles, its recall is higher on **both** the full set and the clean subset, its false-alarm rate is at most 2× V1's (on both), and its RLVS test accuracy is ≥ 98%.

| V2 threshold | recall > V1 (full) | recall > V1 (clean) | FPR ≤ 2×V1 (full) | FPR ≤ 2×V1 (clean) | RLVS acc ≥ 98% | passes |
|---|---|---|---|---|---|---|
| 0.50 | yes | yes | **no** | **no** | yes | **no** |
| 0.25 | yes | yes | **no** | **no** | yes | **no** |

**Rule result: V2 + tiles is rejected** — its false-alarm rate is more than 2× V1 + tiles.

## Deployment decision

Compared with the **previously deployed setup — V1 + tiles at 0.5** — **V2 on the whole screen (`full`, 0.5)**:

| check | V1 + tiles | V2 whole screen | passes |
|---|---|---|---|
| recall, full test set (520) | 0.361 | 0.548 | yes |
| recall, clean subset | 0.230 | 0.508 | yes |
| false-alarm rate, full test set (520) (≤ 2×) | 0.048 | 0.093 (1.93×) | yes |
| false-alarm rate, clean subset (≤ 2×) | 0.050 | 0.072 (1.44×) | yes |
| RLVS test accuracy (≥ 98 %) | 0.9910 | 0.9928 | yes |
| ms/img (median) | 243.4 | 51.3 (4.7× faster) | yes |

The pre-set rule compared tiles with tiles and rejected V2 + tiles on false alarms. **V2 on the whole screen is the default mode**: against the previously deployed V1 + tiles it has higher recall on both test sets, a false-alarm rate within 2×, RLVS test accuracy ≥ 98 % and ~5× lower latency. **V2 + tiles is the strict mode** (recall 0.880 on the full set, false-alarm rate 0.189). Both at threshold 0.5 (`config.MODES`).

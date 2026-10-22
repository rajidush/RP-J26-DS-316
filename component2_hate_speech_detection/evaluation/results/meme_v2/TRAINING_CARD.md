# Training card: meme head v2 (trained on what C2 actually reads)

Run 2026-10-22 · branch `c2/finetune-pp1` · commit `8048efac46` · NVIDIA L40 · 4 min of training

## What was trained
- **Model:** frozen CLIP ViT-B/32 (`openai/clip-vit-base-patch32 (frozen)`) + Hate-CLIPper align-fusion head, 4,200,449 trainable parameters.
- **v1 recipe:** clean captions only, as the shipped head was trained. **v2:** clean + C2's OCR caption + the meme cut out of a built screenshot.
- **Data:** Hateful Memes, 8,000 train / 500 val / 500 test (dev_seen) memes; 24,000 training rows for v2.
- **Settings (v1's, unchanged):** `{"fusion": "align", "proj_dim": 1024, "pre_layers": 3, "hidden": 1024, "dropout": 0.4, "lr": 0.0001, "weight_decay": 0.0001, "grad_clip": 0.1, "batch_size": 64, "epochs": 20, "patience": 5, "class_weight": false, "rgcl": 0.0, "knn": 0}`; seeds [316, 317, 318, 319, 320].
- **Protocol, fixed before results:** validation chooses everything; test reported once per view; a meme the finder misses scores 0 on the screen view.

## Fine-tuning techniques used
| Technique | What it does here | Code (at the commit above) | Source |
|---|---|---|---|
| Transfer learning with a frozen backbone | CLIP ViT-B/32 turns each image and caption into 512-number features once; only the small head on top is trained | `meme_model/views.py` `embed_views()` | Radford et al., ICML 2021 (CLIP) |
| Hate-CLIPper align fusion | image and text features are projected and multiplied element-wise, so the head sees how picture and caption interact | `meme_model/heads.py` `FusionHead.fuse()` | Kumar & Nandakumar, NLP4PI @ EMNLP 2022 |
| Deployment-matched training views (data augmentation) | each training meme is also shown with C2's own OCR caption and cut out of a built screenshot, the inputs C2 really gets | `meme_model/views.py` `make_views()` | Kirk et al., WOAH 2021 (OCR text hurts meme models in the wild) |
| AdamW with weight decay 1e-4 | optimiser with decoupled weight decay, to limit overfitting | `meme_model/train.py` `train_one()` | Loshchilov & Hutter, ICLR 2019 |
| Dropout 0.4 | randomly drops units while training so the head can't memorise | `meme_model/heads.py` `FusionHead` | Srivastava et al., JMLR 2014 |
| Gradient clipping 0.1 | caps each update's size, keeping training stable | `meme_model/train.py` `train_one()` | Pascanu et al., ICML 2013 |
| Early stopping on validation AUROC (patience 5) | keeps the epoch that scored best on validation, never on test | `meme_model/train.py` `train_one()` | Prechelt 1998 (foundation) |
| Temperature scaling | rescales the scores on validation so 0.7 means about 70 % | `meme_model/train.py` `fit_temperature()` | Guo et al., ICML 2017 |
| Threshold chosen on validation | the decision cut-off is picked on validation memes only | `meme_model/train.py` `best_threshold()` | standard practice |
| Five-seed ensemble | five heads with different seeds; their scores are averaged, and mean ± sd shows the spread | `meme_model/train_views.py` `run_recipe()` | Lakshminarayanan et al., NeurIPS 2017 |
| Controlled baseline | the v1 recipe is retrained with identical settings and seeds, so only the training data differs | `meme_model/train_views.py` `RECIPES` | ablation design |
| Bootstrap 95 % confidence interval | 1,000 resamples of the test memes show how sure each AUROC is | `meme_model/train.py` `bootstrap_auroc()` | Efron & Tibshirani 1993 (foundation) |

## Results (test AUROC, 500 memes; ensemble of 5 seeds with 95 % CI)

| Test view | Shipped v1 head | v1 recipe, retrained | v2 (three views) |
|---|---|---|---|
| Clean caption (as in the dataset) | 0.752 | 0.762 (0.723–0.802); seeds 0.757 ± 0.005 | 0.753 (0.712–0.795); seeds 0.748 ± 0.011 |
| C2's OCR caption | 0.735 | 0.746 (0.703–0.789); seeds 0.741 ± 0.005 | 0.744 (0.699–0.787); seeds 0.740 ± 0.009 |
| Cut from a screenshot | 0.718 | 0.731 (0.689–0.778); seeds 0.727 ± 0.006 | 0.736 (0.695–0.783); seeds 0.731 ± 0.010 |

**Success rule** (v2 beats v1 on OCR and screen, loses ≤ 0.01 on clean): OCR -0.002, screen +0.006, clean -0.009 → **FAIL**.

## Where and how it ran
- **Machine:** NVIDIA L40 (CUDA 13.0), 12 vCPUs, Linux-6.17.0-35-generic-x86_64-with-glibc2.35, Python 3.14.6.
- **Packages:** torch 2.14.1, transformers 5.20.2, numpy 2.5.4, scikit-learn 1.9.1, pillow 13.0.0, rapidocr 3.10.0, onnxruntime 1.31.0, tokenizers 0.23.3, opencv-python-headless 5.0.0.93 (full list: `pip_freeze.txt`).
- **Code:** https://github.com/rajidush/RP-J26-DS-316.git at commit `8048efac46219fd3734a90bf25a04302d4f9c791`.
- **Commands:** `meme_model/thunder/run_meme_v2.sh`, which runs `meme_model.views` then `meme_model.train_views`.

## Data fingerprints
- `train.jsonl`: 8000 memes (2830 hateful), SHA-256 `e6367e6f41830a33…`
- `val.jsonl`: 500 memes (177 hateful), SHA-256 `13432cebf96502c8…`
- `test.jsonl`: 500 memes (246 hateful), SHA-256 `59b6f71745685ab9…`
- Images on the instance: 10000. Shipped v1 head SHA-256 `b172b2d7eb6210e1…`
- Clean-view features vs v1's stored features: image cosine 0.999996, text cosine 1.0 (1.0 = identical inputs).
- Picture finder found the meme on 97.2 % of test pages; OCR read no text on 0.2 % of test memes.

## Evidence files in this folder
- `env.json`, `pip_freeze.txt`, `nvidia_smi.txt`: machine and package versions.
- `views.log`, `train.log`: full console output, including every seed's epoch-by-epoch validation AUROC.
- `meme_v2_views.json`: every number above, per seed, with training histories.
- `views_stats.json`: OCR and picture-finder statistics per split.

Not committed (git-ignored): the images, captions, embeddings, predictions and the trained weights (`models/meme/meme_clip-b32_align_v2views.pt`). Hateful Memes is licensed for research only.

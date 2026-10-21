# Component 1 — experiment results

One folder per experiment, in the order they were run. Every number below is
copied from the file named in the last column. Colab files (sections 1–15 of
`notebooks/finetune_violence.ipynb`) were downloaded from
`MyDrive/pp1_component1/outputs/` (sections 1–14) and `…/results/` (section 15);
the rest were produced locally by the scripts in `src/` (paths in `src/config.py`,
`RESULTS_*`).

**Models.** *Pretrained* = `jaranohaal/vit-base-violence-detection` (ViT-base,
already fine-tuned on RLVS frames). **ft-v1** = `models/violence-vit-screen-ft-v1`,
the pretrained model fine-tuned on full-frame RLVS images (01; `config.MODEL_DIR`).
**ft-v2** = `models/violence-vit-screen-ft-v2`, the pretrained model fine-tuned on
screen composites + tiles (04). RLVS splits: 7751 train / 1661 val / 1661 test
frames (`01_rlvs_finetune/splits.csv`, seed 42).

| folder | experiment | date | model | data | headline result | main file |
|---|---|---|---|---|---|---|
| [`00_week1_baseline/`](00_week1_baseline/) | Week 1 sanity check of the pretrained checkpoint (`src/baseline_eval.py`) | 2026-10-06 | pretrained (timm `vit_base_patch16_224`) | 60 local images in `data/val/` (30 violence / 30 non-violence) | accuracy 0.9833 (59/60; violence 29/30, non-violence 30/30) | `baseline_accuracy.json` |
| [`01_rlvs_finetune/`](01_rlvs_finetune/) | First fine-tune on RLVS frames (notebook §1–12) | 2026-10-07 | pretrained → **ft-v1** (last 2 blocks, lr 3e-5, 5 epochs) | RLVS, 7751 / 1661 / 1661 | RLVS test: baseline acc 0.996 / F1 0.997 → ft-v1 acc 0.991 / F1 0.991 (−0.5 pp); local inference 47.6 ms median (M1, mps) | `before_after.md` |
| [`02_param_sweep/`](02_param_sweep/) | Parameter sweep (§13) + threshold sweep (§14) | 2026-10-08 | 6 fresh fine-tunes of the pretrained model, 3 epochs each | RLVS val (selection) / test (scored once) | best on val: head_only (val F1 0.9937) → test F1 0.9954, vs baseline 0.9966; thresholds for val recall ≥ 0.99: baseline 0.85, ft-v1 0.45 | `sweep_results.md` |
| [`03_screen_eval_pretrained/`](03_screen_eval_pretrained/) | Screen-style test set, whole screen vs region tiles (`src/eval_screens.py`) | 2026-10-14 | **ft-v1** (the file's `model_dir`) | screen test: 416 composites + 104 pages | full frame recall 0.019 / F1 0.038; tiles recall 0.361 / F1 0.503, 15 false alarms / 312 safe; inference 46.0 ms full, 202.5 ms tiles (median, live frames) | `screen_eval.md` |
| [`04_screen_finetune/`](04_screen_finetune/) | Screen fine-tuning (§15), sweep of 3 runs; leakage check of the training set | 2026-10-20 | pretrained → **ft-v2** (u4_lr3e-5: last 4 blocks, lr 3e-5, 3 epochs) | 6181 screen train images + RLVS train frames (40 % of each epoch); screen val 1078 | screen val @0.5: P 0.895 / R 0.693 / F1 0.781, RLVS val acc 0.9922; max-F1 threshold 0.25 (F1 0.816); leakage check: all 15 checks pass | `screen_finetune_sweep.md` |
| [`05_model_comparison/`](05_model_comparison/) | ft-v1 (A) vs ft-v2 (B) on the screen test set + RLVS test (`src/compare_models.py`) | 2026-10-21 | A = ft-v1 @0.5; B = ft-v2 @0.5 and @0.25 | screen test (520) + clean subset (242); RLVS test 1661 | tiles @0.5: recall A 0.361 → B 0.880 (full), 0.230 → 0.885 (clean); FPR A 0.048 → B 0.189 (full); RLVS test acc A 0.991, B 0.993. **Decision: keep A** (B's false-alarm rate > 2× A's) | `model_comparison.md` |
| [`function2/`](function2/) | Function 2 (response overlay): end-to-end latency, test output | — | — | — | empty on this branch — `e2e_latency.json` and `pytest_output.txt` live on the Function 2 branch | — |

## What is in each folder

- **00_week1_baseline** — `baseline_accuracy.json`.
- **01_rlvs_finetune** — `splits.csv` (§2), `baseline_metrics.json` (§4),
  `training_curves.png` (§7), `finetuned_metrics.json` (§9), `before_after.csv/.md`,
  `confusion_matrices.png`, `misclassified_test.png` (§10), `training_info.json` (§12;
  identical to the copy inside `models/violence-vit-screen-ft-v1/`), `latency.json`
  (local `src/detector.py --benchmark 20`). The 20 Oct re-run of §3–14 on Colab
  reproduced the same metrics; this folder keeps the 7 Oct files that match ft-v1.
- **02_param_sweep** — `sweep_results.csv/.md/.png`, `sweep_best_on_test.csv` (§13);
  `threshold_sweep_val.csv`, `threshold_sweep.png`, `threshold_chosen_test.csv` (§14).
- **03_screen_eval_pretrained** — `screen_eval.json/.md`, `screen_eval_predictions.csv`,
  `screen_eval_examples.png` (violent examples blurred), `latency_regions.json`
  (`src/run_monitor.py --benchmark 20`). The folder name is historical: the model
  evaluated was ft-v1, `config.MODEL_DIR` at the time.
- **04_screen_finetune** — `screen_finetune_info.json`, `screen_finetune_sweep.csv/.md/.png`,
  `screen_finetune_thresholds.csv/.png` (§15e–g); `screen_train_leakage.json`
  (`src/check_leakage.py`). The screen test set was not used.
- **05_model_comparison** — `model_comparison.md/.json`; `runs/` holds the six
  `eval_screens.py` runs it reads (`<model>_t<threshold>_<full|clean>.*`).
- **function2** — reserved for Function 2 results.

## Caveats

- The screen test set's 300 RLVS frames come from all three RLVS splits (203 train,
  48 val, 49 test), so ft-v1 trained on most of them; the **clean subset** in 05 keeps
  only composites whose frames are RLVS val/test.
- The screen test set was never used for training or model/threshold selection.

## Adding results

New runs go into an experiment folder, never the `results/` root:
`src/eval_screens.py` writes to `03_screen_eval_pretrained/` (default model) or
`05_model_comparison/runs/` (with `--model-dir`) and refuses to overwrite existing
files without `--overwrite`. Start a new numbered folder for a new experiment and
add a row above.

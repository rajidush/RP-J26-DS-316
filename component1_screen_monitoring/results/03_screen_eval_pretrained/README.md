# 03 · Screen-style evaluation — model V1 (not the pretrained model)

**Despite the folder name, this evaluation used V1**, `models/violence-vit-screen-ft-v1`
— the pretrained `jaranohaal/vit-base-violence-detection` fine-tuned on full-frame
RLVS images (experiment 01). It was `config.MODEL_DIR` at the time, and
`screen_eval.json` records it as `model_dir`. The folder keeps its original name so
existing links still work.

| File | What |
|---|---|
| `screen_eval.md` / `.json` | V1 on the screen test set (416 composites + 104 pages), threshold 0.5, whole screen (`full`) vs region tiles (`tiles`), 2026-10-14 |
| `screen_eval_predictions.csv` | one row per image × strategy |
| `screen_eval_examples.png` | 8 misclassified examples (violent ones blurred) |
| `latency_regions.json` | V1 latency per region strategy on live frames (`src/run_monitor.py --benchmark 20`, 2026-10-07) |

The true pretrained model (P) was first evaluated on this test set in
[`../05_model_comparison/`](../05_model_comparison/model_comparison.md), next to V1
and V2.

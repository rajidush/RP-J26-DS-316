# Component 3 evidence

Every recorded run is one dated folder, never overwritten. Run folders are grouped by what they show. The scripts that produce them stay at the top of this folder, so every `python -m evidence.<script>` command is unchanged.

```
evidence/
├── intercept/   INTERCEPT: model samples, constrained decoding, adversarial runs, token masking
├── inquire/     INQUIRE fine-tuning: drafts, dataset, training, blind comparison, served GGUFs
├── evaluate/    EVALUATE: emotional-state baseline and rule evaluation
├── system/      whole pipeline: end-to-end runs, latency, schema validation, readability
├── research/    dated research notes (some are local-only, see .gitignore)
└── *.py         the scripts that write the runs
```

## Rules

- **One folder per run, named `<what>_<date>[_<tag>]`, inside its group.** Scripts refuse to overwrite an existing run, so pass `--tag` for a second run on the same day.
- **Recorded files are byte-identical.** `.gitattributes` turns off line-ending conversion here, because manifests and run records hold file hashes. Never edit a recorded file. Add a new dated file next to it instead, as the human-review score files do.
- **Weights never go here.** `*.gguf`, `*.safetensors` and `models/` are gitignored.

## Where each script writes

| Script | Writes to |
|---|---|
| `capture_intercept_samples`, `run_adversarial_intercept`, `check_temperature`, `trace_token_masking` | `intercept/` |
| `capture_inquire_served`, `inquire_finetune.py build` / `sheet` (one folder up) | `inquire/` |
| `evaluate_emotion_rules` | prints to the terminal; save its output under `evaluate/` |
| `run_end_to_end`, `measure_latency`, `validate_schemas`, `readability_check` | `system/` |
| `score_human_review --run intercept/<run>` | into the reviewed run's own folder |

## Contents

**`intercept/`**

| Run | Shows |
|---|---|
| `intercept_samples_2026-10-06` | The "before" baseline: 16 unconstrained model lines, 0/16 matching the INTERCEPT grammar |
| `intercept_samples_constrained_2026-10-11` | The same capture under constrained decoding (ADR 0004) |
| `adversarial_intercept_2026-10-11`, `_after-fix` | The first adversarial run, before and after the description-invite post-check rule |
| `adversarial_intercept_2026-10-13` | 104 attacks + 8 controls at temperature 1.0, and the human-review scores (19 and 21 Oct) |
| `temperature_check_2026-10-13` | The llama.cpp log confirming the sent temperature (server default 0.8 before 12 Oct) |
| `token_masking_2026-10-18` | A step-by-step trace of token masking on the INTERCEPT grammar |

**`inquire/`**

| Run | Shows |
|---|---|
| `inquire_drafts_2026-10-08` | 150 drafted INQUIRE rows, two educator drafts each |
| `inquire_seed_inputs` | Seed overrides and labels that `build` reads |
| `inquire_dataset_2026-10-10` | The scenario-grouped train/validation/test split (104/25/50) and its manifest |
| `inquire_training_2026-10-10` | The LoRA training run record and executed notebook |
| `inquire_comparison_2026-10-10` | The blind control-vs-adapted comparison: sheet, key, scores, summary |
| `gguf_conversion_2026-10-17` | Merging the adapter and converting it to GGUF |
| `inquire_served_2026-10-17` | The held-out prompts run on the served GGUFs in LM Studio |

**`evaluate/`**

| Run | Shows |
|---|---|
| `eva001_baseline_2026-10-09` | The EVALUATE prompt baseline: prompt, raw outputs, review |
| `emotion_rules_eval_2026-10-14` | Phrase-rule agreement on dev and held-out children, before and after each fix |

**`system/`**

| Run | Shows |
|---|---|
| `end_to_end_2026-10-12_{online,offline}` | The first end-to-end runs through Components 1/2 → 3 → 4 |
| `end_to_end_2026-10-14_{online,offline}[_after-review]` | The same after the 14 Oct bug fixes, and after review |
| `latency_2026-10-13_before-fix`, `_after-fix` | Per-state latency before and after the `127.0.0.1` fix |
| `schema_validation_2026-10-15_after-review` | Every recorded trigger, record and delivered line re-validated (761/763) |
| `readability_2026-10-20` | Readability of child-facing lines for the youngest band |

## Moved on 25 Oct 2026

Until 25 Oct every run sat directly in `evidence/`. Each one moved unchanged into its group: `evidence/<run>/` is now `evidence/<group>/<run>/`, and the research notes are now in `evidence/research/`. Live code, tests and docs point to the new paths. **Recorded files inside the runs, executed notebooks, old tickets and the dated five-week plan still cite the old flat paths, and are deliberately left as recorded.** Read `evidence/<run>` in them as `evidence/<group>/<run>`.

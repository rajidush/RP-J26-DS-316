# Component 3 — Socratic Educator

**Owner:** IT23155466
**Status:** PP1 scope (2 functions) implemented and tested; real SLM wiring is a post-PP1 stretch goal (see TODO in `src/grammar_decoder.py`)

## What this component does

Receives a risk trigger from Component 1 (screen monitoring) or Component 2
(hate-speech detection), runs a 4-state Socratic dialogue with the child
(Intercept → Inquire → Evaluate → Contract), and hands a structured,
privacy-safe record to Component 4.

## The two PP1 functions

| Function | File | What to demo |
|---|---|---|
| Function 1: FSM Controller | `src/fsm_controller.py` | Run a full session and show the state trace in `SessionTranscript` |
| Function 2: Grammar-Constrained Decoding | `src/grammar_decoder.py` | Run `run_adversarial_batch(...)` and show a 0% violation rate |

## Input / Output contract

- Input: `TriggerPayload` — see `docs/interface-contracts/comp1_to_comp3.schema.json` and `comp2_to_comp3.schema.json`
- Output: `EvaluateOutput` — see `docs/interface-contracts/comp3_to_comp4.schema.json`

Both are validated against the shared JSON Schemas at runtime (`schemas.py`) —
if a teammate changes a contract without updating the schema file, this
component will fail loudly instead of silently accepting bad data.

## Run it standalone

```bash
cd component3_socratic_educator
pip install -r requirements.txt
python -m src.demo   # interactive: type child responses at the prompt
```

## Run the tests

```bash
pytest component3_socratic_educator/tests -v
```

## Generating the Component 4 dataset

`generate_educator_dataset.py` simulates child sessions, runs them through the
real `FSMController`, validates every record against
`docs/interface-contracts/comp3_to_comp4.schema.json` (the same file
`EvaluateOutput.validate()` uses), and writes `out/educator_sessions.jsonl`,
`out/educator_sessions.csv`, `out/seed_vs_output.csv`, `out/scenario_seeds.csv`
and `out/child_replies.jsonl`. Run everything from `component3_socratic_educator/`.

```bash
pip install -r requirements.txt -r requirements-datagen.txt

# Smoke test: canned child replies, no model anywhere
python generate_educator_dataset.py --offline --stub-generator --children 2 --limit 10
pytest tests/test_dataset_generation.py

# Full dataset: child replies from LM Studio (localhost:1234) through distilabel
python generate_educator_dataset.py --backend lmstudio --stub-generator
# Alternative backend
python generate_educator_dataset.py --backend ollama --model gemma3:1b --stub-generator

# Chunked generation (safe for laptops / preventing overheating):
python generate_educator_dataset.py --backend lmstudio --stub-generator --start-child 1 --children 10 --out out
# (pause/let laptop cool down)
python generate_educator_dataset.py --backend lmstudio --stub-generator --start-child 11 --children 10 --append --out out
python generate_educator_dataset.py --backend lmstudio --stub-generator --start-child 21 --children 10 --append --out out
python generate_educator_dataset.py --backend lmstudio --stub-generator --start-child 31 --children 10 --append --out out
```

- `--start-child` & `--append`: Generates a specific cohort of children and appends
  to existing files without repeating headers. Random seed alignment is globally
  preserved so chunked runs match a single unified run exactly.
- `--batch-size`: Adjusts distilabel batch size (default 16; lower to 4 or 8 to reduce peak thermal/CPU load).
- `--offline` replaces only the child replies (canned templates, no distilabel).
  On its own it is **not** network-free, because the INTERCEPT state still calls LM Studio.
- `--stub-generator` swaps only the INTERCEPT opening text: it uses the template
  pool instead of the live model. No record field depends on that text today.
- All child replies are generated before any session runs, so distilabel and the
  FSM never call the model server at the same time.
- `child_id` is added to the exported rows only. The contract record has just `session_id`.
- Until `FSMController._evaluate()` is replaced, `emotional_state` can only be
  `calm`, `defensive` or `unclear` (never `distressed`), so this output is a
  pipeline test batch, not final data.

## Known current limitations (say this openly at PP1, don't hide it)

- `grammar_decoder.py` generates from a small validated candidate pool,
  not yet the real on-device SLM — the *grammar enforcement* is real and
  tested, the *generation* behind it is the next step.
- The EVALUATE heuristic in `fsm_controller.py` is keyword/confidence-based,
  a placeholder for a proper classifier once more dialogue data exists.
- Tested against ~20 adversarial cases for PP1, not yet the proposal's
  target of 200+.

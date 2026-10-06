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

## Function 1: FSM Controller (`src/fsm_controller.py`)

`FSMController().run(trigger, respond)` runs one whole session and returns
`(EvaluateOutput, SessionTranscript)`. `respond` is any `str -> str` callable
(the CLI `input()` in the demo, a scripted list in tests), so the FSM never
depends on a live input source. Diagram: `docs/architecture/fsm-state-diagram.md`.

| State | What happens | Exit condition |
|---|---|---|
| (entry) | `trigger.validate()` against the comp1/comp2 contract | Invalid trigger raises before the child is asked anything |
| INTERCEPT | One opening question from the generator. Never names the flagged category. | Always -> INQUIRE after one reply |
| INQUIRE | Follow-up questions, a different one each attempt (no repeats in a session) | Reply has >= `MIN_RESPONSE_WORDS_FOR_COMPLETENESS` (4) words, or `MAX_INQUIRE_ATTEMPTS` (3) reached |
| EVALUATE | Builds the typed record (no child-facing text) | Always -> CONTRACT |
| CONTRACT | Sets `escalation_flag`, says the closing line | Always -> DONE; record is schema-validated and returned |

**EVALUATE rules (transparent rule-based baseline, to be replaced by a trained classifier):**

- `risk_level` comes from the detector's confidence, not from the dialogue:
  `>= 0.85` high, `>= 0.50` moderate, otherwise low.
- `emotional_state` comes from `src/emotion_rules.py`, which matches phrase lists over
  every child reply in the session. Its four signals are:
  - *reflective*, e.g. "maybe I should talk to my mom"
  - *distress*, e.g. "so scared", "alone", "heavy"; softened forms like "kinda scary" don't count
  - *defensive*, e.g. "not my fault", "I didn't do it"
  - *withdrawn*, e.g. "I don't know", "quiet", "leave me alone"

  The result is calm if planning is at least as strong as distress. Otherwise the strongest of
  distressed > defensive > unclear wins. No signal plus only 1-2 word replies gives unclear
  (a bare "no" is not counted as defensive). `classify()` also returns the phrases that
  fired, so every label can be explained. On the simulated dataset it agrees with the
  intended child profile in 89.8% of development sessions (C001-C020, used to write the
  rules) and **85.2% of held-out sessions** (C021-C040). The word-count rule it replaced
  scored 30.5%. Reproduce with `python -m evidence.evaluate_emotion_rules --split heldout`.
- `self_regulation_shown` is True if any child reply contains, as whole words,
  `instead`, `next time`, `should have` or `i could` (so "I couldn't" does not count).

**CONTRACT rule:** `escalation_flag = risk_level in ESCALATION_RISK_LEVELS`
(currently `{"high"}`). An escalated session closes with *"I'm going to let a
parent know we talked about this, so they can support you."*; otherwise
*"Thanks for talking this through with me."*

`dialogue_turns` counts turns that have a child reply (INTERCEPT + INQUIRE), so
it ranges from 2 to 4 with the current config. All tunables are in `src/config.py`.

**Tests.** `tests/test_synthetic_dialogues.py` holds 18 scripted dialogues
(normal flow, minimal/silent/defensive children, keyword edge cases, risk
boundaries), checks each one's outcome, and checks the invariants every session
must satisfy: state order, no category leak, no repeated question, schema-valid
output. `tests/conftest.py` blocks the live model by default, so the unit tests are
deterministic and work offline. Tests marked `live_model` use LM Studio and
skip when it is not running:

```bash
pytest tests -v                 # everything (live_model tests skip if LM Studio is down)
pytest tests -v -m live_model   # only the live-model checks
```

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
- `--rescore`: re-runs the FSM over the seeds and replies already saved in `--out`,
  with no model calls, and rewrites the session outputs (it also removes duplicate seed rows).
  Use it after changing FSM/evaluation logic instead of regenerating the LLM replies,
  which would produce a different dataset. Example: `python generate_educator_dataset.py --rescore --out out`
- `--offline` replaces only the child replies (canned templates, no distilabel).
  On its own it is **not** network-free, because the INTERCEPT state still calls LM Studio.
- `--stub-generator` swaps only the INTERCEPT opening text: it uses the template
  pool instead of the live model. No record field depends on that text today.
- All child replies are generated before any session runs, so distilabel and the
  FSM never call the model server at the same time.
- `child_id` is added to the exported rows only. The contract record has just `session_id`.
- `out/DATASET_CARD.md` describes the handed-over dataset for Component 4: files, columns,
  how it was made, and its limitations.

## Known current limitations (say this openly at PP1, don't hide it)

- Since 5 Oct, INTERCEPT asks the live model (LM Studio) with **no constraint
  applied yet**: its output is returned as-is, without `validate()`. Samples in
  `evidence/intercept_samples_2026-10-06.jsonl` show 0/16 match the INTERCEPT
  grammar (none named the flagged category). Constraining it is the 6 Oct task.
  If LM Studio is unreachable or times out (30 s), INTERCEPT falls back to the
  template pool.

- `grammar_decoder.py` generates from a small validated candidate pool,
  not yet the real on-device SLM — the *grammar enforcement* is real and
  tested, the *generation* behind it is the next step.
- EVALUATE is rule-based (phrase lists + detector confidence), not a trained
  classifier. Its 85.2% held-out agreement is measured on LLM-simulated children
  against the profile they were *generated* with, not on real children or
  human-annotated labels. Weakest case: withdrawn children who voice sadness are
  labelled `distressed` (65% agreement for `unclear`).
- Tested against ~20 adversarial cases for PP1, not yet the proposal's
  target of 200+.

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

## Known current limitations (say this openly at PP1, don't hide it)

- `grammar_decoder.py` generates from a small validated candidate pool,
  not yet the real on-device SLM — the *grammar enforcement* is real and
  tested, the *generation* behind it is the next step.
- The EVALUATE heuristic in `fsm_controller.py` is keyword/confidence-based,
  a placeholder for a proper classifier once more dialogue data exists.
- Tested against ~20 adversarial cases for PP1, not yet the proposal's
  target of 200+.

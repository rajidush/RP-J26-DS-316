# Integration

This folder is where the 4 components stop being independent and become
one running system. Keep it empty of real logic — it should only ever
*import and call* each component's `src/`, never contain detection or
dialogue logic itself.

## What "integrated" means here

There is no server, no message queue, no separate processes per
component. `end_to_end_pipeline.py` is a single Python script that:

1. Imports each component's module directly
2. Calls them in sequence, passing plain dicts shaped by the shared
   schemas in `docs/interface-contracts/`
3. Prints the result

That's the entire "deployment" for a laptop demo — `python
integration/end_to_end_pipeline.py` *is* the running software.

## PP1 demo (panel proof)

Non-interactive, offline-safe proof that C1 and C2 both wake C3 and that
C3's record reaches C4. No live model, no GPU. Use this for the viva.

```bash
# One command — checklist + evidence JSON
python integration/prove_integration.py

# Optional live walkthrough (scripted replies, offline INTERCEPT)
python integration/end_to_end_pipeline.py --offline --auto --source component1
python integration/end_to_end_pipeline.py --offline --auto --source component2

# Automated tests
pytest integration/integration_tests -v
```

Evidence file written for the panel:

`integration/evidence/pp1_integration_proof.json`

**What to tell the panel**

1. One Python process wires all four components in-process.
2. Handoffs use shared JSON schemas under `docs/interface-contracts/`.
3. C1 (screen) or C2 (text) emits a trigger → C3 runs the Socratic FSM →
   C4 builds the parent-facing summary (and emotional-pattern cluster when
   `component4_profiling_xai/models/` artifacts are present).
4. C2 text triggers open with *message* wording, not “on your screen”.
5. `prove_integration.py` + green pytest are the proof artifacts.

**PP1-safe flags:** `--auto` implies offline (no 30s LM Studio wait) unless
you pass `--with-model`. C2 `--text` uses lexicon-only heads by default so
the demo does not download model weights.

## Run it (interactive)

```bash
pip install -r component3_socratic_educator/requirements.txt
python integration/end_to_end_pipeline.py --offline
```

Component 2 can trigger the session with its real engine on a message
(without `--text` it sends its mock fixture):

```bash
python integration/end_to_end_pipeline.py --offline --auto --source component2 --text "nobody likes you, just leave the group" --age 10
```

## When to touch this folder

Only when wiring components together against the shared schemas. Keep
deep detection / dialogue logic inside each component's `src/`.

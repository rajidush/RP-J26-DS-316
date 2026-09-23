# Integration

This folder is where the 4 components stop being independent and become
one running system. Keep it empty of real logic — it should only ever
*import and call* each component's `src/`, never contain detection or
dialogue logic itself.

## What "integrated" means here

There is no server, no message queue, no separate processes per
component. `end_to_end_pipeline.py` is a single Python script that:

1. Imports each component's module directly (currently the stubs; swap
   for real implementations as each owner finishes)
2. Calls them in sequence, passing plain dicts shaped by the shared
   schemas in `docs/interface-contracts/`
3. Prints the result

That's the entire "deployment" for a laptop demo — `python
integration/end_to_end_pipeline.py` *is* the running software.

## Run it

```bash
pip install -r component3_socratic_educator/requirements.txt
python integration/end_to_end_pipeline.py
```

## Run the automated (non-interactive) version

```bash
pytest integration/integration_tests -v
```

## When to touch this folder

Only once at least two real (non-stub) components exist. Until then, keep
building and testing each component in isolation against its own
`mock_inputs/` — that's the whole point of the interface-contract
approach.

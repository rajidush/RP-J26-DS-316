# Component 1 — Screen Monitoring (STUB)

**Owner:** IT23377844
**Status:** Interface stub only — replace `src/component1.py` with the real capture + classification pipeline.

## Contract this component must satisfy

Output must validate against `docs/interface-contracts/comp1_to_comp3.schema.json`.
The stub in `src/component1.py` already produces a schema-valid payload from
`mock_inputs/sample_trigger.json` — swap the body of `simulate_screen_capture()`
for the real model call, keep the return shape identical, and Component 3
requires no changes.

## Run its test

```bash
pytest component1_screen_monitoring/tests -v
```

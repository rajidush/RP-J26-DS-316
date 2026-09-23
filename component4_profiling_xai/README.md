# Component 4 — Behavioral Profiling / XAI Reporting (STUB)

**Owner:** IT23135116
**Status:** Interface stub only — formats Component 3's output; real profiling/XAI layer not yet built.

## Contract this component must satisfy

Input must validate against `docs/interface-contracts/comp3_to_comp4.schema.json`.
`src/component4.py` currently just formats that dict into a parent-facing
summary — the real version would persist sessions over time and add an
explainability layer (why this session was flagged the way it was).

## Run its test

```bash
pytest component4_profiling_xai/tests -v
```

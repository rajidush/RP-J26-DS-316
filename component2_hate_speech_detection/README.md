# Component 2 — Hate-Speech / Cyberbullying Detection (STUB)

**Owner:** IT23209152
**Status:** Interface stub only — replace `src/component2.py` with the real NLP classifier.

## Contract this component must satisfy

Output must validate against `docs/interface-contracts/comp2_to_comp3.schema.json`.
Swap the body of `analyze_text()` for the real model call; keep the return
shape identical so Component 3 requires no changes.

## Run its test

```bash
pytest component2_hate_speech_detection/tests -v
```

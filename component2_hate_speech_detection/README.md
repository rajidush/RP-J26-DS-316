# Component 2 — Hate-Speech / Cyberbullying Detection

**Owner:** IT23209152
**Status:** Step 1 done — stub + schema validation. Real NLP classifier is next.

## Contract this component must satisfy

Output must validate against `docs/interface-contracts/comp2_to_comp3.schema.json`.
`analyze_text()` already validates against that schema before returning.
Swap the mock load for a real model call later; keep the return shape
identical so Component 3 requires no changes.

## Run its test

```bash
pip install -r component2_hate_speech_detection/requirements.txt
pytest component2_hate_speech_detection/tests -v
```

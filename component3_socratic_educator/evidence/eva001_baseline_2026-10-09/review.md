# EVA001 baseline check (2026-10-09)

**This is a development check, not test evidence.** EVA001 belongs to scenario S026, and every EVALUATE seed comes from that one scenario (see ADR 0001). The check sent the existing inference baseline the exact EVA001 prompt, with the expected answer kept out of the prompt. It is the "pending EVA001 baseline" listed in `skills/model-fine-tuning-context.md`.

| File | Contents |
|---|---|
| `prompt.txt` | The EVA001 prompt, copied unchanged from `data/educator_seed.jsonl` |
| `raw_outputs.jsonl` | 3 runs: raw text, finish reason, latency and the full API response |
| `settings.json` | Model, checkpoint, runtime, endpoint and generation settings ("not recorded" where unknown) |

## Setup
- **Model:** `google/gemma-3-1b` through LM Studio 0.4.19.0, checkpoint `gemma-3-1B-it-QAT-Q4_0.gguf`, context 8192 tokens.
- **Request:** a single user message with no system message, temperature 0.8 (the recorded baseline temperature) and max_tokens 60.
- **Not recorded:** top_p, top_k and seed. They were not sent, so LM Studio used its defaults.
- **Runs:** 3, all kept.

## Expected answer (kept outside the prompt)
`{"emotional_state":"calm","self_regulation_shown":true}`

## Results, each measure reported separately

| Measure | Run 0 | Run 1 | Run 2 | Total |
|---|---|---|---|---|
| Raw output is valid JSON as returned | no | no | no | **0 / 3** |
| Valid JSON once the Markdown code fence is removed | yes | yes | yes | 3 / 3 |
| Schema: exactly 2 keys, allowed value, `self_regulation_shown` is a real boolean | yes | yes | yes | 3 / 3 |
| `emotional_state` = calm | yes | yes | yes | 3 / 3 |
| `self_regulation_shown` = true | yes | yes | yes | 3 / 3 |
| Forbidden extra content (child-facing message, explanation, risk_level) | none | none | none | 0 / 3 |
| Finish reason | stop | stop | stop | — |
| Latency (s) | 2.63 | 2.42 | 2.63 | median 2.63 |

All three outputs were identical:

````
```json
{
  "emotional_state": "calm",
  "self_regulation_shown": true
}
```
````

## Review
- **Classification is correct in all 3 runs.** The model labelled the child's explicit "I feel calm now" as calm. It counted "I'll mute the chat and take a break" as self-regulation, which matches the prompt's rule that a stated intention counts.
- **The output format breaks the prompt's instruction in all 3 runs.** The prompt says "Return only one JSON object". The model wrapped the object in a Markdown ```` ```json ```` fence, so a strict parser (`json.loads`) rejects every raw output. If this model's output were wired into EVALUATE, it would need fence stripping or constrained decoding before parsing.
- **Sampling at temperature 0.8 did not vary the output** across 3 runs on this prompt (30 completion tokens each).

## Limitations
- **One example from a development scenario.** This says nothing about generalisation to held-out scenarios.
- **An easy, positive case.** The child states the emotion explicitly and states a safer plan. Harder cases are untested: refusal (which must not be labelled defensive), "I don't know", conflicting statements, and history handling.
- **Small sample.** 3 runs is enough to show the format habit, not to estimate a rate.
- **Generation settings not sent** (top_p, top_k, seed) are marked "not recorded".

## Next action
None blocking. EVALUATE stays rule-based (`emotion_rules.classify()`, ADR 0001), so this result is recorded as baseline history only. If a Gemma EVALUATE classifier is benchmarked after PP1, reuse this check and score raw JSON validity separately from correctness, as done here.

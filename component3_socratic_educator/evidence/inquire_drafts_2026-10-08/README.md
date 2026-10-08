# INQUIRE drafts — 8 Oct 2026 (unreviewed)

Raw output of `draft_inquire_dataset.py` for the Component 3 fine-tuning pilot. **Nothing here is a seed example yet**: every row has `review.status = "pending"` until the owner picks a draft or rewrites it.

| File | Contents |
|---|---|
| `scenarios.json` | 30 new scenarios (S027–S056) over 8 risk categories, and the 10 test scenarios drawn with seed 20261008 before any reply existed: S028, S029, S031, S033, S034, S035, S036, S041, S049, S056 |
| `scenarios_raw.json` | Raw model output for the scenario stage |
| `replies.json` | Per scenario: previous educator message + one child reply per reply type |
| `inquire_drafts.jsonl` | 150 rows (INQ030–INQ179), 2 educator drafts each; 50 rows are `split: "test"` |

## Drafting models (recorded per row in `provenance.drafts`)

| Rows | Model | Why |
|---|---|---|
| INQ030–INQ097 (68) | `google/gemini-3.1-pro-preview` via Colab built-in models | Planned drafting model |
| INQ098–INQ179 (82) | `claude-opus-5-5` in a Claude Code session, same rules and few-shots (`inquire-draft-v1`) | Colab quota (403, cost reservation) and AI Studio free tier (20 requests/day) exhausted on 8 Oct |

Scenarios and child replies for all rows came from Gemini 3.1 Pro preview.

## Automatic checks on all 300 drafts (8 Oct)

| Check | Gemini 3.1 Pro (136 drafts) | Claude Opus 5.5 (164 drafts) |
|---|---|---|
| Format check (1–2 sentences, one terminal `?`, ≤ 30 words, no forbidden substring) | 136 / 136 | 164 / 164 |
| Names an emotion the child did not use | 0 | 0 |
| Mentions a trusted adult / parent / teacher | 66 (49%) | 118 (72%) |
| Identical draft pairs | 5 | 0 |
| Mean words | 20.1 | 19.7 |

The trusted-adult difference is partly scenario mix (the Claude rows are mostly cyberbullying, grooming and self-harm scenarios, where the rules require a trusted adult) and partly drafter style; check it during review so the adapted model does not learn to redirect every reply to an adult.

Limitations: the emotion check uses a fixed word list, so it can miss paraphrased emotions; test-set coverage omits cyberbullying and self_harm_language by chance of the seeded draw.

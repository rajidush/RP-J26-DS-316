# Component 3 - INTERCEPT Generation Path (two layers + fallback line)

How the opening line a child sees is produced (`src/grammar_decoder.py`). Every path ends in a line that met both layers, or in a pre-checked fallback line.

```mermaid
flowchart TD
    T["Trigger arrives<br/>nothing from it enters the prompt"] --> P["INTERCEPT prompt<br/>no risk category and no upstream text in it"]
    P --> M{"Local model reachable?<br/>LM Studio, gemma-3-1b"}
    M -- "no: model_unreachable" --> F
    M -- "yes" --> G["Layer 1: constrained decoding<br/>JSON schema compiled to a grammar,<br/>disallowed tokens masked while generating"]
    G --> J{"Reply parses and<br/>matches the grammar?"}
    J -- "no: invalid_json or grammar" --> F
    J -- "yes" --> C{"Layer 2: post-check<br/>word and phrase rules"}
    C -- "forbidden_word, alert_language,<br/>category_named or invites_description" --> F
    C -- "passes" --> D["Model line delivered to the child"]
    F["Fallback line<br/>pre-checked template line,<br/>picked by the trigger's risk_category"] --> D2["Fallback line delivered to the child"]
```

**What each layer guarantees**
- **Layer 1, constrained decoding** (structure): the opener is one of two fixed safe openers; the question is one plain sentence (letters, spaces, commas, straight apostrophes) of at most 100 characters ending in `?`. The model cannot produce anything else.
- **Layer 2, post-check** (words a grammar cannot forbid): forbidden words; words that reveal an alert (`alert`, `flagged`, `warning`, ...); any risk-category word; questions inviting the child to describe the content ("What did you see?").
- **Fallback line**: a template line that passes both layers, used for any failure, including no model.

**Limitation:** the layers guarantee structure and banned words, not meaning. A reworded invitation can pass both (12 Oct end-to-end online run: "What's on your screen right now?" was blocked, "What's happening on your screen right now?" passed). The 13 Oct unseen run also delivered "What did it look like?", "Why did you open it?" and "What were you working on when it stopped?".

**Evidence** (`component3_socratic_educator/evidence/`, one folder per run)

| Run (date run) | Files | Result |
|---|---|---|
| 16 samples before/after (6 Oct before, 11 Oct after) | `intercept_samples_2026-10-06/`, `intercept_samples_constrained_2026-10-11/` | grammar match 0/16 → 16/16; alert leaks 4/16 → 0/16 |
| 24 adversarial attacks (11 Oct) | `adversarial_intercept_2026-10-11/`, `adversarial_intercept_2026-10-11_after-fix/` | 0/24 rule violations delivered; 5 blocked by the post-check after the fix |
| 104 attacks + 8 no-attack controls, seen + unseen, temperature 1.0 (13 Oct) | `adversarial_intercept_2026-10-13/` | 0/112 post-check violations delivered (by construction); 9 blocked; measurement-only audit flagged 20/112 (seen 4/24, unseen 16/88), owner review pending |
| End-to-end, online (12 Oct) | `end_to_end_2026-10-12_online/` | 7/8 model lines delivered, 1 blocked live (`invites_description`) |
| End-to-end, offline (12 Oct) | `end_to_end_2026-10-12_offline/` | 8/8 fallback lines (`model_unreachable`), sessions complete |

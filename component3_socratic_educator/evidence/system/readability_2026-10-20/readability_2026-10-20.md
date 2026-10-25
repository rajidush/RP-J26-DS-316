# Reading level of child-facing lines

Measured 2026-10-20T13:53:10 by `evidence/readability_check.py` (textstat 0.7.13). Measurement only: nothing in the dialogue changed. Each set is scored as one passage, because a one-sentence line is too short for these formulas. US grade + 5 is roughly the age, so grade 3 is about age 8.

| Set | Lines (unique) | Words | Flesch–Kincaid grade | Spache grade | Dale–Chall score |
|---|---|---|---|---|---|
| INTERCEPT, model (13 Oct adversarial) | 103 (68) | 1512 | 1.9 | 3.0 | 6.2 (above 4.9) |
| INTERCEPT, model (14 Oct online end-to-end) | 8 (5) | 114 | 2.2 | 3.1 | 6.2 (above 4.9) |
| INTERCEPT fallback pool | 2 (2) | 31 | 1.5 | 3.0 | 6.1 (above 4.9) |
| fixed openers | 2 (2) | 17 | 2.3 | 3.0 | 5.9 (above 4.9) |
| INQUIRE templates | 9 (9) | 115 | 1.0 | 2.4 | 5.4 (above 4.9) |
| CONTRACT closing lines (14 Oct runs) | 3 (3) | 40 | 2.2 | 3.1 | 6.9 (above 4.9) |

Dale–Chall: 4.9 or lower is easily read by an average 4th-grade pupil (about age 9–10); 5.0–5.9 is grades 5–6, 6.0–6.9 grades 7–8.

**Read Dale–Chall with care.** textstat looks each word up in the familiar-word list as written, so it counts regular -ing, -ed and -er forms of familiar words ("doing", "noticed", "closer") and "okay" as unfamiliar. The published method counts those as familiar, so these scores are too high. The words it counted:

- INTERCEPT, model (13 Oct adversarial) (14% of words): noticed ×74, closer ×33, doing ×29, focusing ×14, working ×12, thinking ×9, looking ×6, happened ×6
- INTERCEPT, model (14 Oct online end-to-end) (14% of words): focusing ×6, closer ×5, noticed ×3, focused ×1, wandering ×1
- INTERCEPT fallback pool (13% of words): thinking ×1, noticed ×1, feeling ×1, closer ×1
- fixed openers (12% of words): noticed ×1, closer ×1
- INQUIRE templates (6% of words): okay ×2, safer ×1, doing ×1, pressure ×1, alright ×1, telling ×1
- CONTRACT closing lines (14 Oct runs) (10% of words): support ×1, talking ×1, okay ×1, later ×1

## Hardest lines per set (per-line Flesch–Kincaid, for finding outliers only)

**INTERCEPT, model (13 Oct adversarial)**
- 6.0: I noticed something on your screen just now. Were you thinking about anything before opening it?
- 5.1: I noticed something on your screen just now. Were you working on anything specific right before that?
- 4.5: Something you just saw might need a closer look. What were you imagining or focusing on?

**INTERCEPT, model (14 Oct online end-to-end)**
- 2.4: I noticed something on your screen just now. What were you focusing on?
- 2.3: Something you just saw might need a closer look. What were you focusing on?
- 2.3: Something you just saw might need a closer look. What was your mind wandering on right now?

**INTERCEPT fallback pool**
- 1.5: I noticed something on your screen just now. Can you tell me what you were thinking?
- 1.5: Something you just saw might need a closer look. How are you feeling right now?

**fixed openers**
- 2.3: Something you just saw might need a closer look.
- 2.3: I noticed something on your screen just now.

**INQUIRE templates**
- 3.7: No pressure. Is there one small thing you remember?
- 3.7: I understand. What would you like to happen now?
- 1.6: That's alright. What's one word for how you feel right now?

**CONTRACT closing lines (14 Oct runs)**
- 5.4: I'm going to let a parent know we talked about this, so they can support you.
- 0.9: That's okay, you don't have to talk right now. I'm here if you want to talk later.
- 0.6: Thanks for talking this through with me.

## Limitations

- The formulas measure word and sentence length and word familiarity, not meaning: a short question can still be hard to answer, for example a hypothetical one for an 8-year-old.
- Readability formulas were built for US school texts in passages, not single spoken questions.
- Component 3 receives no age today, so this checks the lines against the youngest band only.

## Inputs

- `adversarial_intercept_2026-10-13\adversarial_intercept_2026-10-13.jsonl`: sha256 `8bc3c58007f10d52c428542fb982f90ecf37abb242af9d0dd156c87e2d6c4176`
- `end_to_end_2026-10-14_online_after-review\end_to_end_2026-10-14_online_after-review.jsonl`: sha256 `e1fc9447248b7d0f93b6ab4b6ed893a61eb2745bdcff27113ff50c1add0a47f7`
- `end_to_end_2026-10-14_offline_after-review\end_to_end_2026-10-14_offline_after-review.jsonl`: sha256 `70f04423d90dfa97dd6d75c7cee457e8af456b1dceee41dc1cca4aa85d8010f9`

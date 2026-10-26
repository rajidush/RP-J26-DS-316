# Reading level of child-facing lines

Measured 2026-10-26T12:48:28 by `evidence/readability_check.py` (textstat 0.7.13). Measurement only: nothing in the dialogue changed. Each set is scored as one passage, because a one-sentence line is too short for these formulas. US grade + 5 is roughly the age, so grade 3 is about age 8.

| Set | Lines (unique) | Words | Flesch–Kincaid grade | Spache grade | Dale–Chall score |
|---|---|---|---|---|---|
| INTERCEPT, model (13 Oct adversarial) | 103 (68) | 1512 | 1.9 | 3.0 | 6.2 (above 4.9) |
| INTERCEPT, model (14 Oct online end-to-end) | 8 (5) | 114 | 2.2 | 3.1 | 6.2 (above 4.9) |
| INTERCEPT fallback pool | 2 (2) | 31 | 1.5 | 3.0 | 6.1 (above 4.9) |
| fixed openers | 2 (2) | 17 | 2.3 | 3.0 | 5.9 (above 4.9) |
| INQUIRE templates, ages 8-10 | 6 (6) | 66 | 2.1 | 3.3 | 6.1 (above 4.9) |
| INQUIRE templates, ages 11-13 | 6 (6) | 57 | 2.6 | 2.9 | 5.8 (above 4.9) |
| INQUIRE templates, ages 14-15 | 6 (6) | 70 | 2.0 | 2.7 | 5.3 (above 4.9) |
| CONTRACT closing lines (14 Oct runs) | 3 (3) | 40 | 2.2 | 3.1 | 6.9 (above 4.9) |

Dale–Chall: 4.9 or lower is easily read by an average 4th-grade pupil (about age 9–10); 5.0–5.9 is grades 5–6, 6.0–6.9 grades 7–8.

**Read Dale–Chall with care.** textstat looks each word up in the familiar-word list as written, so it counts regular -ing, -ed and -er forms of familiar words ("doing", "noticed", "closer") and "okay" as unfamiliar. The published method counts those as familiar, so these scores are too high. The words it counted:

- INTERCEPT, model (13 Oct adversarial) (14% of words): noticed ×74, closer ×33, doing ×29, focusing ×14, working ×12, thinking ×9, looking ×6, happened ×6
- INTERCEPT, model (14 Oct online end-to-end) (14% of words): focusing ×6, closer ×5, noticed ×3, focused ×1, wandering ×1
- INTERCEPT fallback pool (13% of words): thinking ×1, noticed ×1, feeling ×1, closer ×1
- fixed openers (12% of words): noticed ×1, closer ×1
- INQUIRE templates, ages 8-10 (11% of words): okay ×3, grownup ×2, doing ×1, using ×1
- INQUIRE templates, ages 11-13 (9% of words): pressure ×3, doing ×1, happens ×1
- INQUIRE templates, ages 14-15 (6% of words): later ×2, affect ×1, options ×1
- CONTRACT closing lines (14 Oct runs) (10% of words): support ×1, talking ×1, later ×1, okay ×1

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

**INQUIRE templates, ages 8-10**
- 4.5: Next time, would you like to tell a grown-up you trust or close it and do something else?
- 2.9: That's okay. Next time, would you tell a grown-up or close it?
- 2.5: Can you tell me what you were doing just before?

**INQUIRE templates, ages 11-13**
- 4.8: No pressure. What's one thing you remember from just before?
- 2.6: What do you think might happen if it comes up again?
- 2.5: No pressure. What might happen next?

**INQUIRE templates, ages 14-15**
- 3.6: What are your options if this comes up again, and which one fits you best?
- 2.6: Can you walk me through what was going on just before?
- 2.5: How do you think this could affect you later on?

**CONTRACT closing lines (14 Oct runs)**
- 5.4: I'm going to let a parent know we talked about this, so they can support you.
- 0.9: That's okay, you don't have to talk right now. I'm here if you want to talk later.
- 0.6: Thanks for talking this through with me.

## Limitations

- The formulas measure word and sentence length and word familiarity, not meaning: a short question can still be hard to answer, for example a hypothetical one for an 8-year-old.
- Readability formulas were built for US school texts in passages, not single spoken questions.
- Only the INQUIRE templates are worded per age band; the other sets are read against the youngest.
- Each band's INQUIRE set is six short lines (three moves, each with its easier form), a small passage for these formulas.

## Inputs

- `intercept\adversarial_intercept_2026-10-13\adversarial_intercept_2026-10-13.jsonl`: sha256 `8bc3c58007f10d52c428542fb982f90ecf37abb242af9d0dd156c87e2d6c4176`
- `system\end_to_end_2026-10-14_online_after-review\end_to_end_2026-10-14_online_after-review.jsonl`: sha256 `e1fc9447248b7d0f93b6ab4b6ed893a61eb2745bdcff27113ff50c1add0a47f7`
- `system\end_to_end_2026-10-14_offline_after-review\end_to_end_2026-10-14_offline_after-review.jsonl`: sha256 `70f04423d90dfa97dd6d75c7cee457e8af456b1dceee41dc1cca4aa85d8010f9`

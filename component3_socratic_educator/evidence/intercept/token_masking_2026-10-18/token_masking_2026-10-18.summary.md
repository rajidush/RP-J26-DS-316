# Finite-automaton token masking, 2026-10-18

Model: `gemma-3-1B-it-QAT-Q4_0.gguf` (the INTERCEPT inference baseline's weights, loaded in transformers on CPU). Greedy decoding, same prompt for both runs. The child-facing path is unchanged (ADR 0004).

| Run | Line | Controller grammar + post-check |
|---|---|---|
| unmasked | 'Okay, let’s talk about what you were doing on your screen just now.' | grammar |
| masked | 'I noticed something on your screen just now. Would you like to tell me a little about what you were doing?' | pass |

The model's first choice was blocked at 8 of 26 steps.

| Step | Text so far | Model's top choices before masking (✗ = blocked) | Chosen | Allowed tokens | Blocked p |
|---|---|---|---|---|---|
| 0 | "" | ✗ "Okay" 0.74, ✗ "It" 0.12, ✗ "That" 0.07, ✗ "“" 0.02, "I" 0.02 | "I" | 8 | 0.984 |
| 1 | "I" | ✗ " understand" 0.75, " notice" 0.12, " noticed" 0.04, ✗ "’" 0.04, ✗ " see" 0.04 | " notice" | 7 | 0.841 |
| 2 | "I notice" | ✗ " you" 0.56, ✗ " your" 0.36, ✗ " something" 0.06, ✗ " a" 0.01, ✗ " that" 0.01 | "d" | 2 | 1.000 |
| 3 | "I noticed" | ✗ " you" 0.76, ✗ " that" 0.09, ✗ " your" 0.07, " something" 0.03, ✗ " a" 0.02 | " something" | 9 | 0.969 |
| 4 | "I noticed something" | " on" 0.42, ✗ "’" 0.15, ✗ " is" 0.08, ✗ " has" 0.04, ✗ " stopped" 0.04 | " on" | 4 | 0.576 |
| 5 | "I noticed something on" | " your" 1.00, ✗ " the" 0.00, " you" 0.00, ✗ " that" 0.00, ✗ " our" 0.00 | " your" | 6 | 0.001 |
| 6 | "I noticed something on your" | " screen" 0.96, ✗ " device" 0.04, ✗ " phone" 0.00, ✗ " tablet" 0.00, ✗ " devices" 0.00 | " screen" | 7 | 0.041 |
| 7 | "I noticed something on your screen" | ✗ "," 0.38, ✗ " –" 0.23, ✗ " recently" 0.10, " just" 0.10, ✗ " stopped" 0.04 | " just" | 6 | 0.899 |
| 8 | "I noticed something on your screen just" | " now" 0.83, ✗ " stopped" 0.10, ✗ " froze" 0.06, ✗ "…" 0.00, ✗ " paused" 0.00 | " now" | 5 | 0.167 |
| 9 | "I noticed something on your screen just now" | ✗ " –" 0.67, ✗ "," 0.21, "." 0.06, ✗ ";" 0.04, ✗ "—" 0.02 | "." | 2 | 0.944 |
| 10 | "I noticed something on your screen just now." | ✗ "\n\n" 0.19, " " 0.17, " Tell" 0.17, " How" 0.15, " What" 0.12 | " " | 30849 | 0.257 |
| 11 | "I noticed something on your screen just now. " | ✗ "\n\n" 1.00, ✗ "\n\n\n" 0.00, ✗ "\n" 0.00, ✗ "\n\n\n\n" 0.00, ✗ (special) 0.00 | "Would" | 18095 | 1.000 |
| 12 | "I noticed something on your screen just now. Would" | " you" 1.00, " it" 0.00, " that" 0.00, " anyone" 0.00, " the" 0.00 | " you" | 139646 | 0.000 |
| 13 | "I noticed something on your screen just now. Would you" | " like" 0.98, " mind" 0.01, " be" 0.01, " feel" 0.00, " prefer" 0.00 | " like" | 139648 | 0.000 |
| 14 | "I noticed something on your screen just now. Would you like" | " to" 1.00, " a" 0.00, " me" 0.00, " share" 0.00, " it" 0.00 | " to" | 139648 | 0.000 |
| 15 | "I noticed something on your screen just now. Would you like to" | " tell" 0.95, " talk" 0.04, " share" 0.01, " start" 0.00, " think" 0.00 | " tell" | 139648 | 0.000 |
| 16 | "I noticed something on your screen just now. Would you like to tell" | " me" 1.00, " a" 0.00, " us" 0.00, " your" 0.00, "me" 0.00 | " me" | 139648 | 0.000 |
| 17 | "I noticed something on your screen just now. Would you like to tell me" | " a" 0.49, " about" 0.34, " what" 0.16, " how" 0.00, " more" 0.00 | " a" | 139648 | 0.000 |
| 18 | "I noticed something on your screen just now. Would you like to tell me a" | " little" 0.85, " bit" 0.15, " tiny" 0.00, " few" 0.00, " quick" 0.00 | " little" | 139648 | 0.000 |
| 19 | "I noticed something on your screen just now. Would you like to tell me a little" | " about" 0.92, " bit" 0.07, " more" 0.00, "?" 0.00, " what" 0.00 | " about" | 139648 | 0.000 |
| 20 | "I noticed something on your screen just now. Would you like to tell me a little about" | " what" 0.89, " it" 0.11, " how" 0.00, " that" 0.00, " where" 0.00 | " what" | 139648 | 0.000 |
| 21 | "I noticed something on your screen just now. Would you like to tell me a little about what" | " you" 0.99, " was" 0.01, ✗ "’" 0.00, "'" 0.00, " happened" 0.00 | " you" | 139648 | 0.004 |
| 22 | "I noticed something on your screen just now. Would you like to tell me a little about what you" | " were" 0.99, ✗ "’" 0.01, "'" 0.00, " had" 0.00, " might" 0.00 | " were" | 139648 | 0.008 |
| 23 | "I noticed something on your screen just now. Would you like to tell me a little about what you were" | " doing" 0.75, " focusing" 0.11, " occupied" 0.04, " working" 0.03, " up" 0.03 | " doing" | 139648 | 0.000 |
| 24 | "I noticed something on your screen just now. Would you like to tell me a little about what you were doing" | "?" 0.84, " before" 0.11, " or" 0.03, " when" 0.01, " while" 0.00 | "?" | 139648 | 0.000 |
| 25 | "I noticed something on your screen just now. Would you like to tell me a little about what you were doing?" | (special) 1.00, ✗ " " 0.00, ✗ "\n" 0.00, ✗ " 😊" 0.00, ✗ "\n\n" 0.00 | "(end of generation)" | 2 | 0.003 |

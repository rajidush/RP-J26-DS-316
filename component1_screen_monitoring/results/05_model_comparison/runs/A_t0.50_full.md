# Screen-style evaluation — whole screen vs. region inference

Model: `component1_screen_monitoring/models/violence-vit-screen-ft-v1` · threshold 0.5 · device mps · 2026-10-21T03:09:29+00:00

Violence = positive class. `tiles` = whole frame + 2×2 overlapping tiles + centre crop (identical to `tiles+motion` on single images). Precision / recall / F1 are — when undefined (e.g. the scraped pages contain no violent images).

| Source | Strategy | n (viol/safe) | Accuracy | Precision | Recall | F1 | Missed | False alarms | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| scraped | full | 104 (0/104) | 1.000 | — | — | — | 0 | 0 | 52.5 |
| scraped | tiles | 104 (0/104) | 0.962 | 0.000 | — | — | 0 | 4 | 182.2 |
| composite | full | 416 (208/208) | 0.510 | 1.000 | 0.019 | 0.038 | 204 | 0 | 56.7 |
| composite | tiles | 416 (208/208) | 0.654 | 0.872 | 0.361 | 0.510 | 133 | 11 | 194.1 |
| overall | full | 520 (208/312) | 0.608 | 1.000 | 0.019 | 0.038 | 204 | 0 | 55.9 |
| overall | tiles | 520 (208/312) | 0.715 | 0.833 | 0.361 | 0.503 | 133 | 15 | 191.7 |

Counts per source: scraped: 104, composite: 416, recorded: 0

## Mean inference time per image

| Model | full | tiles |
|---|---|---|
| `component1_screen_monitoring/models/violence-vit-screen-ft-v1` | 55.9 ms | 191.7 ms |

Missed = violent images not flagged (false negatives). False alarms = safe images flagged (false positives).

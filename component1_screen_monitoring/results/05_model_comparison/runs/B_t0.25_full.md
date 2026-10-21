# Screen-style evaluation — whole screen vs. region inference

Model: `component1_screen_monitoring/models/violence-vit-screen-ft-v2` · threshold 0.25 · device mps · 2026-10-21T03:17:02+00:00

Violence = positive class. `tiles` = whole frame + 2×2 overlapping tiles + centre crop (identical to `tiles+motion` on single images). Precision / recall / F1 are — when undefined (e.g. the scraped pages contain no violent images).

| Source | Strategy | n (viol/safe) | Accuracy | Precision | Recall | F1 | Missed | False alarms | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| scraped | full | 104 (0/104) | 0.942 | 0.000 | — | — | 0 | 6 | 54.1 |
| scraped | tiles | 104 (0/104) | 0.827 | 0.000 | — | — | 0 | 18 | 190.7 |
| composite | full | 416 (208/208) | 0.736 | 0.750 | 0.707 | 0.728 | 61 | 49 | 53.0 |
| composite | tiles | 416 (208/208) | 0.784 | 0.714 | 0.947 | 0.814 | 11 | 79 | 192.2 |
| overall | full | 520 (208/312) | 0.777 | 0.728 | 0.707 | 0.717 | 61 | 55 | 53.2 |
| overall | tiles | 520 (208/312) | 0.792 | 0.670 | 0.947 | 0.785 | 11 | 97 | 191.9 |

Counts per source: scraped: 104, composite: 416, recorded: 0

## Mean inference time per image

| Model | full | tiles |
|---|---|---|
| `component1_screen_monitoring/models/violence-vit-screen-ft-v2` | 53.2 ms | 191.9 ms |

Missed = violent images not flagged (false negatives). False alarms = safe images flagged (false positives).

# Screen-style evaluation — whole screen vs. region inference

Model: `component1_screen_monitoring/models/violence-vit-pretrained-hf` · threshold 0.5 · device mps · 2026-10-21T06:22:26+00:00

Violence = positive class. `tiles` = whole frame + 2×2 overlapping tiles + centre crop (identical to `tiles+motion` on single images). Precision / recall / F1 are — when undefined (e.g. the scraped pages contain no violent images).

| Source | Strategy | n (viol/safe) | Accuracy | Precision | Recall | F1 | Missed | False alarms | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| scraped | full | 104 (0/104) | 1.000 | — | — | — | 0 | 0 | 51.4 |
| scraped | tiles | 104 (0/104) | 0.971 | 0.000 | — | — | 0 | 3 | 185.1 |
| composite | full | 416 (208/208) | 0.505 | 1.000 | 0.010 | 0.019 | 206 | 0 | 51.8 |
| composite | tiles | 416 (208/208) | 0.639 | 0.854 | 0.337 | 0.483 | 138 | 12 | 190.6 |
| overall | full | 520 (208/312) | 0.604 | 1.000 | 0.010 | 0.019 | 206 | 0 | 51.7 |
| overall | tiles | 520 (208/312) | 0.706 | 0.824 | 0.337 | 0.478 | 138 | 15 | 189.5 |

Counts per source: scraped: 104, composite: 416, recorded: 0

## Mean inference time per image

| Model | full | tiles |
|---|---|---|
| `component1_screen_monitoring/models/violence-vit-pretrained-hf` | 51.7 ms | 189.5 ms |

Missed = violent images not flagged (false negatives). False alarms = safe images flagged (false positives).

# Screen-style evaluation — whole screen vs. region inference

Model: `component1_screen_monitoring/models/violence-vit-screen-ft-v2` · threshold 0.5 · device mps · 2026-10-21T03:13:18+00:00

Violence = positive class. `tiles` = whole frame + 2×2 overlapping tiles + centre crop (identical to `tiles+motion` on single images). Precision / recall / F1 are — when undefined (e.g. the scraped pages contain no violent images).

| Source | Strategy | n (viol/safe) | Accuracy | Precision | Recall | F1 | Missed | False alarms | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| scraped | full | 104 (0/104) | 0.971 | 0.000 | — | — | 0 | 3 | 55.5 |
| scraped | tiles | 104 (0/104) | 0.894 | 0.000 | — | — | 0 | 11 | 203.7 |
| composite | full | 416 (208/208) | 0.712 | 0.814 | 0.548 | 0.655 | 94 | 26 | 55.5 |
| composite | tiles | 416 (208/208) | 0.825 | 0.792 | 0.880 | 0.834 | 25 | 48 | 196.0 |
| overall | full | 520 (208/312) | 0.763 | 0.797 | 0.548 | 0.650 | 94 | 29 | 55.5 |
| overall | tiles | 520 (208/312) | 0.839 | 0.756 | 0.880 | 0.813 | 25 | 59 | 197.5 |

Counts per source: scraped: 104, composite: 416, recorded: 0

## Mean inference time per image

| Model | full | tiles |
|---|---|---|
| `component1_screen_monitoring/models/violence-vit-screen-ft-v2` | 55.5 ms | 197.5 ms |

Missed = violent images not flagged (false negatives). False alarms = safe images flagged (false positives).

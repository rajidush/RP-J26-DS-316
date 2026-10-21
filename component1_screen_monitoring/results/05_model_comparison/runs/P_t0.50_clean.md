# Screen-style evaluation — whole screen vs. region inference

**Subset: clean** — only composites whose RLVS frame is from the RLVS val/test split (never trained on), plus the plain scraped pages.

Model: `component1_screen_monitoring/models/violence-vit-pretrained-hf` · threshold 0.5 · device mps · 2026-10-21T06:23:36+00:00

Violence = positive class. `tiles` = whole frame + 2×2 overlapping tiles + centre crop (identical to `tiles+motion` on single images). Precision / recall / F1 are — when undefined (e.g. the scraped pages contain no violent images).

| Source | Strategy | n (viol/safe) | Accuracy | Precision | Recall | F1 | Missed | False alarms | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| scraped | full | 104 (0/104) | 1.000 | — | — | — | 0 | 0 | 51.2 |
| scraped | tiles | 104 (0/104) | 0.971 | 0.000 | — | — | 0 | 3 | 187.4 |
| composite | full | 138 (61/77) | 0.558 | — | 0.000 | 0.000 | 61 | 0 | 51.2 |
| composite | tiles | 138 (61/77) | 0.594 | 0.632 | 0.197 | 0.300 | 49 | 7 | 187.1 |
| overall | full | 242 (61/181) | 0.748 | — | 0.000 | 0.000 | 61 | 0 | 51.2 |
| overall | tiles | 242 (61/181) | 0.756 | 0.545 | 0.197 | 0.289 | 49 | 10 | 187.2 |

Counts per source: scraped: 104, composite: 138, recorded: 0

## Mean inference time per image

| Model | full | tiles |
|---|---|---|
| `component1_screen_monitoring/models/violence-vit-pretrained-hf` | 51.2 ms | 187.2 ms |

Missed = violent images not flagged (false negatives). False alarms = safe images flagged (false positives).

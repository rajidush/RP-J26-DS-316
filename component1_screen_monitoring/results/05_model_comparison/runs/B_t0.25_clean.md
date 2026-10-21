# Screen-style evaluation — whole screen vs. region inference

**Subset: clean** — only composites whose RLVS frame is from the RLVS val/test split (never trained on), plus the plain scraped pages.

Model: `component1_screen_monitoring/models/violence-vit-screen-ft-v2` · threshold 0.25 · device mps · 2026-10-21T03:18:22+00:00

Violence = positive class. `tiles` = whole frame + 2×2 overlapping tiles + centre crop (identical to `tiles+motion` on single images). Precision / recall / F1 are — when undefined (e.g. the scraped pages contain no violent images).

| Source | Strategy | n (viol/safe) | Accuracy | Precision | Recall | F1 | Missed | False alarms | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| scraped | full | 104 (0/104) | 0.942 | 0.000 | — | — | 0 | 6 | 51.6 |
| scraped | tiles | 104 (0/104) | 0.827 | 0.000 | — | — | 0 | 18 | 189.5 |
| composite | full | 138 (61/77) | 0.696 | 0.661 | 0.639 | 0.650 | 22 | 20 | 66.6 |
| composite | tiles | 138 (61/77) | 0.746 | 0.648 | 0.934 | 0.765 | 4 | 31 | 234.0 |
| overall | full | 242 (61/181) | 0.802 | 0.600 | 0.639 | 0.619 | 22 | 26 | 60.2 |
| overall | tiles | 242 (61/181) | 0.781 | 0.538 | 0.934 | 0.683 | 4 | 49 | 214.9 |

Counts per source: scraped: 104, composite: 138, recorded: 0

## Mean inference time per image

| Model | full | tiles |
|---|---|---|
| `component1_screen_monitoring/models/violence-vit-screen-ft-v2` | 60.2 ms | 214.9 ms |

Missed = violent images not flagged (false negatives). False alarms = safe images flagged (false positives).

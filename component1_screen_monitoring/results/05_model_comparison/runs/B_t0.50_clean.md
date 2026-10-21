# Screen-style evaluation — whole screen vs. region inference

**Subset: clean** — only composites whose RLVS frame is from the RLVS val/test split (never trained on), plus the plain scraped pages.

Model: `component1_screen_monitoring/models/violence-vit-screen-ft-v2` · threshold 0.5 · device mps · 2026-10-21T03:14:35+00:00

Violence = positive class. `tiles` = whole frame + 2×2 overlapping tiles + centre crop (identical to `tiles+motion` on single images). Precision / recall / F1 are — when undefined (e.g. the scraped pages contain no violent images).

| Source | Strategy | n (viol/safe) | Accuracy | Precision | Recall | F1 | Missed | False alarms | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| scraped | full | 104 (0/104) | 0.971 | 0.000 | — | — | 0 | 3 | 56.9 |
| scraped | tiles | 104 (0/104) | 0.894 | 0.000 | — | — | 0 | 11 | 201.8 |
| composite | full | 138 (61/77) | 0.710 | 0.756 | 0.508 | 0.608 | 30 | 10 | 54.6 |
| composite | tiles | 138 (61/77) | 0.797 | 0.720 | 0.885 | 0.794 | 7 | 21 | 197.6 |
| overall | full | 242 (61/181) | 0.822 | 0.705 | 0.508 | 0.591 | 30 | 13 | 55.6 |
| overall | tiles | 242 (61/181) | 0.839 | 0.628 | 0.885 | 0.735 | 7 | 32 | 199.4 |

Counts per source: scraped: 104, composite: 138, recorded: 0

## Mean inference time per image

| Model | full | tiles |
|---|---|---|
| `component1_screen_monitoring/models/violence-vit-screen-ft-v2` | 55.6 ms | 199.4 ms |

Missed = violent images not flagged (false negatives). False alarms = safe images flagged (false positives).

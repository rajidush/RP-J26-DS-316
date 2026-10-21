# Screen-style evaluation — whole screen vs. region inference

**Subset: clean** — only composites whose RLVS frame is from the RLVS val/test split (never trained on), plus the plain scraped pages.

Model: `component1_screen_monitoring/models/violence-vit-screen-ft-v1` · threshold 0.5 · device mps · 2026-10-21T03:10:46+00:00

Violence = positive class. `tiles` = whole frame + 2×2 overlapping tiles + centre crop (identical to `tiles+motion` on single images). Precision / recall / F1 are — when undefined (e.g. the scraped pages contain no violent images).

| Source | Strategy | n (viol/safe) | Accuracy | Precision | Recall | F1 | Missed | False alarms | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| scraped | full | 104 (0/104) | 1.000 | — | — | — | 0 | 0 | 56.1 |
| scraped | tiles | 104 (0/104) | 0.962 | 0.000 | — | — | 0 | 4 | 193.4 |
| composite | full | 138 (61/77) | 0.565 | 1.000 | 0.016 | 0.032 | 60 | 0 | 59.2 |
| composite | tiles | 138 (61/77) | 0.623 | 0.737 | 0.230 | 0.350 | 47 | 5 | 195.6 |
| overall | full | 242 (61/181) | 0.752 | 1.000 | 0.016 | 0.032 | 60 | 0 | 57.9 |
| overall | tiles | 242 (61/181) | 0.769 | 0.609 | 0.230 | 0.333 | 47 | 9 | 194.7 |

Counts per source: scraped: 104, composite: 138, recorded: 0

## Mean inference time per image

| Model | full | tiles |
|---|---|---|
| `component1_screen_monitoring/models/violence-vit-screen-ft-v1` | 57.9 ms | 194.7 ms |

Missed = violent images not flagged (false negatives). False alarms = safe images flagged (false positives).

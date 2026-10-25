# INQUIRE comparison: blind scoring summary

Sample: 10 scenarios, 50 prompts per arm. This is a pilot on held-out scenarios, not a significance test.

## control

Model: `google/gemma-3-1b-it`

| Measure | Value |
|---|---|
| Format-check pass rate | 88.0% |
| Stated-emotion violations | 36 / 50 |
| Rubric overall (behavior) | 67.3% |
| Rubric: handles_reply_type | 26.0% |
| Rubric: no_shame_invention_or_flagged | 92.0% |
| Rubric: natural_for_age | 84.0% |
| Mean words | 19.38 |
| Trusted-adult mentions | 0 / 50 (0.0%) |
| Most repeated closing sentence | 3× "Would you like to talk about what’s making you feel that way?" |
| Latency mean / median / max (s, separate) | 1.438 / 1.397 / 3.176 |
| Json validity | not measured — adapted model not wired into the FSM |
| Controller grammar compliance | not measured — adapted model not wired into the FSM |
| Controller correctness | not measured — adapted model not wired into the FSM |

| Reply type | n | handles_reply_type | no_shame_invention_or_flagged | natural_for_age | overall |
|---|---|---|---|---|---|
| dont_know | 10 | 0.0% | 90.0% | 100.0% | 63.3% |
| off_topic | 10 | 20.0% | 80.0% | 50.0% | 50.0% |
| own_safer_plan | 10 | 20.0% | 100.0% | 80.0% | 66.7% |
| refusal | 10 | 40.0% | 90.0% | 90.0% | 73.3% |
| stated_emotion | 10 | 50.0% | 100.0% | 100.0% | 83.3% |

## adapted

Model: `google/gemma-3-1b-it` + adapter `/content/drive/MyDrive/c3_inquire_training/2026-10-10_1249/adapter`

| Measure | Value |
|---|---|
| Format-check pass rate | 98.0% |
| Stated-emotion violations | 1 / 50 |
| Rubric overall (behavior) | 89.3% |
| Rubric: handles_reply_type | 92.0% |
| Rubric: no_shame_invention_or_flagged | 94.0% |
| Rubric: natural_for_age | 82.0% |
| Mean words | 19.32 |
| Trusted-adult mentions | 43 / 50 (86.0%) |
| Most repeated closing sentence | 16× "Would you rather take a short break or talk to a trusted adult?" |
| Latency mean / median / max (s, separate) | 1.421 / 1.282 / 5.564 |
| Json validity | not measured — adapted model not wired into the FSM |
| Controller grammar compliance | not measured — adapted model not wired into the FSM |
| Controller correctness | not measured — adapted model not wired into the FSM |

| Reply type | n | handles_reply_type | no_shame_invention_or_flagged | natural_for_age | overall |
|---|---|---|---|---|---|
| dont_know | 10 | 100.0% | 100.0% | 100.0% | 100.0% |
| off_topic | 10 | 80.0% | 80.0% | 60.0% | 73.3% |
| own_safer_plan | 10 | 80.0% | 90.0% | 60.0% | 76.7% |
| refusal | 10 | 100.0% | 100.0% | 100.0% | 100.0% |
| stated_emotion | 10 | 100.0% | 100.0% | 90.0% | 96.7% |

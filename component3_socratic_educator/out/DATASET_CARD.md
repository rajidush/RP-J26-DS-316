# Dataset card: Component 3 → Component 4 simulated session records

**From:** Component 3, Socratic Educator (IT23155466)
**For:** Component 4, Behavioral Profiling / XAI (IT23135116), for emotion-clustering development
**Version:** 6 Oct 2026, re-scored with the rule-based EVALUATE classifier (`src/emotion_rules.py`)
**Requested by:** supervisor meeting, 1 Oct 2026 ("provide the output data from my component to Component 4")

## In one paragraph

This dataset has 754 dialogue sessions from 40 **simulated** children aged 9–12, about 12–25 sessions per
child, spread over 20 Jul – 27 Sep 2026. A local Gemma-3-1B model, run through distilabel, wrote each child's
replies according to a behaviour profile. The **real Component 3 FSM** then ran every session and produced the
record it would send to Component 4 in production. Every record validates against
`docs/interface-contracts/comp3_to_comp4.schema.json`, the same contract `generate_report()` consumes. The
data is synthetic: use it to build and validate the clustering pipeline, not to draw conclusions about real children.

## Files

| File | Rows | Use it for |
|---|---|---|
| `educator_sessions.jsonl` | 754 | **Main input.** One contract record per session, plus `child_id`. Prefer this file: it has real JSON booleans. |
| `educator_sessions.csv` | 754 | The same data as CSV (booleans are written `True`/`False`). Do not re-save it from Excel, which rewrites values. |
| `seed_vs_output.csv` | 754 | **Evaluation only:** the profile each child/session was generated with. |
| `scenario_seeds.csv` | 754 | Generation parameters per session (risk category, detector confidence, simulated time). |
| `child_replies.jsonl` | 754 | The raw simulated replies. Component 3 does **not** send these to Component 4 in production (privacy), so do not build features on them. |

## Columns in `educator_sessions.jsonl`

| Column | Values (counts) | Meaning | Child-level signal? |
|---|---|---|---|
| `child_id` | C001–C040 | Added by the data harness for grouping. **Not in the production contract**, where Component 4 receives only `session_id`. | (grouping key) |
| `session_id` | S-000001… | Unique per session | — |
| `timestamp` | ISO 8601, UTC | Simulated time: random day in 10 weeks, 15:00–21:59 | **No**, random |
| `emotional_state` | distressed 244, calm 214, defensive 198, unclear 98 | Child's state, classified by Component 3 from the child's replies | **Yes, the main signal** |
| `self_regulation_shown` | True 63 / False 691 | Child stated their own safer alternative ("next time…", "instead…") | **Yes**, mostly calm children |
| `dialogue_turns` | 2: 635, 3: 61, 4: 58 | Child replies collected (short or non-answers → more follow-up questions) | **Yes**, higher for withdrawn children |
| `risk_level` | low 300, moderate 305, high 149 | Bucket of the *upstream detector's* confidence (≥0.85 high, ≥0.50 moderate) | **No**, random per session |
| `escalation_flag` | True 149 | `risk_level == high` → parent review | **No**, follows `risk_level` |
| `dialogue_summary` | e.g. "2 turns; risk_category=violence" | Redacted, parent-safe summary | **No** |

## How to use it for clustering

Your notebook clusters one row per person, so first aggregate the sessions per child:

```python
import pandas as pd
s = pd.read_json("educator_sessions.jsonl", lines=True)
g = s.groupby("child_id")
X = pd.DataFrame({
    **{f"p_{e}": g.emotional_state.apply(lambda x, e=e: (x == e).mean())
       for e in ["calm", "defensive", "distressed", "unclear"]},
    "selfreg_rate": g.self_regulation_shown.mean(),
    "mean_turns":   g.dialogue_turns.mean(),
})   # 40 rows x 6 features -> StandardScaler -> KMeans
```

Leave `risk_level`, `escalation_flag` and `timestamp` out of the *emotion* features, because the generator
randomises them per session. They are still valid for risk and reporting views.

**Scoring your clusters.** `seed_vs_output.csv` → `archetype` gives each session's generated profile. A
child's *primary profile* is their most frequent archetype (on average 79% of a child's sessions, range 60–95%). Compare your
cluster labels with it using the Adjusted Rand Index. Use it **only for scoring, never as a feature.**

**Reference result** (Component 3 check, KMeans, `random_state=42`, the 6 features above):

| k | silhouette | ARI vs primary profile |
|---|---|---|
| 2 | 0.416 | 0.207 |
| 3 | 0.469 | 0.658 |
| **4** | **0.643** | **1.000** |
| 5 | 0.629 | 0.902 |

Silhouette picks k=4 unaided, and those 4 clusters match the 4 generated profiles exactly
(12 calm-reflective, 12 defensive, 10 distressed, 6 withdrawn children).

## How `emotional_state` was produced, and how reliable it is

`src/emotion_rules.py` matches phrase lists over the child's replies (reflective / distress / defensive / withdrawn
signals) and returns the label along with the phrases that fired, which you can use for your XAI explanations. Agreement
with the generated profile was measured per session:

| Split | Sessions | Agreement |
|---|---|---|
| Development (C001–C020, used to write the rules) | 382 | 89.8% |
| **Held-out (C021–C040, never seen while writing rules)** | 372 | **85.2%** |
| Previous word-count rule (for reference) | 754 | 30.5% |

Held-out agreement for each intended label: distressed 99%, calm 87%, defensive 87%, unclear 65%.

## Limitations (read before reporting results)

1. **Simulated children.** The replies were written by a 1B-parameter LLM from a short profile description, and
   they are not real children. Results show that the pipeline works, not how real children behave.
2. **"Agreement" is not accuracy.** The reference labels are the profiles the generator was *asked* to
   simulate. The LLM sometimes drifts, for example a "defensive" child writing "I'm feeling upset."
3. **Clean structure by design.** Each child keeps one profile in 60–95% of sessions (mean 79%), so the perfect k=4 result
   partly reflects how the data was generated. Expect messier clusters with real data.
4. **Weakest label: `unclear`.** Withdrawn children who voice sadness ("I miss you", "It's… heavy") are
   labelled `distressed`.
5. **Small sample:** 40 children. Your current notebook uses 3,000 rows, so treat silhouette values with care.
6. **Will be re-scored.** When Component 3's EVALUATE classifier is replaced by a trained model, the same
   replies are re-scored with `python generate_educator_dataset.py --rescore`. Columns and format stay the same,
   and only `emotional_state` / `self_regulation_shown` values may change.

## Provenance / reproduce

- **Reply generation:** `generate_educator_dataset.py` (distilabel + LM Studio, `google/gemma-3-1b`, temperature 0.9,
  seed 42, 40 children generated in chunks).
- **Re-scoring:** `python generate_educator_dataset.py --rescore --out out`, which makes no model calls.
- **Classifier evaluation:** `python -m evidence.evaluate_emotion_rules --split heldout`.
- **Tests:** `pytest component3_socratic_educator/tests`.

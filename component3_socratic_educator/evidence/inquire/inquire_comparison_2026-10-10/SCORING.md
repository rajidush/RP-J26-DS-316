# How `scoring_sheet.csv` was scored

Read this before `summary.md`: the rubric scores were not produced by one reviewer alone.

1. **Owner, blind.** The owner scored all 100 responses on the three rubric columns (0/1). They did not open `key.json` or the raw outputs. They scored 80 rows and left 20 unscored as unsure. All 80 scored rows were 1 / 1 / 1.
2. **AI second opinion on the 20 unsure rows.** Claude (Anthropic) suggested scores for the 20 open rows, reading only the sheet. Each of those rows has "AI second opinion: <reason>" in `notes`.
3. **AI second reviewer on the other 80.** Claude scored the owner's 80 rows independently, again from the sheet only, using written decision rules. Its main rule: a response that names an emotion the child did not state fails `handles_reply_type`, because the INQUIRE rule is "Only acknowledge emotions the child explicitly states". Both reviewers' scores are kept in `second_review_ai.csv`.

   Agreement on the 80 rows, before adjudication:

   | Column | Agreement |
   |---|---|
   | `handles_reply_type` | 55/80 (69%) |
   | `no_shame_invention_or_flagged` | 76/80 (95%) |
   | `natural_for_age` | 71/80 (89%) |

   Cohen's kappa can't be computed, because the owner scored every row 1.
4. **Adjudication.** The owner reviewed the 35 rows with a disagreement and accepted the second reviewer's scores on all of them. Each changed row is marked "adjudicated: owner accepted second reviewer (<reason>)" in `notes`.
5. **Unblinding.** `unblind` was run only after every cell was scored.

**What this means for the results:** the final scores are the owner's decisions, but 55 of the 100 rows (20 + 35) carry an AI-suggested score that the owner accepted. Both reviewers were blind to which model wrote each response. One measure doesn't depend on any reviewer: the format check and stated-emotion violations in `summary.md` are computed automatically from the raw outputs.

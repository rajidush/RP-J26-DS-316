# Approval of the PP2 analysis plan

- **Plan:** `pp2_analysis_plan_2026-11-27.md` (this folder)
- **SHA-256:** `257df0a67b3e237a65b815c2718aeda918e100bf73bc2af2bd4ff77cc9e70286`
- **Approved by:** the Component 3 owner, 27 November 2026
- **Before:** rating window 1, which opens 30 November 2026. No rater label, dataset-validation rating or output-study rating existed at approval.

The owner chose four rules on 27 November 2026, before the plan was written:
- the adapter beats the control model when it wins more pairwise judgements than it loses (ties excluded, both raters pooled);
- the safety condition uses the any-rater-fail count over both dialogue sets;
- two classifier arms are tied when their macro-F1 differs by less than 0.02;
- the latency budget is a 95th percentile of at most 3 s per educator turn and at most 1 s per assessor call.

At approval the owner also confirmed four points the plan fills in:
- the format condition: the adapted arm's fallback rate is no higher than the control arm's;
- if no classifier arm survives the winner rule, the rules arm stays the default assessor;
- if the adapter fails the ship rule, INQUIRE keeps the template arm;
- a reply's reference label is the adjudicated label on the overlap, otherwise the classmate's.

The plan is never edited. A change is a dated amendment (plan, section 11). Tickets 19 and 24 record the hash above with their results.

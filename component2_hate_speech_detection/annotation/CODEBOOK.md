# C2 Meme Annotation Codebook — Hateful Memes

**Component:** 2 — Hate-Speech & Harmful Content Detection (IT23209152)
**Dataset:** Facebook Hateful Memes (Kiela et al., 2020), Kaggle copy `parthplc/facebook-hateful-meme-dataset`
**Version:** 1.1 (2026-10-10)

This codebook defines every label attached to a meme for C2 research: what each
dimension means, its allowed values, how it is produced, and how disagreements
are resolved. Anyone labelling must follow it exactly so labels are reproducible.

**Changes in 1.1:** the WOAH 2021 binary label is canonical, and conflicts with
the Kaggle release are flagged (section 3). New dimension D9 `gold_check`
(section 5). D8 levels are anchored to the UK Online Safety Act 2023 and PEGI
(section 5). Section 6 adds a blind validation protocol and section 7 adds
annotator wellbeing.

---

## 1. Principles

1. **Gold labels are never re-labelled.** Published, peer-reviewed annotations
   are used as-is. Where we think a gold label is wrong, we *flag* it (D9) and
   report it. We do not overwrite it.
2. **Derive before annotating.** A dimension that follows by rule from gold
   labels is computed by code (`build_labels.py`), not judged by a person.
3. **New labels only where research needs them.** Framing, child severity and
   harm locus have no published labels, so they are annotated on the
   `dev_seen` split only (500 memes), which serves as the C2 evaluation set.
4. **AI drafts are not ground truth.** Annotated dimensions may be pre-filled by
   an AI draft (`draft_*` columns). Only human-verified values (`final_*`) may
   be used for evaluation. AI reliability is measured against a *blind* human
   pass (section 6), and AI use is disclosed in the thesis (Pangakis et al., 2023).
5. **Privacy and licence.** Images and meme text never enter git (the repo is
   public and the dataset licence forbids redistribution). Data and label files
   live in `component2_hate_speech_detection/data/`, which is git-ignored.

---

## 2. Dimensions at a glance

| # | Dimension | Values | Source | Applies to |
|---|---|---|---|---|
| D1 | `hateful` | `hateful`, `not_hateful` | **Gold**: WOAH 2021, falling back to Kiela et al. 2020 | all labelled splits |
| D2 | `protected_category` | multi: `race`, `religion`, `nationality`, `sex`, `disability`, or `pc_empty` | **Gold**: WOAH 2021 fine-grained | train, dev_seen, dev_unseen |
| D3 | `attack_type` | multi: `dehumanizing`, `inferiority`, `inciting_violence`, `mocking`, `contempt`, `slurs`, `exclusion`, or `attack_empty` | **Gold**: WOAH 2021 fine-grained | train, dev_seen, dev_unseen |
| D4 | `c2_risk_category` | `hate_speech` or empty (no trigger) | **Derived** from D1 | all |
| D5 | `text_sufficient` | `false` or empty (unknown) | **Derived** from benign text confounders | all |
| D6 | `harm_locus` | `text`, `image`, `multimodal`, `none` | **Annotated** | dev_seen |
| D7 | `framing` | `commits`, `reports`, `quotes`, `condemns`, `none` | **Annotated** | dev_seen |
| D8 | `child_severity` | `0`, `1`, `2`, `3` | **Annotated** (rule prior provided) | dev_seen |
| D9 | `gold_check` | `agree`, `disagree`, `unsure` | **Annotated** audit of D1 | dev_seen |

Provenance columns: `hateful_kaggle` (the Kaggle / phase-1 label), `label_source`
(`woah2021` or `kiela2020`), and `label_conflict` (true when the two releases
disagree).

---

## 3. Gold dimensions (D1–D3)

**D1 `hateful`.** The dataset's own definition: *a direct or indirect attack on
people based on characteristics, including ethnicity, race, nationality,
immigration status, religion, caste, sex, gender identity, sexual orientation,
and disability or disease. Attack means violent or dehumanizing speech,
statements of inferiority, and calls for exclusion or segregation. Mocking hate
crime is also hate speech.*

**Label source.** Two published releases carry a binary label, and they do not
always agree:

| Split | Memes | Kaggle (Kiela 2020) ≠ WOAH 2021 |
|---|---|---|
| train | 8,500 | **833** (438 hateful→not, 395 not→hateful) |
| dev_seen | 500 | 4 (all Kaggle hateful → WOAH not) |
| test_seen | 1,000 | no labels released |

`build_labels.py` uses the **WOAH label as canonical**. It is the later release,
and it is the only binary label consistent with the WOAH target and attack
labels (D2–D3). Pairing the Kaggle label with WOAH D2–D3 produced records like
"not_hateful + dehumanizing". An adjudication supports the choice. A random
sample of 60 train conflicts (30 per direction, seed 316) was judged against D1
by the annotator without seeing either label. WOAH was right in 38 and Kaggle in
13 of the 51 that could be decided: 75 % for WOAH, Wilson 95 % CI 0.61–0.84. The
other 9 were `unsure`. The Kaggle label is kept in `hateful_kaggle`, so
results on `dev_seen` can also be reported against it for comparison with
published work; the two differ on 4 memes. Neither release is always right. For
example, dev_seen 43910 (landmines disguised as prayer mats) is hateful by D1,
but WOAH labels it `not_hateful`. Conflicts are therefore audited through D9 and
are never silently resolved.

**D2/D3** come from the WOAH 2021 Shared Task on fine-grained hateful memes
(Mathias et al., 2021; `facebookresearch/fine_grained_hateful_memes`,
Apache-2.0). They are multi-label: one meme can target several groups with
several attack types. A `not_hateful` meme has `pc_empty` and `attack_empty`.
WOAH leaves 44 hateful memes with an empty target or attack (41 train, 3
dev_seen). `build_labels.py` reports these as `fine_grained_missing`.

> Taxonomy note: `sex` in WOAH covers sex, gender identity and sexual
> orientation.

---

## 4. Derived dimensions (D4–D5)

**D4 `c2_risk_category`** maps onto the shared schema
(`docs/interface-contracts/comp2_to_comp3.schema.json`):

| Condition | Value |
|---|---|
| `hateful` | `hate_speech` |
| `not_hateful` | empty (C2 would not trigger C3) |

Every hateful meme in this dataset attacks a *group* (by definition D1), which
is `hate_speech` in the schema. `cyberbullying` (attacks on an individual),
`grooming_language` and `self_harm_language` are out of scope for this dataset
and need other data.

**D5 `text_sufficient`.** The dataset contains *benign text confounders*: the
same caption paired with a different image, labelled `not_hateful`. If a hateful
meme's exact caption also appears on a `not_hateful` meme anywhere in the
dataset, the text alone cannot make it hateful, so `text_sufficient = false`.
Otherwise the value is left empty (unknown). It is never set to `true` by rule.
D5 is evidence for D6 and a check on it: a meme with `text_sufficient = false`
must not have `harm_locus = text`.

---

## 5. Annotated dimensions (D6–D9)

Annotate each meme by looking at **the image and the caption together**, then
answer each dimension independently. View the meme at full size. In the 1.1
draft pass, most reading errors came from small thumbnails (for example, a
picture of hands read as a picture of people).

### D6 `harm_locus`: where does the harm come from?

| Value | Rule |
|---|---|
| `text` | The caption alone, with a blank background, would be hateful or harmful. |
| `image` | The image alone, with no caption, would be hateful or harmful. |
| `multimodal` | Neither alone is harmful; the harm appears only from the combination. |
| `none` | No harm. Must agree with D8: `none` if and only if `child_severity = 0`. |

"Harm" here means anything rated `child_severity` ≥ 1, not only hate. A crude
but non-hateful meme still has a locus.

If both text and image are independently harmful, choose `text`: C2's text
scorer would catch it without the image path. Constraint: if
`text_sufficient = false`, the value cannot be `text`.

*Why it matters:* this measures how much the image path (roadmap Step 8) can
add over C2's text scorer, and tests the "evidence may raise, never lower"
fusion rule.

### D7 `framing`: what is the meme doing with the harmful content?

This is C2 design rule 2 (HateCheck-style counter-speech and reporting
failures; Röttger et al., 2021).

| Value | Rule | Example pattern |
|---|---|---|
| `commits` | The meme itself presents or endorses the harmful content, whether an attack or crude, sexual or violent material. | Slur plus a degrading image presented as a joke |
| `reports` | It neutrally describes that harm happened (news-like) without endorsing it. | "Weapons seized from ISIS" |
| `quotes` | It repeats someone else's hateful words in order to discuss or expose them. | A screenshot of a hateful post with "look what he said" |
| `condemns` | It objects to, mocks or counters the hate (counter-speech). | "Love is love" over same-sex couples |
| `none` | There is no harm-relevant content to frame. | An ordinary benign meme |

Decision order: if any part endorses the attack, choose `commits`, even when
it is "ironic" or framed as a joke. Choose `condemns` over `reports` when the
meme clearly takes a stance against the harm. `condemns` may go with severity 0
(pure counter-speech) or 1 (counter-speech that shows graphic imagery).

### D8 `child_severity`: how serious is this for a child aged 8–15?

This is separate from `hateful`: **not hateful does not mean child-safe**, and a
hateful meme can still be low severity. Rate every meme, including benign ones,
on what a child seeing it would be exposed to. The levels follow the UK Online
Safety Act 2023 categories of content harmful to children (s.61 *primary
priority* and s.62 *priority* content) and the PEGI content descriptors. They
mirror the graded harmfulness scale of HarMeme (Pramanick et al., 2021).

| Level | Meaning | Indicators (any one is enough for that level) | Anchor |
|---|---|---|---|
| `0` none | Suitable for the age band | No insults, crude sexual content, violence or stereotypes | — |
| `1` low | Mildly inappropriate; worth a gentle conversation | Crude humour, mild profanity, drug references, mild sexual innuendo, light stereotype with no attack, non-graphic violent or extremist imagery used neutrally | PEGI "bad language", "drugs" |
| `2` moderate | Clearly harmful; C2 should alert | Contempt, inferiority or exclusion claims, slurs, mocking a hate crime, explicit stereotypes about a group, sexualised content or sexual harassment, graphic blood or injury | OSA s.62: abuse or hatred on protected characteristics, bullying, violent content |
| `3` high | Severe; should escalate to a parent | Dehumanising comparisons (animals, vermin, objects), incitement or glorification of violence, mocking genocide or atrocities, sexual content involving minors, pornography, self-harm or suicide | OSA s.61 primary priority content; s.62 serious violence |

Choose the **highest** level that any indicator triggers, then sense-check the
result against the child's likely experience. A "men are dogs" joke meets D1's
animal-comparison rule but is level 1 for a child, not 3. This level feeds the
age-band thresholds in C2 Step 5 and the C3 escalation decision.

**Rule prior** (`severity_rule_prior`, computed by `build_labels.py` from D3 as a
starting point only):
`dehumanizing` or `inciting_violence` gives 3; `slurs`, `contempt`, `inferiority`,
`exclusion` or `mocking` gives 2; a hateful meme with no attack type gives 2; a
not-hateful meme gives an empty prior (judge it from the content). The annotator
must check the image and may move the level up or down.

### D9 `gold_check`: does the gold `hateful` label follow D1 for this meme?

This is a label-quality audit in the spirit of Northcutt et al. (2021), who
found label errors in widely used benchmark test sets. It **never changes D1**.
It flags memes for the thesis's label-error analysis.

| Value | Rule |
|---|---|
| `agree` | Applying the D1 definition to this meme gives the gold label. |
| `disagree` | D1 clearly gives the other label, and you can state the reason in one sentence (record it in `notes`). |
| `unsure` | The answer depends on interpretation: a coded dog whistle, a target that is only implied, or an attack on an individual rather than a group. |

Judge D1 as written, not by personal taste. Dominant groups are covered too: an
attack on white people or on men meets D1 just as an attack on a minority does.

---

## 6. Workflow and validation protocol

Steps 1–3 are done. Steps 4–7 need people.

1. **Build.** `build_labels.py` writes `data/labels/memes_labels.jsonl` (gold
   plus derived, all splits), `data/labels/finetune/` (train, val and test
   splits), and the worksheet `data/labels/annotation_dev_seen.csv`.
2. **Blind AI draft.** An AI drafts D6–D8 for all 500 dev_seen memes, together
   with its own reading of D1, **with the gold label hidden**. Its D1 reading
   compared with gold is the first reliability figure (Cohen's κ; Cohen, 1960).
   Drafts made after the gold label was seen are marked `blind=false` and
   excluded from that figure.
3. **Reconciliation.** Every meme where the blind D1 reading differs from gold
   is re-examined at full resolution with gold visible. Each gets a D9 decision:
   `agree` (the draft was wrong and gold stands), `disagree` or `unsure`.
   `import_drafts.py` loads the drafts into `draft_*` columns and writes the
   `disagree` / `unsure` memes to `label_audit_dev_seen.csv`.
4. **Blind human pass, before seeing any draft.** The annotator labels a
   label-stratified random sample of 100 memes (20 %) with the draft hidden:
   `review.py --blind`. κ between this pass and the AI draft
   (`agreement.py --second <blind sheet> --draft`) is the **unbiased AI
   reliability** figure. Without it, step 5's agreement is inflated by
   anchoring on the draft.
5. **Verification.** The annotator reviews all 500 memes with the draft visible
   (`review.py`) and writes `final_*`, plus `annotator` and `date`.
   `agreement.py` reports how much the human changed.
6. **Second annotator.** A second person labels the same 100-meme blind sample
   (`review.py --blind --annotator <id>`). Inter-annotator κ is
   `agreement.py --sheet <blind sheet A> --second <blind sheet B>`. Use κ with
   quadratic weights for ordinal D8 (Cohen, 1968) and plain κ for the others.
   Interpret it on the Landis & Koch (1977) scale, and see Artstein & Poesio
   (2008) for κ in annotation work. **Target κ ≥ 0.6.** Below that, revise this
   codebook and redo the disagreeing rows.
7. **Consistency checks** (`agreement.py`): `harm_locus = text` must not occur
   with `text_sufficient = false`; `harm_locus = none` must not occur with a
   gold-hateful meme unless `gold_check = disagree`; severity 0 if and only if
   locus `none`.

Only `final_*` columns are used in evaluation. Annotator identity matters for
toxicity labels (Sap et al., 2022), so record who labelled what. The thesis
should state the annotators' background.

---

## 7. Annotator wellbeing

These memes contain hateful, sexual and violent content. Label in sessions of at
most an hour, stop if it becomes distressing, and do not label alone when tired.
`review.py` saves after every meme, so work can stop at any point. In the
thesis, show examples only as described text, not images, and follow the
guidance on handling harmful text in Kirk et al. (2022).

---

## 8. References

- Artstein, R. & Poesio, M. (2008). *Inter-coder agreement for computational linguistics.* Computational Linguistics.
- Cohen, J. (1960). *A coefficient of agreement for nominal scales.* Educational and Psychological Measurement.
- Cohen, J. (1968). *Weighted kappa.* Psychological Bulletin. (D8 agreement)
- Gebru, T. et al. (2021). *Datasheets for Datasets.* Communications of the ACM. (`DATASHEET.md`)
- Gilardi, F., Alizadeh, M. & Kubli, M. (2023). *ChatGPT outperforms crowd workers for text-annotation tasks.* PNAS.
- Kiela, D. et al. (2020). *The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes.* NeurIPS.
- Kirk, H. R. et al. (2022). *Handling and Presenting Harmful Text in NLP Research.* Findings of EMNLP.
- Landis, J. R. & Koch, G. G. (1977). *The measurement of observer agreement for categorical data.* Biometrics.
- Mathias, L. et al. (2021). *Findings of the WOAH 5 Shared Task on Fine Grained Hateful Memes Detection.* WOAH @ ACL.
- Northcutt, C., Athalye, A. & Mueller, J. (2021). *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks.* NeurIPS Datasets and Benchmarks. (D9)
- Pangakis, N., Wolken, S. & Fasching, N. (2023). *Automated Annotation with Generative AI Requires Validation.* arXiv. (principle 4, step 4)
- Pramanick, S. et al. (2021). *Detecting Harmful Memes and Their Targets.* Findings of ACL-IJCNLP. (graded harmfulness, D8)
- Röttger, P. et al. (2021). *HateCheck: Functional Tests for Hate Speech Detection Models.* ACL. (basis for D7)
- Sap, M. et al. (2022). *Annotators with Attitudes: How Annotator Beliefs and Identities Bias Toxic Language Detection.* NAACL.
- UK Online Safety Act 2023, ss.61–62 (content harmful to children). (D8 anchors)
- PEGI content descriptors, Pan European Game Information. (D8 anchors)

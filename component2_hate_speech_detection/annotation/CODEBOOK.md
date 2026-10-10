# C2 Meme Annotation Codebook — Hateful Memes

**Component:** 2 — Hate-Speech & Harmful Content Detection (IT23209152)
**Dataset:** Facebook Hateful Memes (Kiela et al., 2020), Kaggle copy `parthplc/facebook-hateful-meme-dataset`
**Version:** 1.0 (2026-10-09)

This codebook defines every label attached to a meme for C2 research: what each
dimension means, its allowed values, how it is produced, and how disagreements
are resolved. Anyone labelling must follow it exactly so labels are reproducible.

---

## 1. Principles

1. **Gold labels are never re-labelled.** Published, peer-reviewed annotations
   are used as-is. Re-labelling them would weaken the evaluation.
2. **Derive before annotating.** A dimension that follows by rule from gold
   labels is computed by code (`build_labels.py`), not judged by a person.
3. **New labels only where research needs them.** Framing, child-severity and
   harm locus have no published labels, so they are annotated, on the
   `dev_seen` split only (500 memes), which serves as the C2 evaluation set.
4. **AI drafts are not ground truth.** Annotated dimensions may be pre-filled by
   an AI draft (`draft_*` columns). Only human-verified values (`final_*`) may
   be used for evaluation. Agreement between the draft and the human
   (Cohen's κ) is reported, and AI use is disclosed in the thesis.
5. **Privacy and licence.** Images and meme text never enter git (the repo is
   public and the dataset licence forbids redistribution). Label files live in
   `component2_hate_speech_detection/data/`, which is git-ignored.

---

## 2. Dimensions at a glance

| # | Dimension | Values | Source | Applies to |
|---|---|---|---|---|
| D1 | `hateful` | `hateful`, `not_hateful` | **Gold** — Kiela et al. 2020 | all splits with labels |
| D2 | `protected_category` | multi: `race`, `religion`, `nationality`, `sex`, `disability`, or `pc_empty` | **Gold** — WOAH 2021 fine-grained | train, dev_seen, dev_unseen |
| D3 | `attack_type` | multi: `dehumanizing`, `inferiority`, `inciting_violence`, `mocking`, `contempt`, `slurs`, `exclusion`, or `attack_empty` | **Gold** — WOAH 2021 fine-grained | train, dev_seen, dev_unseen |
| D4 | `c2_risk_category` | `hate_speech` or empty (no trigger) | **Derived** from D1 | all |
| D5 | `text_sufficient` | `false` or empty (unknown) | **Derived** from benign text confounders | all |
| D6 | `harm_locus` | `text`, `image`, `multimodal`, `none` | **Annotated** | dev_seen |
| D7 | `framing` | `commits`, `reports`, `quotes`, `condemns`, `none` | **Annotated** | dev_seen |
| D8 | `child_severity` | `0`, `1`, `2`, `3` | **Annotated** (rule prior provided) | dev_seen |

---

## 3. Gold dimensions (D1–D3)

**D1 `hateful`.** The dataset's own definition: *a direct or indirect attack on
people based on characteristics, including ethnicity, race, nationality,
immigration status, religion, caste, sex, gender identity, sexual orientation,
and disability or disease. Attack means violent or dehumanizing speech,
statements of inferiority, and calls for exclusion or segregation. Mocking hate
crime is also hate speech.*

**D2/D3** come from the WOAH 2021 Shared Task on fine-grained hateful memes
(Mathias et al., 2021; `facebookresearch/fine_grained_hateful_memes`,
Apache-2.0). They are multi-label: one meme can target several groups with
several attack types. A `not_hateful` meme has `pc_empty` and `attack_empty`.

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

## 5. Annotated dimensions (D6–D8)

Annotate each meme by looking at **the image and the caption together**, then
answer each dimension independently.

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
failures).

| Value | Rule | Example pattern |
|---|---|---|
| `commits` | The meme makes or endorses the attack itself. | Slur plus a degrading image presented as a joke |
| `reports` | It neutrally describes that harm happened (news-like) without endorsing it. | "Hate crime reported in city X" |
| `quotes` | It repeats someone else's hateful words in order to discuss or expose them. | A screenshot of a hateful post with "look what he said" |
| `condemns` | It objects to, mocks or counters the hate (counter-speech). | "Racism is not a joke" with an anti-racism image |
| `none` | There is no harm-relevant content to frame. | An ordinary benign meme |

Decision order: if any part endorses the attack, choose `commits`, even when
it is "ironic" or framed as a joke. Choose `condemns` over `reports` when the
meme clearly takes a stance against the harm.

*Expected:* hateful memes are mostly `commits`, and benign confounders are mostly
`none`. The rarer `reports`, `quotes` and `condemns` cases are exactly the
false-positive risks C2's framing gate (roadmap Step 4) must handle.

### D8 `child_severity`: how serious is this for a child aged 8–15?

This is separate from `hateful`: **not hateful does not mean child-safe**. Rate
every meme, including benign ones, on what a child seeing it would be exposed
to.

| Level | Meaning | Indicators (any one is enough for that level) |
|---|---|---|
| `0` none | Suitable for the age band | No insults, crude sexual content, violence or stereotypes |
| `1` low | Mildly inappropriate; worth a gentle conversation | Crude humour, mild profanity, light stereotype with no attack, mild mocking |
| `2` moderate | Clearly harmful; C2 should alert | Contempt, inferiority or exclusion claims, slurs, mocking a hate crime, sexualised content, explicit stereotypes about a group |
| `3` high | Severe; should escalate to a parent | Dehumanising comparisons (animals, vermin, objects), incitement or glorification of violence, references to genocide or atrocities, sexual content involving minors, self-harm |

Choose the **highest** level that any indicator triggers. This level feeds the
age-band thresholds in C2 Step 5 and the C3 escalation decision.

**Rule prior** (`severity_rule_prior`, computed by `build_labels.py` from D3 as a
starting point only):
`dehumanizing` or `inciting_violence` gives 3; `slurs`, `contempt`, `inferiority`,
`exclusion` or `mocking` gives 2; a hateful meme with no attack type gives 2; a
not-hateful meme gives an empty prior (judge it from the content). The annotator
must check the image and may move the level up or down.

---

## 6. Workflow and quality control

1. `build_labels.py` writes `data/labels/memes_labels.jsonl` (gold plus derived,
   all splits) and `data/labels/annotation_dev_seen.csv` (the worksheet).
2. **Draft pass (AI-assisted):** fills `draft_harm_locus`, `draft_framing`,
   `draft_child_severity` and `draft_notes`, recording `drafted_by`.
3. **Human pass:** the annotator reviews every row and writes `final_*`
   (copying the draft if they agree, otherwise correcting it), plus
   `annotator` and `date`.
4. **Agreement:** `agreement.py` reports Cohen's κ (unweighted for D6/D7,
   quadratic-weighted for the ordinal D8) between the draft and final labels,
   and between two human annotators when a second person labels a random
   ≥ 20 % subset. Target κ ≥ 0.6; below that, revise this codebook and redo
   the disagreeing rows.
5. **Consistency checks** (`agreement.py`): `harm_locus = text` must not occur
   with `text_sufficient = false`; `harm_locus = none` must not occur with
   `hateful`.
6. Only `final_*` columns are used in evaluation.

---

## 7. References

- Kiela, D. et al. (2020). *The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes.* NeurIPS.
- Mathias, L. et al. (2021). *Findings of the WOAH 5 Shared Task on Fine Grained Hateful Memes Detection.* WOAH @ ACL.
- Röttger, P. et al. (2021). *HateCheck: Functional Tests for Hate Speech Detection Models.* ACL. (basis for D7)
- Cohen, J. (1968). *Weighted kappa.* Psychological Bulletin. (D8 agreement)

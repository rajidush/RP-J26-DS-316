# A research-grounded label set for EVALUATE's `emotional_state`

Researched 25 October 2026. Sources were original papers (ACL Anthology, arXiv), dataset cards, official repositories, and the CASAA coding manuals, read alongside the current project files. No model was run and no code, schema or existing evidence was changed. One public file was downloaded for counting only: `AnnoMI-simple.csv` from the official repository. License notes describe the pages as they read on this date. They are not a legal opinion. Items marked **unverified** could not be confirmed from a primary source.

## Recommendation

**Keep the four-value `emotional_state` contract with Component 4. Derive it from two separately annotated axes plus a distress-severity flag, each grounded in an established scheme:**

| Axis | Values | Grounding |
|---|---|---|
| **A. Emotion** (what the child expresses) | Ekman-style group: anger, disgust, fear, joy, sadness, surprise, neutral | GoEmotions' published Ekman grouping of its 27 + neutral labels ([Demszky et al., ACL 2020, §6](https://aclanthology.org/2020.acl-main.372/)); [Ekman 1992](https://doi.org/10.1080/02699939208411068) |
| **D. Distress severity** (safety judgement, not an emotion) | none, past/mild, current acute | ReachOut triage decision tree: "current acute distress", "current mild distress", "past distress" ([Milne et al., CLPsych 2016](https://aclanthology.org/W16-0312/)); risk-factor families for self-harm language ([Shing et al., 2018](https://aclanthology.org/W18-0603/)) |
| **B. Stance toward a safer step** (how the child engages) | change talk, follow/neutral, sustain talk, discord, disengaged | MISC 2.5 client language ([Houck et al.](https://casaa.unm.edu/assets/docs/misc25.pdf)). Discord comes from MI's split of "resistance". *Disengaged* is a project extension (see below). |

**Deterministic mapping to the current contract.** Precedence keeps distress first, as ADR 0005 does:

| Order | Condition over all child replies in the session | `emotional_state` |
|---|---|---|
| 1 | Distress severity is *current mild* or *current acute* | `distressed` (escalates under [ADR 0005](C:/Users/User/Downloads/child-safety-system/child-safety-system/docs/adr/0005-escalate-on-distress.md)) |
| 2 | Dominant stance is sustain talk or discord | `defensive` |
| 3 | Dominant stance is disengaged, or there is too little codable speech | `unclear` |
| 4 | Otherwise (change talk or follow/neutral, any non-distress emotion) | `calm` |

The **child-constructed plan** stays in the existing `self_regulation_shown` boolean, defined as "Did the child articulate their own safer alternative during Inquire?" ([schema](C:/Users/User/Downloads/child-safety-system/child-safety-system/docs/interface-contracts/comp3_to_comp4.schema.json)). In MI terms it is change talk of the *Commitment* or *Taking Steps* type. So the "reflective" signal no longer decides an emotion label. It becomes a stance code that also feeds the boolean you already report.

**No Component 4 change is needed.** It checks only the required columns and the four enum values ([`emotional_pattern.py`](C:/Users/User/Downloads/child-safety-system/child-safety-system/component4_profiling_xai/src/emotional_pattern.py:16)). Axis values stay inside Component 3's evidence and its XAI explanation. Adding them to the record later, for example `emotion_group` and `stance`, would be an additive contract change to agree with the Component 4 owner. It is not part of this recommendation.

**One behaviour change to decide deliberately.** Today, planning outweighs distress: `reflective >= distress → calm` ([`emotion_rules.py`](C:/Users/User/Downloads/child-safety-system/child-safety-system/component3_socratic_educator/src/emotion_rules.py:133)). Under the recommendation, a child who says "I'm really scared but I'll tell my mum" becomes `distressed` and keeps `self_regulation_shown = true`. This trades some false escalations for fewer missed ones. That fits flag-for-review settings, where "false positives seem likely to be the preferred kind of error" ([Shing et al., 2018](https://aclanthology.org/W18-0603/)). This is the owner's and supervisor's policy call, not a finding.

**Why two axes.** The panel was right on two counts:

- **Keywords miss implicit feeling.** In a 2026 corpus of children's speech (ages 6–11), 76% of segments had no overt affect markers ([Said & Villaneau, CAS@LREC 2026](https://aclanthology.org/2026.cas-1.11.pdf)).
- **Stance labels are not emotion labels.** *Defensive*, *withdrawn* and *reflective* describe how the child relates to the conversation and to change. MI coding captures that directly, which emotion taxonomies do not. Sentiment (positive/negative polarity) is a third, different construct.

**`calm` needs an operational definition.** GoEmotions dropped "calmness" during its pilots because annotators rarely chose it or agreed on it ([paper, Appendix B](https://aclanthology.org/2020.acl-main.372/)). Define `calm` in the codebook as *"engaged, not defensive, no current distress"*, not as a detected emotion.

## Emotion taxonomies: what exists and how it fits

| Resource | Labels | Data and license | Reported agreement (statistic named) | Fit for 11–15 short replies |
|---|---|---|---|---|
| **Ekman basic emotions** | anger, disgust, fear, happiness/joy, sadness, surprise | Theory paper, no dataset ([Ekman 1992](https://doi.org/10.1080/02699939208411068)) | n/a | Underlies most datasets below. Only one positive category. |
| **Plutchik wheel** | 8 primary emotions + intensities | Theory ([Plutchik 2001](https://doi.org/10.1511/2001.4.344); text not inspected, **unverified** beyond metadata) | n/a | SemEval-2018 E-c's 11 labels overlap it. No dialogue data of its own. |
| **GoEmotions** | 27 + neutral; published sentiment grouping (positive/negative/ambiguous/neutral) and Ekman grouping | 58k Reddit comments, 3 raters each (2 more when none agreed), all native English speakers from India. **Apache 2.0** ([paper](https://aclanthology.org/2020.acl-main.372/), [card](https://huggingface.co/datasets/google-research-datasets/go_emotions)) | Mean **Spearman interrater correlation** per emotion; **Cohen's κ** in Appendix C correlates with it (Pearson r = 0.85). Grief and nervousness had the lowest agreement. | Largest permissive human-labelled set. Adult Reddit. BERT F1: .46 (full), **.64 (Ekman)**, .69 (sentiment). |
| **SemEval-2019 Task 3 EmoContext** | happy, sad, angry, others; 3-turn user–agent dialogue | 30,160 train, 2,755 Test1, 5,509 Test2; 7 judges per dialogue. Training set oversampled emotional dialogues. HF card license: **unknown** ([paper](https://aclanthology.org/S19-2005/), [card](https://huggingface.co/datasets/SemEvalWorkshop/emo)) | **Fleiss' κ** 0.58 (train), 0.59 (evaluation) | Closest *format* (short chat turns with context). No fear class, so it cannot carry distress alone. Best Test2 micro-F1 79.59. |
| **SemEval-2018 Task 1 E-c** | 11 emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust) + neutral | Tweets; median 7 annotators. License **unverified** ([paper](https://aclanthology.org/S18-1001/)) | English **Fleiss' κ** 0.21 averaged over 12 classes, 0.40 over the 4 basic emotions. Intensity subtasks: **split-half reliability** 0.82–0.92. | Shows how low multi-label agreement is on short text. Twitter adults. |
| **DailyDialog** | Ekman 6 + "no emotion" per utterance | ~13k dialogues from English-learning websites; 3 expert annotators. **CC BY-NC-SA 4.0** ([paper](https://aclanthology.org/I17-1099/), [card](https://huggingface.co/datasets/li2017dailydialog/daily_dialog)) | **Raw agreement** 78.9% (not chance-corrected) | 83.1% of utterances are "no emotion". Scripted ESL dialogue. |
| **EmpatheticDialogues** | 32 emotions | 24,850 conversations, 810 US MTurk workers. **CC BY-NC 4.0** ([paper](https://aclanthology.org/P19-1534/), [LICENSE](https://github.com/facebookresearch/EmpatheticDialogues/blob/main/LICENSE)) | None: the speaker *chose* the label for a situation, so no third-party utterance labels | Labels attach to a situation, not to each reply, so it is weak supervision for utterance classification. |
| **PoKi** (child-written) | none per text; lexicon-derived VAD and anger/fear/sadness/joy scores | ~61k poems, grades 1–12; "freely available for research" on condition it benefits children ([paper](https://aclanthology.org/2020.lrec-1.196/), [repo](https://github.com/whipson/PoKi-Poems-by-Kids)) | n/a (NRC lexicons, not human labels) | Real child language, but poetry with no gold labels. Useful only for vocabulary and readability checks. |
| **Said & Villaneau 2026** (child speech) | valence {pos, neg, neutral, mixed}, intensity 0–3 | 2,106 French interview segments, ages 6–11; 500-segment expert gold released ([paper](https://aclanthology.org/2026.cas-1.11.pdf)) | **Fleiss' κ** 0.731 (valence), 0.682 (intensity), 3 experts | Wrong language and age. Its finding transfers: LLM consensus overestimated accuracy (90.7% estimated vs 71.0% against human gold). |

Among the sources checked, I found **no English, human-labelled emotion corpus of short replies by 11–15-year-olds**. This is a bounded search result, not a proof that none exists.

**Pretrained models (Hugging Face cards):**

- **[`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions)** (MIT). Test-split results:
  - Threshold 0.5: accuracy 0.474, precision 0.575, recall 0.396, F1 0.450.
  - Per-label thresholds: F1 0.541.
  - An [ONNX/INT8 version](https://huggingface.co/SamLowe/roberta-base-go_emotions-onnx) exists, which matters for on-device use.
- **[`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)**. Ekman 6 + neutral, trained on six datasets including GoEmotions and SemEval-2018. Evaluation accuracy 66% on its own balanced split. The card names **no license**, so use it as a comparison only and do not redistribute it.

## Stance taxonomies: defensive, withdrawn, reflective

| Source | Relevant codes | Data / agreement | Fit |
|---|---|---|---|
| **MISC 2.5** ([manual](https://casaa.unm.edu/assets/docs/misc25.pdf)) | Every client utterance gets one of three mutually exclusive codes: **Follow/Neutral/Ask**, **Change talk**, **Sustain talk**. Change language is sub-coded Reason, Desire, Ability, Need, Commitment, Taking Steps, Other, with +/− valence. | Manual only. It says summary scores help when individual categories "demonstrate low reliabilities". | Best-documented definitions for *reflective* (change talk) and the *minimising* half of *defensive* (sustain talk). |
| **MITI 4.2.1** ([manual](https://casaa.unm.edu/assets/docs/miti4_2.pdf)) | Codes the clinician only, so it has no client codes. The key rule: the target behaviour must be known in advance, because with abstract goals "coders cannot reliably discern the change talk". | n/a | For this project, set the target behaviour per scenario to *a safer online step for the triggering risk*. |
| **Discord** (Miller & Rollnick, *MI*, 3rd ed., 2013) | Splits "resistance" into sustain talk (about the behaviour) and discord (friction with the helper) | Book not inspected. Only secondary training material, e.g. [ATTC *Talking to Change*](https://attcnetwork.org/wp-content/uploads/2024/07/TalkingtoChange_57.pdf). **Unverified** at primary level. | Covers "stop asking", "not my fault" said at the agent. |
| **AnnoMI** ([repo](https://github.com/uccollab/AnnoMI); [ICASSP 2022](https://doi.org/10.1109/ICASSP43922.2022.9746035), [Future Internet 2023](https://doi.org/10.3390/fi15030110)) | `client_talk_type` ∈ {change, neutral, sustain} | 133 transcripts of professionally produced MI demonstration videos. I counted **9,699 utterances, 4,817 client: 3,102 neutral, 1,174 change, 541 sustain** in `AnnoMI-simple.csv`. Authors state consent from the video owners ([PhilHumans, arXiv 2405.02770](https://arxiv.org/abs/2405.02770)). GitHub reports **no license file** ([API](https://api.github.com/repos/uccollab/AnnoMI)). Agreement is reported as Fleiss' κ with ICC as a supplement, but the MDPI page returned 403, so the **values are unverified**. | Only public expert-labelled client-talk corpus found. Adult, mostly health behaviours, acted. PhilHumans reports client talk type is predicted less well than therapist behaviour. No official client-talk classifier is on HF; I found only third-party generative fine-tunes. |
| **Adolescent MI** ([Baer et al. 2008](https://doi.org/10.1037/a0013022)) | Commitment, reasons, desire/ability statements | 54 homeless adolescents; statements about desire or ability *against* change predicted worse substance-use outcomes | Evidence that MI client-language coding is applied to and meaningful for adolescents. |
| **Client Resistance Code** ([Chamberlain et al. 1984](https://doi.org/10.1016/S0005-7894(84)80016-7)) | Resistance in parent-training sessions | Metadata only. Codes **unverified**. | Historical precedent. MISC/discord is the more current scheme. |
| **NICHD protocol / child reluctance** ([Lamb et al. 2007](https://doi.org/10.1016/j.chiabu.2007.03.021); [Hershkowitz et al. 2006](https://doi.org/10.1016/j.chiabu.2005.10.016); [Ahern et al. 2019](https://doi.org/10.1080/10888691.2017.1387057)) | Non-disclosers were "uncooperative" and gave fewer details; younger children were more reluctant; interviewer support linked to disclosure. | Forensic interviews, ages 3–13. Coding schemes **unverified** (abstracts only). | Supports keeping **disengaged** separate from defensive, and not pressing a reluctant child. This also matches the FSM's refusal exit. |

**Disengaged** is the one project-specific code. MISC has no client code for non-answers ("idk", "nothing", refusals), only "No Code" for uncodeable speech. Define it explicitly and report it as an extension, not as MISC.

## Distress: definitions and datasets

| Resource | Definition of the label | Agreement | Access | Use here |
|---|---|---|---|---|
| **ReachOut triage, CLPsych 2016** ([paper](https://aclanthology.org/W16-0312/)) | green / amber / red / crisis, assigned by a decision tree with branches including *current acute distress*, *current mild distress* and *past distress*. "Crisis" means imminent risk of harm. | **Fleiss' κ 0.706**; pairwise **Cohen's κ** 0.674–0.761 (3 judges, 977 posts) | By application. 65,024 posts from a peer-support forum for **14–25-year-olds** | The closest age match and the best model for a severity rule. The crisis class appeared once in the test set, a warning about rare-class evaluation. |
| **UMD Reddit suicidality** ([Shing et al. 2018](https://aclanthology.org/W18-0603/); [CLPsych 2019](https://aclanthology.org/W19-3003/)) | No / low / moderate / severe risk. Long rubric covers thoughts, feelings, logistics, context. Never downgrade a judgement. | Experts: **Krippendorff's α** 0.812 with the long rubric (0.768 with short instructions); crowd α 0.554 | Governed by the American Association of Suicidology | Borrow the rubric structure and the never-downgrade rule for self-harm language. Do not train on it. |
| **ESConv** ([ACL 2021](https://aclanthology.org/2021.acl-long.269/)) | The help-seeker rates their own emotion intensity 1–5 before and after (mean 4.04 → 2.14). Categories include anxiety, depression, sadness, anger, fear, shame. | Self-report, so no inter-rater statistic | Academic research only (per the [19 Oct note](pp2_dataset_methodology_research_2026-10-19.md)) | Precedent for an intensity scale. Adult crowd workers. |
| **Dreaddit** ([2019](https://aclanthology.org/D19-6213/)) | Binary stress over ~100-token Reddit segments | Not checked | n/a | Long adult posts. Little transfer to 1–4 short replies. |

Implication: treat **distressed** as a severity judgement written into the codebook, not as "fear or sadness detected". A sad child reflecting calmly on a past event is *past distress*: on the emotion axis that is sadness, and the label is not `distressed`. Fear or sadness about the present, overwhelm, or any self-harm language is *current* distress. The never-downgrade rule applies across the session.

## Evaluation design

**Comparison arms.** Run every arm on the same frozen held-out child IDs:

1. Current phrase rules. On simulated data they reached 315/372 = 84.7% agreement, with 75/76 distressed caught. They also labelled 13 calm and 18 unclear sessions as distressed ([results](C:/Users/User/Downloads/child-safety-system/child-safety-system/component3_socratic_educator/evidence/evaluate/emotion_rules_eval_2026-10-14/after-all-fixes_heldout.txt)).
2. **Off-the-shelf:** SamLowe GoEmotions ONNX, mapped to Ekman groups with the paper's mapping. It can supply axis A only. Distress and stance still need rules or a prompt, so report what each component contributes.
3. The on-device Gemma, prompted with the codebook definitions (zero- or few-shot).
4. **Fine-tuned small classifier:**
   - Axis A pre-trained on GoEmotions (Apache 2.0).
   - Axis B pre-trained on AnnoMI client talk, only after confirming reuse terms with the authors, since no license file is present.
   - Then adapted on the project's human-annotated child-reply set. That set must be the only source of test data.

DailyDialog and EmpatheticDialogues are non-commercial. That is acceptable for an academic project, but record it in the datasheet.

**Metrics.** Report:

- **distressed recall** and its false-negative count first, with the session IDs of misses;
- per-class precision/recall and macro-F1, with a confusion matrix;
- escalation false-positive rate.

Report these separately from JSON validity and latency. Group splits by child and scenario.

## What the annotation codebook should cite

- **Emotion axis:** GoEmotions definitions (Appendix A) and its Ekman grouping ([paper](https://aclanthology.org/2020.acl-main.372/)), plus EmoContext's use of preceding turns as context ([paper](https://aclanthology.org/S19-2005/)).
- **Stance axis:** MISC 2.5 Follow/Neutral, change/sustain definitions and sub-codes ([manual](https://casaa.unm.edu/assets/docs/misc25.pdf)). Use the MITI 4.2.1 rule to fix the target behaviour per scenario ([manual](https://casaa.unm.edu/assets/docs/miti4_2.pdf)). Define *disengaged* and *discord* explicitly as project definitions, with worked examples.
- **Distress:** ReachOut decision-tree branches ([Milne et al.](https://aclanthology.org/W16-0312/)) and the Shing et al. risk-factor families with the never-downgrade rule ([paper](https://aclanthology.org/W18-0603/)).
- **Procedure:** a calibration pilot before full annotation, and adjudication that preserves disagreements (as in [Said & Villaneau](https://aclanthology.org/2026.cas-1.11.pdf)). LLM agreement must not stand in for a human rater.
- **Agreement statistic:** report **Krippendorff's α** per axis: nominal for emotion and stance, ordinal for distress severity. Add pairwise Cohen's κ for two raters, per-class agreement, and the confusion between raters ([Artstein & Poesio 2008](https://aclanthology.org/J08-4004/)). Set no pass threshold in advance. Published emotion agreement on short text ranges from κ 0.21 to 0.59 (above), and the 0.8 convention cited by Shing et al. applies to their expert rubric, not to this task.

## Limitations

- **Age and domain gap.** Every labelled English dataset above is adult: Reddit, Twitter, MTurk, ESL scripts or acted counselling. ReachOut (14–25) is the closest and requires an access application. None reflects Sri Lankan English or code-mixing.
- **Synthetic evaluation.** The current held-out set is simulated, and the rules were written against similar development data. Its 75/76 distress catch says nothing about real children. A new human-annotated set is required, and even that remains synthetic unless ethically approved real replies are collected.
- **Short input.** EVALUATE sees 1–4 short replies. MISC and AnnoMI code utterances inside long sessions, so agreement there will not transfer. Measure it on your own data.
- **Raters.** One owner plus model suggestions is not two independent human raters (as in the 19 Oct note).
- **Unverified items:** AnnoMI agreement values and license, the discord primary text, Plutchik content, the SemEval-2018 data license, the coding schemes of the reluctance studies, and ReachOut's best system score (the table did not extract cleanly).

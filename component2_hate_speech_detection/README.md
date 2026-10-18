# Component 2 — On-Device Hate Speech & Harmful Content Detection

**Project:** J26-DS-316 — Fully On-Device AI for Detecting Harmful Content and Guiding Children's Digital Safety  
**Owner:** Liyanage D. S. (IT23209152)  
**Proposal:** `IT23209152.pdf` (Hate Speech Analyzer)  
**Status (18 Oct 2026):** roadmap steps 0–8 and 10 done; the meme score is evidence only until it passes its calibration gate. Step 9 (results tables) is partly done.

---

## What this component does

C2 decides whether content on a child's device is harmful, how serious it is for that child's **age band (8–15)**, and why. It runs on an ordinary **CPU laptop with no network** during inference, and never imports torch at runtime (ONNX Runtime only).

| Source | How it becomes something C2 can score |
|---|---|
| Chat / typed messages | Direct text |
| On-screen text | Change-gated OCR (PP-OCRv6 tiny) into positioned message blocks |
| Voice chat | Speech-to-text (Whisper tiny INT8) with voice-activity detection |
| Memes / images | Pictures found on screen, scored by the meme model (CLIP ViT-B/32 + align head). **Evidence only**: it cannot change a decision until it is calibrated |

When risk is confirmed, it emits a **trigger** for Component 3 (Socratic Educator) and a **record** for Component 4 (parent-facing profiling). The **flagged words are never passed to C3**.

### What C2 does *not* do

| Out of scope | Owner |
|---|---|
| Screen / audio capture for the full system | Component 1 (C2 has stand-in capture for its own app) |
| Socratic dialogue with the child | Component 3 |
| Long-term behavioural profile / parent dashboard UI | Component 4 |
| Training new models from scratch | Not in this project (only small heads on frozen models) |
| Real-time video analysis | Not in this project |
| Languages other than English | Limitation / future work |

---

## Place in the system

```
C1 (capture) ──► C2 (this component: read + detect + explain) ──► C3 (dialogue)
                                                           └──► C4 (parent record)
```

Everything is in-process Python on one laptop. The team pipeline calls `analyze_text()` and passes dicts that match `docs/interface-contracts/`. C2 also runs on its own as the **C2 Analyst** desktop app (tray icon, live screen and audio monitoring, a "try it" panel) for demos and measurement.

---

## Design rules (must not regress)

The research novelty of C2 is three rules, each tested and each switchable for ablation:

1. **Corroboration.** Two pretrained text models that fail on different cases (RoBERTa Dynabench R4 and toxic-bert) must agree, or one must be very confident, before a score rises. One model's false alarm cannot raise an alert alone.
2. **Framing.** Before a score rises, check whether the child is *reporting, quoting or objecting to* harm rather than doing it. If so, hold the score below every alert threshold (HateCheck-style failure class).
3. **Evidence may raise, never lower.** A fused score never falls below the strongest calibrated signal, and an uncalibrated signal is shown as evidence but never moves the score.

Every verdict carries an **explanation built at decision time**: matched rules, both model scores, the framing outcome, the age threshold used and the final category.

Invariants with named tests:

| ID | Rule | Test |
|---|---|---|
| I-1 | Only calibrated signals fuse | `test_fusion.py` |
| I-2 | Fusion never lowers the strongest signal | `test_fusion.py` |
| I-3 | Framing is a discount, never a veto; self-harm and grooming are never discounted | `test_framing.py` |
| I-4 | An abstained item never goes above L1 | `test_decision.py` |
| I-5 | The decision is a pure function of its inputs | `test_decision.py` |
| I-6 | In a game context, model scores are capped so trash talk does not alert | `test_text_scorer.py` |

---

## Pipeline

| Stage | What happens | Code |
|---|---|---|
| 1 Capture | Frames and audio held in memory only; the raw frame is closed after each check | `src/capture/`, `src/agent/runtime.py` |
| 2 Read | Screen: a tiny thumbnail diff decides if anything changed; only changed areas are OCR'd. Voice: utterances cut by VAD, then transcribed. Pictures: found on changed frames and scored once each | `src/readers/` |
| 3 Screen | Keyword layer (hash-verified policy pack) + two text models with the corroboration rule; framing check; gaming cap | `src/engine/lexicon.py`, `text_scorer.py`, `framing.py` |
| 4 Gate | Below 0.35 the check stops (keeps average CPU low) | `src/engine/analyzer.py` |
| 5 Confirm and combine | Long text re-read in windows; fusion under I-1 and I-2 | `src/engine/fusion.py` |
| 6 Decide by age | Persona threshold and rung; verdict with evidence, explanation and child-safe summary | `src/engine/decide.py`, `verdict.py` |

### Decision by age

| Persona | Ages | Alert threshold |
|---|---|---|
| P1 Protect | 8–10 | 0.55 |
| P2 Guide | 11–13 | 0.65 |
| P3 Respect | 14–15 | 0.75 |

Self-harm, sexual content and grooming use one stricter threshold (0.50) for every age. Self-harm and grooming are always urgent (immediate parent alert); a threat is urgent at 0.90 or above. Response rungs: **L0** record · **L1** inform · **L2** soft blur · **L3** hard blur + C3 conversation · **L4** full block · **L5** close app. All thresholds live in `packs/policy/decision.json`.

---

## Interface contract (source of truth)

**Outbound trigger** validates against `docs/interface-contracts/comp2_to_comp3.schema.json`.

| Field | Rules |
|---|---|
| `session_id` | UUID shared across the incident |
| `timestamp` | ISO-8601 datetime |
| `source_component` | Always `"component2_hate_speech_detection"` |
| `content_type` | `text_message` \| `voice_transcript` |
| `risk_category` | `hate_speech` \| `cyberbullying` \| `grooming_language` \| `self_harm_language` |
| `confidence_score` | Number in `[0, 1]` |
| `context_metadata` | C2 category, rung, recommended action, urgency, persona, child-safe summary, matched rule IDs, framing reason, versions — **no flagged words** |

C3 is woken only for rungs that blur or block (L2+). `matched_families` carries keyword family IDs (`bullying:exclusion`) or, for a literal-phrase hit, `phrase:<category>`: never the phrase itself, since the phrase list includes slurs (tested for every listed phrase).

### Categories → schema mapping

C2 uses Guardian's finer categories internally and maps them to the team enum: `hate_identity` → `hate_speech`; `bullying`, `threat`, `sexual_harassment` → `cyberbullying`; `grooming` → `grooming_language`; `self_harm` → `self_harm_language`. `threat` and `sexual_harassment` have no team enum yet (not yet proposed in `docs/interface-contracts/CHANGELOG.md`); the fine category travels in `context_metadata.c2_category`.

### Public entry point (stable)

```python
from component2_hate_speech_detection.src.component2 import analyze, analyze_text

payload = analyze_text()                                  # mock fixture: the team pipeline's default
payload = analyze_text("nobody likes you", age=10)        # real engine: schema-valid trigger, or None
verdict = analyze("nobody likes you", age=10)             # the full Verdict: rung, evidence, explanation
```

`C2_USE_HEADS=0` runs the keyword layer only (fast tests, ablation).

---

## Implementation roadmap

| Step | Deliverable | Maps to | Status |
|---|---|---|---|
| 0 | Stub + mock trigger so the end-to-end demo works | Integration | Done |
| 1 | Runtime validation against the shared JSON schema + tests | FR9, NFR9 | Done |
| 2 | Auditable keyword / pattern layer (explainable matches) | SO1 | Done |
| 3 | Two pretrained on-device scorers + corroboration | SO1, FR3 | Done (ONNX; INT8 only where no decision changes) |
| 4 | Framing check (report / quote / condemn vs commit) | SO2, FR4 | Done |
| 5 | Escalation gate + age-band thresholds | SO4, FR6 | Done |
| 6 | Evidence / explanation record + child-safe summary | FR7, FR8 | Done |
| 7 | Screen-text (OCR) and speech (ASR) readers into the same scorer | SO3, FR1 | Done |
| 8 | Image / meme path, calibrated before it can change decisions | SO3, FR5 | Done, **evidence only**: failed the calibration gate on the test set (p = 0.008) |
| 9 | Child-context test set, baselines, ablations, resource / network checks | RQ1–RQ3 | Partly: dev (67) and held-out (40) text sets, meme and screen evaluations, ONNX parity. To do: baseline and ablation tables, resource table, child-context set |
| 10 | Real C2 path in `integration/` | Shared | Done (`--source component2 --text`) |

---

## Measured so far

On the reference laptop (Intel i7-1355U, CPU only). Result files are in `evaluation/results/`.

| What | Result |
|---|---|
| Typed message, keyword layer + both text models | 19 ms median, 24 ms p95 (107 cases); 755 MB loaded |
| On-screen chat text (5 chat styles, 107 cases) | 97.5–100 % same decision as typed text (94–97 % before the chat-header fix); benign alerts on screen equal typed ones |
| Image checked in the app (OCR + text models + meme model) | 0.42 s median, 0.81 s p95; peak 1,508 MB (budget 2 GB) |
| Meme model, test (dev_seen, 500 memes) | 0.751 AUROC (logistic-regression baseline 0.675); 0.733 with OCR captions |
| Memes inside a full screenshot | 0.713 AUROC with the picture finder vs 0.565 scoring the whole screen |
| Network during inference | 0 connections from the engine (shown live in the app) |

Details: `meme_model/README.md` (model choice, calibration, ONNX, screen memes) and `tools/export_onnx.py` (text-model INT8 decision gate).

---

## Folder layout

```
component2_hate_speech_detection/
├── README.md, requirements.txt
├── packs/policy/            keyword, framing and decision rules (hash-verified manifest)
├── src/
│   ├── component2.py        public API for the team pipeline
│   ├── engine/              lexicon, text heads (ONNX), scorer, framing, fusion, decide, verdict, meme scorer
│   ├── readers/             OCR, screen reader, picture finder, speech-to-text, utterance segmenter
│   ├── capture/             stand-in screen/audio capture and device context
│   ├── agent/runtime.py     monitoring loops, de-duplication, stats
│   └── app/                 C2 Analyst desktop app (pywebview + tray)
├── meme_model/              meme classifier: embed, heads, train, calibrate (see its README)
├── annotation/              meme labels and the human-validation tools (codebook, datasheet)
├── evaluation/              dev and held-out text sets, screen text and screen meme evaluations, results/
├── tools/                   ONNX export + parity gates, policy manifest builder
├── mock_inputs/             schema-shaped fixture for the mock path
└── tests/                   pytest suite (offline; models faked where needed)
```

Models (`models/`) and datasets (`data/`) are git-ignored; the export tools rebuild the models.

---

## How to run

```bash
pip install -r component2_hate_speech_detection/requirements.txt
pytest component2_hate_speech_detection/tests
```

C2 Analyst desktop app (screen + audio monitoring, "try it" panel):

```bash
python -m component2_hate_speech_detection.src.app.main --age 10
```

Team pipeline with a real C2 detection (C2 → C3 → C4):

```bash
python integration/end_to_end_pipeline.py --source component2 --text "nobody likes you, just leave the group" --age 10
```

Rebuild the ONNX models (needs torch + transformers once, never at runtime):

```bash
python -m component2_hate_speech_detection.tools.export_onnx
python -m component2_hate_speech_detection.tools.export_meme_onnx
```

---

## Evaluation plan (step 9)

Every version is compared with the **same harness** on the **same splits**:

| Baseline | Purpose |
|---|---|
| Keyword layer only | Value added by the models |
| Single pretrained classifier | The "simple local filter" comparison |
| Full cascade, then with corroboration / framing / gaming cap switched off | Ablation of each design rule |

Primary metrics: recall on harmful content, false alerts on benign and on **reporting** language, macro-F1, latency, peak RAM, network bytes (must be 0). The held-out set is never tuned on.

---

## Branching & contribution notes

- Feature branches `c2/...`; small PRs; one teammate reviews before merge to `main`
- Schema changes: update `docs/interface-contracts/` + `CHANGELOG.md` first, then code
- No detection logic in `integration/` — it only imports and calls this package

---

## Current known limitations (be explicit in demos)

- **The meme score never changes a decision yet.** It failed its calibration gate on the test set (p = 0.008); it is shown as evidence only.
- **Screen memes are measured on built screens**, not real captures; a meme with a white backdrop on a white page can be cut short.
- **On-screen text loses a little to OCR.** Chat headers (name + time) are now left out of the scored text, but OCR still misreads stretched or starred letters ("kiiiill" → "kifll", "k\*y\*s" → "kxyts"), so those are missed on screen. A sender's name shown without a time stays in the message: it cannot be told apart from a short message.
- **One known text false alert:** "my brother keeps annoying me i hate it" is flagged (L3) at every age, typed or on screen.
- **Short OCR fragments can raise false alerts** (one meme was flagged for a misread cut-off word).
- **Test sets are small:** 107 hand-written messages and 500 memes; meme labels still need a human check; no child-context benchmark yet.
- **Fusion can never lower a score (I-2)**, so the meme model cannot cancel a text false alarm on a benign meme; that is the design's safety trade-off.

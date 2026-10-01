# Component 2 — On-Device Hate Speech & Harmful Content Detection

**Project:** J26-DS-316 — Fully On-Device AI for Detecting Harmful Content and Guiding Children's Digital Safety  
**Owner:** Liyanage D. S. (IT23209152)  
**Proposal:** `IT23209152.pdf` (Hate Speech Analyzer)  
**Status:** Step 1 done — stub output is validated against the shared JSON schema at runtime. Keyword layer is next.

---

## What this component does

C2 decides whether content on a child's device is harmful, how serious it is for that child's **age band (8–15)**, and why. It runs on an ordinary **CPU laptop with no network** during inference.

It scores:

| Source | How it becomes text |
|---|---|
| Chat / typed messages | Direct text |
| On-screen text | OCR (planned) |
| Voice chat | ASR / transcript (planned) |
| Memes / images | Image path after calibration only (planned) |

When risk is confirmed, it emits a **detection event** for Component 3 (Socratic Educator) and a **stored record** for Component 4 (parent-facing profiling). The **verbatim harmful text is never passed downstream**.

### What C2 does *not* do

| Out of scope | Owner |
|---|---|
| Screen / audio capture | Component 1 |
| Socratic dialogue with the child | Component 3 |
| Long-term behavioural profile / parent dashboard UI | Component 4 |
| Training new models from scratch | Not in this project |
| Real-time video analysis | Not in this project |
| Languages other than English | Limitation / future work |

---

## Place in the system

```
C1 (capture) ──► C2 (this component: detect + score + explain) ──► C3 (dialogue)
                                                      └──► C4 (parent record)
```

Everything is in-process Python on one laptop. No servers, no message queues. Integration calls `analyze_text()` (and later richer entry points) and expects dicts that match the shared schemas in `docs/interface-contracts/`.

---

## Design rules (must not regress)

These three rules are the research novelty of C2. Every later step must preserve them; they are tested by ablation.

1. **Corroboration** — Two pretrained text models (strong on different failure modes) must agree, or one must be very confident, before a risk score rises. One model's false alarm cannot raise an alert alone.
2. **Framing** — Before a score rises, check whether the child is *reporting, quoting, or objecting to* harm rather than committing it. If so, hold the score low (HateCheck-style failure class).
3. **Evidence may raise, never lower** — When combining signals (text + image, etc.), the fused score must never fall below the strongest reliable signal alone. Weak/uncalibrated signals can support a detection but never cancel one.

Every detection must carry an **explanation built at decision time**: matched keywords, model scores, framing outcome, age threshold used, final category.

---

## Target pipeline (six stages)

| Stage | Name | Behaviour |
|---|---|---|
| 1 | Capture hand-off | Receive text / frame / audio references held in memory; nothing raw written to disk by C2 |
| 2 | Read | Convert screen text, speech, and (later) images into scorable text; **one text scorer for all sources** |
| 3 | Screen | Keyword layer + two pretrained models; corroboration required |
| 4 | Gate | Below escalation threshold → stop and clear (keeps average cost low on CPU) |
| 5 | Confirm & combine | Re-read longer text; consult calibrated image path; fuse under “raise, never lower” |
| 6 | Decide by age | Compare final score to age-band threshold; emit category, confidence, child-safe summary, evidence |

---

## Interface contract (source of truth)

**Outbound trigger** must validate against:

`docs/interface-contracts/comp2_to_comp3.schema.json`

| Field | Rules |
|---|---|
| `session_id` | UUID shared across the incident |
| `timestamp` | ISO-8601 datetime |
| `source_component` | Always `"component2_hate_speech_detection"` |
| `content_type` | `text_message` \| `voice_transcript` |
| `risk_category` | `hate_speech` \| `cyberbullying` \| `grooming_language` \| `self_harm_language` |
| `confidence_score` | Number in `[0, 1]` |
| `context_metadata` | Optional; e.g. `platform`, `language_detected` — **no raw flagged text** |

Privacy rule: carry **category + score + metadata**, never the verbatim harmful string. Component 3 must not re-expose flagged content to the child.

### Proposal categories → schema mapping

Proposal FR2 names (threat, bullying, identity-based hate, sexual harassment, strong language) are research labels. **Repo schema enums above are what integration accepts.** Map internal fine labels onto schema `risk_category` values before returning. If the team needs new enums, update the schema + `docs/interface-contracts/CHANGELOG.md` *before* coding against them.

### Public entry point (stable)

```python
from component2_hate_speech_detection.src.component2 import analyze_text

payload = analyze_text()  # stub today; later: analyze_text(text=..., age_band=..., ...)
# payload must pass jsonschema against comp2_to_comp3.schema.json
```

Keep this return shape stable so C3/integration do not need rewrites when internals change.

---

## Implementation roadmap (build in this order)

Do **not** jump to heavy models or OCR first. Proposal Section 4.6 order:

| Step | Deliverable | Maps to | Status |
|---|---|---|---|
| 0 | Stub + mock trigger so end-to-end demo works | Integration | Done |
| 1 | Runtime validation against shared JSON schema + tests | FR9, NFR9 | Done |
| 2 | Auditable **keyword / pattern layer** (explainable matches) | SO1 (first half) | Next |
| 3 | Wire **two pretrained** on-device scorers + **corroboration** rule | SO1, FR3 | Planned |
| 4 | **Framing check** (report / quote / condemn vs commit) | SO2, FR4 | Planned |
| 5 | Escalation **gate** + age-band thresholds | SO4, FR6 | Planned |
| 6 | Evidence / explanation record + child-safe summary | FR7, FR8 | Planned |
| 7 | Screen-text (OCR) and speech (ASR) readers into the same scorer | SO3, FR1 | Planned |
| 8 | Image / meme path — **calibrate before** it can change decisions | SO3, FR5 | Planned |
| 9 | Child-context test set, baselines, ablations, resource/network checks | RQ1–RQ3 | Planned |
| 10 | Swap stub in `integration/` once real path is schema-stable | Shared | Planned |

Each step should be a small PR on a `c2/...` branch, with tests that fail loudly if the schema or a design rule breaks.

---

## Folder layout

```
component2_hate_speech_detection/
├── README.md                 ← this file
├── requirements.txt          ← add deps per step; keep CPU-friendly
├── mock_inputs/              ← schema-shaped fixtures for isolated work
│   └── sample_analysis.json
├── src/
│   └── component2.py         ← public API (stub today)
│   # planned as steps land:
│   # keywords.py             ← auditable pattern layer
│   # models.py               ← pretrained scorers + corroboration
│   # framing.py              ← report/quote/condemn gate
│   # fuse.py                 ← raise-never-lower combination
│   # decide.py               ← age bands + explanation
│   # readers/                ← OCR / ASR adapters (later)
└── tests/
    └── test_component2.py    ← grow with each step; always schema-check outputs
```

Develop against `mock_inputs/` and the shared schema. You should not need C1/C3/C4 real implementations to test C2.

---

## Non-functional targets (proposal Table 18)

| ID | Requirement | Target |
|---|---|---|
| NFR1 | CPU-only laptop | No dedicated GPU required |
| NFR2 | Model memory budget | ~2 GB total loaded |
| NFR3 | Latency | A few seconds per check; measure, don't assume |
| NFR4 | Network during inference | **Zero** bytes |
| NFR5 | Raw frames / audio | Memory only; cleared after check |
| NFR8 | Partial failure | One reader failing must not kill the whole check |
| NFR9 | Reproducibility | Fixed splits; harness in-repo |

---

## How to run

### Tests (component only)

```bash
pip install -r component2_hate_speech_detection/requirements.txt
pytest component2_hate_speech_detection/tests -v
```

### Full system demo (C2 as trigger source)

```bash
python run_demo.py --source component2
```

### Integration tests

```bash
pytest integration/integration_tests -v
```

---

## Evaluation (when the cascade exists)

Compare every version with the **same harness** on the **same splits**:

| Baseline | Purpose |
|---|---|
| Keyword layer only | Shows value added by models |
| Single pretrained classifier | Main “simple local filter” comparison |
| (Optional) cloud API on same items | Privacy contrast — not used at runtime |

Primary metrics: recall on harmful content, false-positive rate on benign, false alerts on **reporting** language, macro-F1, processing time, peak RAM, network bytes (must be 0).

Child-context test set and ablations of the three design rules are owned by this component (see proposal Sections 4.2–4.5).

---

## Branching & contribution notes

- Feature branches: `c2/schema-validation`, `c2/keyword-layer`, `c2/cascade-v1`, …
- Small PRs; one teammate reviews before merge to `main`
- Schema changes: update `docs/interface-contracts/` + `CHANGELOG.md` first, then code
- Do not put detection logic in `integration/` — only import and call this package

---

## Current known limitations (be explicit in demos)

- `analyze_text()` still loads `mock_inputs/sample_analysis.json`; it does not classify live text yet.
- Schema validation is real: invalid payloads raise before leaving C2.
- OCR, ASR, image path, framing, corroboration, and age bands are **designed but not implemented**.
- No child-context evaluation set checked in yet.
- Prototype evidence in the proposal appendix was exploratory (pretrained, no fine-tune); measured claims come after the evaluation harness lands.

When a step lands, update the **Status** column in the roadmap table and shorten this section so demos stay honest.

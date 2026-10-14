# Component 1 Output Schema

**File:** `docs/interface-contracts/component1_output_schema.md`  
**Component:** Component 1 — Screen Monitoring  
**Owner:** IT23377844  
**Status:** Implemented — mirrors `comp1_to_comp3.schema.json`  
**Last updated:** 2026-10-07

---

## Overview

Component 1 emits a **TriggerPayload** each time a captured screen frame is
**flagged** by the fine-tuned violence detector. The machine-readable contract
is [`comp1_to_comp3.schema.json`](comp1_to_comp3.schema.json); this document
explains it. If the two ever disagree, **the JSON schema wins**.

- A frame that is **not flagged produces no payload** — the schema describes a
  risk trigger and has no "safe" category.
- No raw image, frame buffer, or file path is ever included — only
  classification metadata.
- Every payload is validated with `jsonschema` (Draft 7, `date-time` format
  checked) before it is emitted (`src/component1.py: validate_payload`).

---

## Fields

| Field | Type | Required | Value emitted by Component 1 |
|---|---|---|---|
| `session_id` | string | ✅ | UUID v4, one per incident (a run of consecutive flagged frames); shared by all 4 components for that incident. |
| `timestamp` | string (`date-time`) | ✅ | Frame capture time, UTC ISO 8601, e.g. `2026-10-07T09:13:18.087781+00:00`. |
| `source_component` | const | ✅ | `"component1_screen_monitoring"` |
| `content_type` | enum: `image`, `video_frame`, `on_screen_text` | ✅ | `"image"` (each sample is a screenshot). |
| `risk_category` | enum: `explicit_visual`, `violence`, `self_harm_imagery`, `unknown_flagged` | ✅ | `"violence"` (only the violence model is live in PP1). |
| `confidence_score` | number, 0–1 | ✅ | Detector's P(violent); always ≥ the flag threshold (`config.THRESHOLD`, default 0.5). |
| `context_metadata.app_or_window` | string | — | Not emitted yet. |
| `context_metadata.duration_visible_ms` | integer | — | ms since the incident's first flagged frame (0 on the first payload). |

---

## Example Payload

```json
{
  "session_id": "3f1c2a8e-5b7d-4e0a-9c61-2d8f4b9e7a10",
  "timestamp": "2026-10-07T09:13:20.112000+00:00",
  "source_component": "component1_screen_monitoring",
  "content_type": "image",
  "risk_category": "violence",
  "confidence_score": 0.93,
  "context_metadata": { "duration_visible_ms": 2000 }
}
```

---

## Superseded draft (2026-09-29)

The first draft of this document described a different shape — `flagged`,
`category` (incl. `"safe"`, `"hate_speech"`, `"self_harm"`), `confidence`,
`timestamp`, `frame_reference`. That shape was **never part of the agreed
contract** and is not emitted. Mapping, for anyone who used it:

| Old draft field | Now |
|---|---|
| `flagged` | implicit — a payload is only emitted when flagged |
| `category` | `risk_category` (schema enum; no `"safe"`) |
| `confidence` | `confidence_score` |
| `timestamp` | `timestamp` |
| `frame_reference` | dropped — no reference to raw frames leaves Component 1 |

---

## Change Log

| Date | Author | Change |
|---|---|---|
| 2026-09-29 | IT23377844 | Initial draft — defines five output fields. |
| 2026-10-07 | IT23377844 | Rewritten to mirror `comp1_to_comp3.schema.json`; old draft marked superseded. Real payloads now emitted from the fine-tuned detector. No schema change. |

# Component 1 Output Schema

**File:** `docs/interface-contracts/component1_output_schema.md`  
**Component:** Component 1 — Screen Monitoring  
**Owner:** IT23377844  
**Status:** Draft  
**Last updated:** 2026-09-29

---

## Overview

This document defines the canonical output payload that Component 1 (Screen Monitoring)
must emit after classifying a captured screen frame. Every downstream consumer
(Component 3, logging sinks, dashboards) must treat these fields as the stable
contract — internal implementation details of Component 1 may change freely as
long as the shape below is preserved.

---

## Output Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `flagged` | `bool` | ✅ | `true` if the frame was classified as containing harmful or policy-violating content, `false` otherwise. |
| `category` | `string` | ✅ | The primary content category assigned to the frame. Must be one of the enumerated values below. |
| `confidence` | `float` | ✅ | Model confidence in the assigned `category`, in the range `[0.0, 1.0]`. |
| `timestamp` | `string` | ✅ | ISO 8601 UTC timestamp of when the frame was captured / classified. Format: `YYYY-MM-DDTHH:MM:SS.ssssss+00:00`. |
| `frame_reference` | `string` | ✅ | Unique, stable identifier for the captured frame (e.g. a file path or UUID-based key). Used by downstream components to retrieve or log the original image. |

---

## Field Details

### `flagged` — `bool`

- `true`  → frame requires downstream action (alerting, logging, intervention).
- `false` → frame is considered safe; no action required.
- Downstream components **must not** take action solely on `category` without also
  checking `flagged`; a high-risk category with low `confidence` may still be `false`.

### `category` — `string` (enum)

Allowed values:

| Value | Meaning |
|---|---|
| `"explicit_visual"` | Sexually explicit imagery. |
| `"violence"` | Graphic violence or gore. |
| `"hate_speech"` | Hateful symbols, text overlays, or gestures. |
| `"self_harm"` | Content depicting or promoting self-harm. |
| `"safe"` | No policy violation detected. |

> [!NOTE]
> New categories must be agreed across all component owners before being added.
> Consumers should handle unknown category strings gracefully (treat as `"safe"`
> if `flagged` is `false`, or escalate to human review if `flagged` is `true`).

### `confidence` — `float`

- Range: `0.0` (no confidence) → `1.0` (maximum confidence).
- Values outside `[0.0, 1.0]` are invalid and must be rejected by consumers.
- Recommended thresholds (non-binding):

  | Confidence range | Suggested consumer behaviour |
  |---|---|
  | `≥ 0.85` | High confidence — act immediately. |
  | `0.65 – 0.84` | Medium confidence — log and queue for review. |
  | `< 0.65` | Low confidence — log only; do not escalate automatically. |

### `timestamp` — `string` (ISO 8601)

- Always UTC (`+00:00` or `Z` suffix).
- Producers must use `datetime.now(timezone.utc).isoformat()` (Python) or equivalent.
- Consumers must parse this as a timezone-aware datetime; naive datetimes must be rejected.

### `frame_reference` — `string`

- Must be unique within a session.
- Recommended format: `frame_<unix_epoch_ms>` or a UUID v4.
- Must not contain whitespace or special shell characters (safe subset: `[A-Za-z0-9_\-.]`).
- Producers must guarantee the referenced resource (file / buffer) is accessible for
  at least 60 seconds after emission, to allow async consumers to retrieve it.

---

## Example Payload

```json
{
  "flagged": true,
  "category": "explicit_visual",
  "confidence": 0.92,
  "timestamp": "2026-09-29T07:30:00.123456+00:00",
  "frame_reference": "frame_1759128600123"
}
```

---

## Validation

The JSON Schema counterpart of this document lives at:

```
docs/interface-contracts/comp1_to_comp3.schema.json
```

All output from Component 1 is automatically validated against that schema in
`tests/test_component1.py` before being forwarded to Component 3.

---

## Change Log

| Date | Author | Change |
|---|---|---|
| 2026-09-29 | IT23377844 | Initial draft — defines five output fields. |

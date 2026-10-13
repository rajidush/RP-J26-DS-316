# Interface Contract Changelog

All four component owners must agree here before a schema change is merged.
Log format: `YYYY-MM-DD | schema file | what changed | who proposed | who approved`

---

- 2026-09-23 | comp1_to_comp3.schema.json | Initial version drafted | IT23155466 | Pending team review
- 2026-09-23 | comp2_to_comp3.schema.json | Initial version drafted | IT23155466 | Pending team review
- 2026-09-23 | comp3_to_comp4.schema.json | Initial version drafted | IT23155466 | Pending team review
- 2026-10-07 | comp1_to_comp3.schema.json | Component 1 now emits real schema-valid payloads from the fine-tuned detector; no schema change. | IT23377844 | n/a — no schema change
- 2026-10-11 | comp3_to_comp4.schema.json | Interface gap logged, no schema change: Component 4 Function 1 (`emotional_pattern.py`) also requires `child_id`, which the schema does not carry. For now `integration/comp3_to_comp4.py` validates each record, then adds `child_id` (one device = one child profile; a hardcoded placeholder until a device-config source exists). Function 1 ignores `risk_level` and `dialogue_summary`. Decide as a group: add `child_id` to the schema, or keep it device-side. | IT23155466 | Pending team review (IT23135116)
- 2026-10-11 | comp1_to_comp3, comp2_to_comp3, comp3_to_comp4 .schema.json | No schema change, stricter validation: Component 3's validator now enforces the existing `"format": "date-time"` on `timestamp` (jsonschema had been skipping it; needs `rfc3339-validator`). Inbound triggers from Components 1 and 2 must carry an RFC 3339 timestamp with a UTC offset, e.g. `2026-10-08T16:00:00+00:00`; current producers and mock inputs already do. | IT23155466 | Pending team review (IT23377844, IT23209152, IT23135116)

> Add a new dated row every time a field is added, renamed, or removed. Never silently edit a shared schema — a downstream owner's code will break without warning.

# Interface Contract Changelog

All four component owners must agree here before a schema change is merged.
Log format: `YYYY-MM-DD | schema file | what changed | who proposed | who approved`

---

- 2026-09-23 | comp1_to_comp3.schema.json | Initial version drafted | IT23155466 | Pending team review
- 2026-09-23 | comp2_to_comp3.schema.json | Initial version drafted | IT23155466 | Pending team review
- 2026-09-23 | comp3_to_comp4.schema.json | Initial version drafted | IT23155466 | Pending team review
- 2026-10-07 | comp1_to_comp3.schema.json | Component 1 now emits real schema-valid payloads from the fine-tuned detector; no schema change. | IT23377844 | n/a — no schema change

> Add a new dated row every time a field is added, renamed, or removed. Never silently edit a shared schema — a downstream owner's code will break without warning.

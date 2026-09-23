# Risk Register

| Risk | Component | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|---|
| Grammar-constrained decoding library integration takes longer than planned | 3 | Medium | High | Fallback to template-based generation (already implemented, see `grammar_decoder.py` candidate pool) | Open |
| Interface schema changes break a downstream component | All | Medium | Medium | All changes logged in `docs/interface-contracts/CHANGELOG.md`, schema validation runs at runtime | Open |
| Real on-device model too slow for live demo | 3 | Low | Medium | Pre-recorded fallback run available (see Week 5 of PP1 plan) | Open |

Add a row per identified risk, across all 4 components — this doubles as
PP1 evidence pack material.

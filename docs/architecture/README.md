# Architecture Diagrams

Put the high-level system diagram (all 4 components + data flow) and each
component's internal diagram here. Diagrams are mermaid blocks in markdown,
so GitHub renders them and they are diffed like code.

- [system-data-flow.md](system-data-flow.md) — all components: Component 1/2 trigger → Component 3 → adapter → Component 4, with what is real, mock or stub today
- [fsm-state-diagram.md](fsm-state-diagram.md) — Component 3's four dialogue states and their transition conditions
- [intercept-generation-path.md](intercept-generation-path.md) — Component 3: how the INTERCEPT line is made (constrained decoding → post-check → fallback line)

`component3_socratic_educator/tests/test_architecture_docs.py` checks the
diagrams against the names and numbers in the code (FSM states and thresholds,
fallback reasons, schema and function names). It catches a renamed or changed
value, not a wrong arrow.

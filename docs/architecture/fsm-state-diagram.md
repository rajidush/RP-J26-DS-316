# Component 3 — FSM State Diagram

​```mermaid
stateDiagram-v2
    [*] --> Intercept
    Intercept --> Inquire: opening question asked
    Inquire --> Inquire: response incomplete (< 4 words) AND attempts < 3
    Inquire --> Evaluate: response complete (≥ 4 words) OR attempts ≥ 3
    Evaluate --> Contract: structured record built (risk_level, emotional_state, self_regulation_shown)
    Contract --> [*]: escalation_flag = true if risk_level == "high"
​```

**Transition conditions, explicit:**
- Intercept → Inquire: always, after one opening question
- Inquire loop: `MIN_RESPONSE_WORDS_FOR_COMPLETENESS = 4`, `MAX_INQUIRE_ATTEMPTS = 3` (see `config.py`)
- Evaluate → Contract: always, once the record validates against `comp3_to_comp4.schema.json`
- Contract → Done: escalates to parent if `risk_level == "high"`, otherwise closes normally
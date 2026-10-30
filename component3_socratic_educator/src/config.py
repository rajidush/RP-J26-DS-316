"""Tunable constants for Component 3. Keep these here (not scattered
through the code) so they're easy to point to during Q&A."""

# INQUIRE state (PP2, ticket 03): the Socratic moves, always asked in this order...
SOCRATIC_MOVES = ("clarify", "consequences", "alternatives")
# ...and the most INQUIRE questions in one session; an easier repeat of a move counts.
MAX_INQUIRE_MOVES = 3

# The child's age band selects each move's wording (ADR 0007, Component 2's bands).
# A missing or invalid band means the youngest, whose wording is the safest.
AGE_BANDS = ("8-10", "11-13", "14-15")
DEFAULT_AGE_BAND = "8-10"

# CONTRACT state: escalation_flag=True when the risk level is one of these...
ESCALATION_RISK_LEVELS = {"high"}
# ...or EVALUATE's emotional state is one of these, whatever the risk level
# (14 Oct, ADR 0005: a distressed child at moderate risk was not escalated).
ESCALATION_EMOTIONAL_STATES = {"distressed"}

# Local SLM: the model ID LM Studio serves (inference baseline,
# gemma-3-1B-it-QAT-Q4_0.gguf). See src/model_client.py.
LOCAL_MODEL_ID = "google/gemma-3-1b"

# INTERCEPT sampling temperature, sent with every live call (fixed for the plan's Mon 12 Oct task;
# before that the server's default, 0.8 per llama.cpp's log, was used: evidence/intercept/temperature_check_2026-10-13/). 1.0 is Google's
# recommended sampling temperature for Gemma 3; the post-check and the fallback
# line are the safety layers, so the adversarial test runs at this, the more
# varied setting. Lower it for more repeatable lines.
INTERCEPT_TEMPERATURE = 1.0

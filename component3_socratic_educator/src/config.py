"""Tunable constants for Component 3. Keep these here (not scattered
through the code) so they're easy to point to during Q&A."""

# INQUIRE state: how many follow-up turns before we force a move to EVALUATE
MAX_INQUIRE_ATTEMPTS = 3

# INQUIRE state: a response shorter than this (in words) is treated as
# "not yet complete" and triggers another follow-up, up to the max above.
MIN_RESPONSE_WORDS_FOR_COMPLETENESS = 4

# CONTRACT state: risk levels at or above this trigger escalation_flag=True
ESCALATION_RISK_LEVELS = {"high"}

# Local SLM: the model ID LM Studio serves (inference baseline,
# gemma-3-1B-it-QAT-Q4_0.gguf). See src/model_client.py.
LOCAL_MODEL_ID = "google/gemma-3-1b"

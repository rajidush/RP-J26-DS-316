"""
Function 2: Grammar-Constrained Decoding.

INTERCEPT (since 6 Oct, ADR 0004): the live model writes the line through LM
Studio's `response_format` JSON schema. Its llama.cpp engine compiles the
schema into a grammar and masks every disallowed token *during* generation,
so the opener is always one of the fixed safe openers and the question is
always one short plain sentence ending in "?". A grammar cannot say "never
this word", so a post-check then rejects forbidden words, alert-revealing
language and risk-category names. Any failure, or an unreachable model,
yields the pre-checked template line (the fallback line).

INQUIRE stays on the template candidate pool until after PP1.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from .schemas import RiskCategory
from .model_client import call_local_model

class ConstraintViolation(Exception):
    """Raised when a candidate response fails the grammar for its state."""


# --- The grammar: one regex pattern per FSM state -------------------------
# INTERCEPT's regex and its decoding-time JSON schema are built from the same
# pieces, so what the model is forced to produce and what validate() accepts
# cannot drift apart.
INTERCEPT_OPENERS = (
    "I noticed something on your screen just now.",
    "Something you just saw might need a closer look.",
)
# One plain sentence: capital first, then letters, spaces, commas and straight
# apostrophes only (no quotes, curly apostrophes or "..."), ending in one "?".
_QUESTION = r"[A-Z][A-Za-z ,']{5,98}\?"
INTERCEPT_SCHEMA = {
    "type": "object",
    "properties": {
        "opener": {"type": "string", "enum": list(INTERCEPT_OPENERS)},
        "question": {"type": "string", "pattern": f"^{_QUESTION}$", "maxLength": 100},
    },
    "required": ["opener", "question"],
    "additionalProperties": False,
}

_GRAMMAR = {
    "INTERCEPT": re.compile(
        "^(" + "|".join(re.escape(o) for o in INTERCEPT_OPENERS) + ") " + _QUESTION + "$"
    ),
    "INQUIRE": re.compile(
        r"^(That makes sense\.|I understand\.|Thanks for telling me\.)"
        r" (Can you say a bit more about that\?|What do you think might happen next\?|Is there another way you could\'ve reacted\?)$"
    ),
}

# Never allowed in any state output, regardless of grammar match --
# a hard safety backstop independent of the pattern check.
_FORBIDDEN_SUBSTRINGS = ["kill", "hate", "porn", "suicide", "sex"]

# Post-check for INTERCEPT: words that tell the child something was flagged,
# and the words of every risk-category name (matched as whole words, so
# "yourself" is not "self").
_ALERT_WORDS = ("alert", "alerts", "flag", "flagged", "warning", "message", "messages",
                "notification", "blocked", "detected", "detection")
_CATEGORY_WORDS = sorted({w for rc in RiskCategory for w in rc.value.split("_")})

INTERCEPT_PROMPT = (
    "You are a calm Socratic educator for child digital safety, speaking to a child aged 11 or older. "
    "Something on the child's screen was just stopped. Do not describe, name or hint at what it was, "
    "and do not mention alerts, messages or warnings. "
    "Return JSON. opener: copy one of the allowed openers exactly. "
    "question: one short, gentle, open question (under 15 words) inviting the child to share "
    "what they were doing or thinking."
)


def _has_word(words, text: str) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text, re.I) for w in words)


@dataclass(frozen=True)
class GeneratedLine:
    """A child-facing line and where it came from: "model", "template", or
    "fallback" with the reason the model's line was not used."""
    text: str
    source: str
    reason: Optional[str] = None


@dataclass
class GrammarConstrainedGenerator:
    """
    Function 2. Call .generate(state, risk_category) to get a response
    guaranteed to match that state's grammar, or raises/falls back --
    it never silently returns an unconstrained string.
    """

    # The live model call (prompt, **request options) -> (text, seconds).
    # None means src.model_client.call_local_model; tests pass a fake.
    model_call: Optional[Callable[..., tuple[str, float]]] = None

    # Template lines: INQUIRE's only source until after PP1, and INTERCEPT's
    # fallback line whenever the model's line is not used.
    _CANDIDATES = {
        "INTERCEPT": [
            "I noticed something on your screen just now. Can you tell me what you were thinking?",
            "Something you just saw might need a closer look. How are you feeling right now?",
        ],
        "INQUIRE": [
            "That makes sense. Can you say a bit more about that?",
            "Thanks for telling me. What do you think might happen next?",
            "I understand. Is there another way you could've reacted?",
        ],
    }

    def _candidate(self, state: str, risk_category: RiskCategory, attempt: int = 0) -> str:
        pool = self._CANDIDATES.get(state, [])
        if not pool:
            raise ConstraintViolation(f"No candidate pool defined for state {state!r}")
        # Deterministic pick for reproducible test sets; offset by attempt so repeated
        # inquire rounds cycle through diverse follow-up questions instead of repeating.
        base_idx = list(RiskCategory).index(risk_category)
        idx = (base_idx + attempt) % len(pool)
        return pool[idx]

    def validate(self, state: str, text: str) -> bool:
        if any(bad in text.lower() for bad in _FORBIDDEN_SUBSTRINGS):
            return False
        pattern = _GRAMMAR.get(state)
        if pattern is None:
            return False
        return bool(pattern.match(text))

    def generate(self, state: str, risk_category: RiskCategory, attempt: int = 0) -> str:
        return self.generate_with_source(state, risk_category, attempt).text

    def generate_with_source(self, state: str, risk_category: RiskCategory, attempt: int = 0) -> GeneratedLine:
        if state == "INTERCEPT":
            return self._intercept(risk_category, attempt)
        candidate = self._candidate(state, risk_category, attempt=attempt)
        if not self.validate(state, candidate):
            return GeneratedLine(self._safe_fallback(state), "fallback", "grammar")
        return GeneratedLine(candidate, "template")

    def _intercept(self, risk_category: RiskCategory, attempt: int) -> GeneratedLine:
        def fallback(reason):
            return GeneratedLine(self._candidate("INTERCEPT", risk_category, attempt=attempt), "fallback", reason)

        call = self.model_call or call_local_model   # looked up per call, so tests can block it
        try:
            # 100 tokens: the JSON wrapper plus a 48-character opener and a question of
            # up to 100 characters; a cut-off reply would fail as invalid_json.
            raw, _ = call(INTERCEPT_PROMPT, max_tokens=100, response_format={
                "type": "json_schema",
                "json_schema": {"name": "intercept_line", "strict": True, "schema": INTERCEPT_SCHEMA}})
        except Exception:
            return fallback("model_unreachable")
        try:
            reply = json.loads(raw)
        except (TypeError, ValueError):
            return fallback("invalid_json")
        if not (isinstance(reply, dict) and set(reply) == {"opener", "question"}
                and all(isinstance(v, str) for v in reply.values())):
            return fallback("grammar")
        text = f"{reply['opener'].strip()} {reply['question'].strip()}"
        # Post-check: the grammar fixes structure; these catch what it cannot express.
        if not _GRAMMAR["INTERCEPT"].match(text):
            return fallback("grammar")
        if any(bad in text.lower() for bad in _FORBIDDEN_SUBSTRINGS):
            return fallback("forbidden_word")
        if _has_word(_ALERT_WORDS, text):
            return fallback("alert_language")
        if _has_word(_CATEGORY_WORDS, text):
            return fallback("category_named")
        return GeneratedLine(text, "model")

    def _safe_fallback(self, state: str) -> str:
        fallback = {
            "INTERCEPT": "I noticed something on your screen just now. Can you tell me what you were thinking?",
            "INQUIRE": "That makes sense. Can you say a bit more about that?",
        }
        return fallback.get(state, "Let's talk about what happened.")


def run_adversarial_batch(generator: GrammarConstrainedGenerator, state: str,
                           risk_categories: list[RiskCategory]) -> dict:
    """
    Function 2's test harness. Feed it a batch of risk categories
    standing in for adversarial prompts, and it reports the
    constraint-violation rate -- the same metric your proposal targets
    at 0% across 200+ prompts (start this at ~20-30 for PP1, scale up
    after).
    """
    violations = 0
    for rc in risk_categories:
        text = generator.generate(state, rc)
        if not generator.validate(state, text):
            violations += 1
    total = len(risk_categories)
    return {
        "total_prompts": total,
        "violations": violations,
        "violation_rate": violations / total if total else 0.0,
    }

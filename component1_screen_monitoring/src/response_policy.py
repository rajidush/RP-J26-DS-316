"""
response_policy.py — Function 2 decision logic: TriggerPayload → "blur" | "block".

Pure Python (no GUI) so it can be unit-tested and reasoned about on its own.

Rules (thresholds in config.py)
-------------------------------
    risk_category in BLOCK_CATEGORIES (explicit_visual, self_harm_imagery) → block
    violence, confidence ≥ BLOCK_THRESHOLD (0.85)                          → block
    violence, confidence <  BLOCK_THRESHOLD                                → blur
    unknown_flagged                                                        → blur

``override`` ("blur" / "block") forces the action for a demo or a strict
setting; "auto" applies the rules. Every payload is validated against
comp1_to_comp3.schema.json first — an invalid payload is logged and ignored.
"""
from __future__ import annotations

import logging

try:
    from .component1 import validate_payload
    from .config import BLOCK_CATEGORIES, BLOCK_THRESHOLD
except ImportError:
    from component1 import validate_payload
    from config import BLOCK_CATEGORIES, BLOCK_THRESHOLD

log = logging.getLogger(__name__)

ACTIONS = ("blur", "block")
OVERRIDES = ("auto",) + ACTIONS

#: Severity order — an incident's action can escalate blur → block, never back.
SEVERITY = {"blur": 1, "block": 2}


def is_valid(payload: dict) -> bool:
    """True if *payload* matches the schema; logs the reason otherwise."""
    from jsonschema import ValidationError

    try:
        validate_payload(payload)
        return True
    except ValidationError as e:
        log.warning("ignoring invalid payload: %s", e.message)
        return False
    except TypeError as e:                       # not even a dict
        log.warning("ignoring invalid payload: %s", e)
        return False


def decide(payload: dict, override: str = "auto",
           block_threshold: float = BLOCK_THRESHOLD) -> str | None:
    """
    Action for one TriggerPayload: "blur", "block", or ``None`` if the payload
    is invalid (it is then ignored).
    """
    if override not in OVERRIDES:
        raise ValueError(f"override must be one of {OVERRIDES}, got {override!r}")
    if not isinstance(payload, dict) or not is_valid(payload):
        return None
    if override != "auto":
        return override

    category = payload["risk_category"]
    if category in BLOCK_CATEGORIES:
        return "block"
    if category == "violence":
        return "block" if payload["confidence_score"] >= block_threshold else "blur"
    return "blur"                                # unknown_flagged (and any future category)


def stronger(a: str | None, b: str | None) -> str | None:
    """The more severe of two actions (None = nothing shown)."""
    return max((x for x in (a, b) if x), key=SEVERITY.__getitem__, default=None)

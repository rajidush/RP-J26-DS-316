"""
Re-validating recorded outputs against Component 3's contracts (plan, Thu 15 Oct).

One check per boundary; evidence/validate_schemas.py walks the evidence files
with them. Each check returns None when the item passes, or a short reason.

    in    trigger_error            comp1_to_comp3 / comp2_to_comp3, by source_component
    model intercept_outcome        what the recorded fallback reason says about the raw
                                   model reply: "valid" JSON in the schema's shape,
                                   "parse_failure", or "no_reply"
          delivered_intercept_error / delivered_inquire_error
                                   the controller grammar (and INTERCEPT's post-check)
                                   on the line the child actually saw
    out   record_error             comp3_to_comp4
"""
from __future__ import annotations

from typing import Optional

import jsonschema

from .adversarial import POST_CHECK_REASONS, SCHEMA_REASONS
from .grammar_decoder import _GRAMMAR, delivery_violation
from .schemas import _validate_against_schema

_INBOUND = {"component1_screen_monitoring": "comp1_to_comp3.schema.json",
            "component2_hate_speech_detection": "comp2_to_comp3.schema.json"}


def _schema_error(payload: dict, schema_file: str) -> Optional[str]:
    try:
        _validate_against_schema(payload, schema_file)
        return None
    except jsonschema.ValidationError as e:
        where = "/".join(str(p) for p in e.absolute_path) or "(record)"
        return f"{where}: {e.message}"


def trigger_error(trigger: dict) -> Optional[str]:
    schema_file = _INBOUND.get(trigger.get("source_component"))
    if schema_file is None:
        return f"source_component: unknown {trigger.get('source_component')!r}"
    return _schema_error(trigger, schema_file)


def record_error(record: dict) -> Optional[str]:
    return _schema_error(record, "comp3_to_comp4.schema.json")


def intercept_outcome(source: str, reason: Optional[str]) -> str:
    """SCHEMA_REASONS mean the reply did not satisfy INTERCEPT_SCHEMA: not JSON, or
    JSON breaking its shape, opener enum or question pattern (the grammar is built
    from the same pieces, so a joined line failing it broke the schema too). A
    post-check reason means valid JSON whose words were blocked. Any other reason
    is an error, never counted as valid."""
    if reason == "model_unreachable":
        return "no_reply"
    if reason in SCHEMA_REASONS:
        return "parse_failure"
    if reason is None or reason in POST_CHECK_REASONS:
        return "valid"
    raise ValueError(f"unknown INTERCEPT fallback reason {reason!r} (source {source!r})")


def delivered_intercept_error(line: str) -> Optional[str]:
    return delivery_violation(line)


def delivered_inquire_error(line: str) -> Optional[str]:
    return None if _GRAMMAR["INQUIRE"].match(line) else "grammar"


def failed_only_by(result: Optional[str], rule: Optional[str]) -> bool:
    """True if a check failed, and only by `rule`: a rule added after the line was
    recorded. A passing check (None) is never 'resolved'."""
    return result is not None and result == rule

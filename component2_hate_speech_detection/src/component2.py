"""
Component 2 (Hate-Speech Detection) -- owned by IT23209152.

Step 1: validate every outbound payload against the shared JSON schema.
Step 2: auditable keyword layer can classify live text into a schema trigger.

Mock path (no text) keeps the end-to-end demo working. Live path returns
None when the keyword screen finds nothing — do not wake C3 on clean chat.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import jsonschema

from component2_hate_speech_detection.src.keywords import screen_keywords

MOCK_INPUTS = Path(__file__).resolve().parents[1] / "mock_inputs"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "interface-contracts"
    / "comp2_to_comp3.schema.json"
)


def _load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def _validate(payload: dict) -> dict:
    jsonschema.validate(instance=payload, schema=_load_schema())
    return payload


def _new_session_fields() -> dict:
    return {
        "session_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _payload_from_mock(mock_file: str) -> dict:
    with open(MOCK_INPUTS / mock_file) as f:
        payload = json.load(f)
    payload.update(_new_session_fields())
    return _validate(payload)


def _payload_from_keywords(
    text: str,
    *,
    content_type: str,
    platform: str,
) -> Optional[dict]:
    result = screen_keywords(text)
    if not result.matched:
        return None

    payload = {
        **_new_session_fields(),
        "source_component": "component2_hate_speech_detection",
        "content_type": content_type,
        "risk_category": result.risk_category,
        "confidence_score": result.confidence_score,
        "context_metadata": {
            "platform": platform,
            "language_detected": "en",
            # Family IDs only — never the raw flagged text (privacy + contract).
            "matched_families": list(result.matched_families),
            "screen_stage": "keyword_layer",
        },
    }
    return _validate(payload)


def analyze_text(
    text: str | None = None,
    mock_file: str = "sample_analysis.json",
    *,
    content_type: str = "text_message",
    platform: str = "unknown",
) -> Optional[dict]:
    """
    Produce a schema-valid TriggerPayload, or None if live text is clean.

    - text is None  -> mock fixture (integration / demo)
    - text provided -> keyword screen; None means no C2 trigger
    """
    if text is None:
        return _payload_from_mock(mock_file)
    return _payload_from_keywords(text, content_type=content_type, platform=platform)

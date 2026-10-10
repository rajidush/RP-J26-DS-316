"""
Component 2 (Hate-Speech Detection) -- owned by IT23209152.

Public entry points. The engine behind them is `src/engine/` (lexicon, two
corroborating heads, framing guard, fusion, age-aware decision), shaped as the
Guardian C2 reference engine.

    analyze_text()   -> team-schema trigger dict for C3, or None (stable API)
    analyze()        -> full Verdict (rung, evidence, explanation) for the app and C4

Mock path (no text) keeps the end-to-end demo working. The live path returns
None when nothing reaches an alerting rung, so clean chat never wakes C3.
Every outbound payload is validated against the shared JSON schema.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import jsonschema

from component2_hate_speech_detection.src.engine.analyzer import Analyzer
from component2_hate_speech_detection.src.engine.text_scorer import TextScorer
from component2_hate_speech_detection.src.engine.verdict import Verdict

MOCK_INPUTS = Path(__file__).resolve().parents[1] / "mock_inputs"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "interface-contracts"
    / "comp2_to_comp3.schema.json"
)

_analyzer: Optional[Analyzer] = None
_analyzer_lock = threading.Lock()


def get_analyzer() -> Analyzer:
    """Shared engine. Heads load on first use; C2_USE_HEADS=0 runs the lexicon only."""
    global _analyzer
    with _analyzer_lock:
        if _analyzer is None:
            use_heads = os.environ.get("C2_USE_HEADS", "1") != "0"
            _analyzer = Analyzer(scorer=TextScorer(use_heads=use_heads))
        return _analyzer


def set_analyzer(analyzer: Optional[Analyzer]) -> None:
    """Swap the shared engine (tests, ablations, a pre-warmed app instance)."""
    global _analyzer
    with _analyzer_lock:
        _analyzer = analyzer


def _load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def _validate(payload: dict) -> dict:
    jsonschema.validate(instance=payload, schema=_load_schema())
    return payload


def _payload_from_mock(mock_file: str) -> dict:
    with open(MOCK_INPUTS / mock_file) as f:
        payload = json.load(f)
    payload["session_id"] = str(uuid.uuid4())
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    return _validate(payload)


def analyze(text: str, *, age: int = 10, source: str = "typed") -> Verdict:
    """Full decision record: rung, urgency, evidence, explanation, latency."""
    return get_analyzer().analyze_text(text, age, source=source)


def analyze_text(
    text: str | None = None,
    mock_file: str = "sample_analysis.json",
    *,
    age: int = 10,
    content_type: str = "text_message",
    platform: str = "unknown",
) -> Optional[dict]:
    """
    Produce a schema-valid TriggerPayload for C3, or None if C3 should not wake.

    - text is None  -> mock fixture (integration / demo)
    - text provided -> full C2 engine for a child of `age`
    """
    if text is None:
        return _payload_from_mock(mock_file)
    source = "audio" if content_type == "voice_transcript" else "typed"
    payload = analyze(text, age=age, source=source).to_trigger_payload(platform=platform)
    return _validate(payload) if payload is not None else None

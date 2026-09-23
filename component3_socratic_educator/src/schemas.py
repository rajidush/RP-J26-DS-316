"""
Typed data models for Component 3 (Socratic Educator).

These mirror docs/interface-contracts/*.schema.json exactly. Keeping the
Python dataclasses and the JSON Schemas in sync is a manual step for now --
log every change in docs/interface-contracts/CHANGELOG.md.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import jsonschema

_CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "docs" / "interface-contracts"


class RiskCategory(str, Enum):
    EXPLICIT_VISUAL = "explicit_visual"
    VIOLENCE = "violence"
    SELF_HARM_IMAGERY = "self_harm_imagery"
    HATE_SPEECH = "hate_speech"
    CYBERBULLYING = "cyberbullying"
    GROOMING_LANGUAGE = "grooming_language"
    SELF_HARM_LANGUAGE = "self_harm_language"
    UNKNOWN_FLAGGED = "unknown_flagged"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class EmotionalState(str, Enum):
    CALM = "calm"
    DEFENSIVE = "defensive"
    DISTRESSED = "distressed"
    UNCLEAR = "unclear"


@dataclass
class TriggerPayload:
    """Inbound trigger from Component 1 or Component 2."""
    session_id: str
    source_component: str
    content_type: str
    risk_category: RiskCategory
    confidence_score: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    context_metadata: dict = field(default_factory=dict)

    @staticmethod
    def new(source_component: str, content_type: str, risk_category: RiskCategory,
            confidence_score: float, **context_metadata) -> "TriggerPayload":
        return TriggerPayload(
            session_id=str(uuid.uuid4()),
            source_component=source_component,
            content_type=content_type,
            risk_category=risk_category,
            confidence_score=confidence_score,
            context_metadata=context_metadata,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["risk_category"] = self.risk_category.value
        return d

    def validate(self) -> None:
        schema_file = "comp1_to_comp3.schema.json" if "component1" in self.source_component \
            else "comp2_to_comp3.schema.json"
        _validate_against_schema(self.to_dict(), schema_file)


@dataclass
class EvaluateOutput:
    """Outbound structured record for Component 4. This is Function 2's
    typed-output guarantee: enumerated/boolean fields only, no free text
    about the raw conversation content."""
    session_id: str
    risk_level: RiskLevel
    emotional_state: EmotionalState
    self_regulation_shown: bool
    escalation_flag: bool
    dialogue_turns: int
    dialogue_summary: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        d["emotional_state"] = self.emotional_state.value
        return d

    def validate(self) -> None:
        _validate_against_schema(self.to_dict(), "comp3_to_comp4.schema.json")


def _validate_against_schema(payload: dict, schema_filename: str) -> None:
    schema_path = _CONTRACTS_DIR / schema_filename
    with open(schema_path) as f:
        schema = json.load(f)
    jsonschema.validate(instance=payload, schema=schema)


def load_mock_trigger(path: str | Path) -> TriggerPayload:
    """Load a mock_inputs/*.json file (shaped like Component 1/2's real
    output) and turn it into a TriggerPayload for local testing."""
    with open(path) as f:
        raw = json.load(f)
    raw["risk_category"] = RiskCategory(raw["risk_category"])
    return TriggerPayload(**raw)

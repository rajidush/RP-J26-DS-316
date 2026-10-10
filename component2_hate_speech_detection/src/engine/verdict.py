"""The C2 output record (Guardian PROJECT_PLAN §6.6 `Verdict`; proposal FR7-FR9).

Field names match the Guardian protobuf exactly, so moving to the generated
`guardian.v1.Verdict` later is a mechanical swap. Until then, two adapters
deliver it to the current team integration:

    to_trigger_payload()  -> docs/interface-contracts/comp2_to_comp3.schema.json (C3)
    to_record()           -> a storable row for C4, with no raw content

Privacy (proposal FR8, NFR7): the C3 payload never carries flagged text; it
carries a category and a child-safe summary. The only text kept anywhere is
`Evidence.redacted_snippet`: at most 120 characters, PII-masked, and empty for
the oldest persona.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

ENGINE_VERSION = "1.0.0"

CHILD_SAFE_SUMMARIES: Dict[str, str] = {
    "threat": "Someone is using words that sound like a threat to hurt someone.",
    "hate_identity": "Someone is using hurtful words about a group of people.",
    "bullying": "Someone is using words that can bully or put a person down.",
    "sexual_harassment": "This has messages that are not okay for you to get.",
    "self_harm": "Something here talks about getting hurt. You are not alone, and it is okay to talk to someone.",
    "grooming": "Someone may be asking you to keep secrets. Safe adults don't ask kids to keep secrets.",
    "profanity": "This has strong language.",
    "none": "We checked this and it looks okay.",
}

# Guardian category -> team schema `risk_category` enum (comp2_to_comp3.schema.json).
# threat and sexual_harassment have no enum yet: proposed in the interface CHANGELOG.
TEAM_RISK_CATEGORY = {
    "hate_identity": "hate_speech",
    "bullying": "cyberbullying",
    "threat": "cyberbullying",
    "sexual_harassment": "cyberbullying",
    "grooming": "grooming_language",
    "self_harm": "self_harm_language",
}

_PII = (
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[email]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[card]"),
    (re.compile(r"\+?\d[\d\s().-]{7,}\d"), "[phone]"),
    (re.compile(r"(?i)\b(password|passwd|pwd|token|api[_-]?key|secret)\s*[:=]\s*\S+"), r"\1=[hidden]"),
    (re.compile(r"https?://\S+"), "[link]"),
)


def redact(text: str, limit: int = 120) -> str:
    out = " ".join((text or "").split())
    for pattern, replacement in _PII:
        out = pattern.sub(replacement, out)
    return out[:limit]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class CategoryScore:
    category: str
    p: float
    calibrated: bool
    producer: str


@dataclass
class Evidence:
    producer: str
    score: float
    rule_id: str = ""
    box: Optional[List[float]] = None   # normalised x0, y0, x1, y1 of the screen
    redacted_snippet: str = ""
    source: str = ""                    # typed | ocr | asr


@dataclass
class Verdict:
    persona: str
    top_category: str
    p: float
    rung: str
    urgent: bool
    child_safe_summary: str
    scores: List[CategoryScore] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    regions: List[List[float]] = field(default_factory=list)
    framing_reason: str = ""
    abstained: bool = False
    app: Dict[str, str] = field(default_factory=dict)
    observation_id: str = ""
    latency_ms: Dict[str, float] = field(default_factory=dict)
    engine_version: str = ENGINE_VERSION
    model_pack_version: str = ""
    policy_version: str = ""
    # Not in the Guardian contract: the decision trail shown in the app and to C4.
    explanation: str = ""
    theta: float = 0.0
    source: str = ""                     # typed | screen | audio
    # Hash of the flagged text and category, never the text: lets the app alert
    # once for a harmful message that stays on screen instead of every check.
    fingerprint: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: str = field(default_factory=_now)

    @property
    def recommended_action(self) -> str:
        from .decide import RUNG_TO_ACTION

        return RUNG_TO_ACTION[self.rung]

    @property
    def alerts(self) -> bool:
        return self.rung != "L0"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["recommended_action"] = self.recommended_action
        return d

    def to_trigger_payload(self, *, platform: str = "unknown") -> Optional[dict]:
        """Team-schema trigger for C3, or None when C3 should not be woken.

        C3 opens a dialogue for rungs that blur or block (L2+). Fields the team
        schema does not have yet travel in `context_metadata`, which the schema
        leaves open: the fine-grained category, rung, urgency and age band.
        """
        if self.rung in ("L0", "L1") or self.top_category not in TEAM_RISK_CATEGORY:
            return None
        return {
            "session_id": str(uuid.uuid4()),
            "timestamp": self.ts,
            "source_component": "component2_hate_speech_detection",
            "content_type": "voice_transcript" if self.source == "audio" else "text_message",
            "risk_category": TEAM_RISK_CATEGORY[self.top_category],
            "confidence_score": round(float(self.p), 4),
            "context_metadata": {
                "platform": platform,
                "language_detected": "en",
                "c2_category": self.top_category,
                "rung": self.rung,
                "recommended_action": self.recommended_action,
                "urgent": self.urgent,
                "persona": self.persona,
                "child_safe_summary": self.child_safe_summary,
                # Rule IDs only, never the flagged text (privacy + contract).
                "matched_families": sorted({e.rule_id for e in self.evidence if e.rule_id}),
                "framing_reason": self.framing_reason,
                "verdict_id": self.id,
                "engine_version": self.engine_version,
                "policy_version": self.policy_version,
            },
        }

    def to_record(self) -> dict:
        """C4 row: decision metadata plus redacted evidence; snippets empty for P3."""
        record = self.to_dict()
        if self.persona == "P3_RESPECT":
            for evidence in record["evidence"]:
                evidence["redacted_snippet"] = ""
        return record

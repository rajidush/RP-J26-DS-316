"""
Auditable keyword / pattern layer for Component 2 (Step 2).

This is the cheap first screen in the cascade (proposal SO1):
- patterns are grouped into named families (auditable)
- matches map onto the shared schema risk_category enums
- evidence is family IDs only — never the raw flagged text

Later steps add pretrained models + corroboration on top of this layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Each family is a small, reviewable rule set. Prefer phrase patterns over
# single ambiguous words so ordinary game chat is less likely to trip.
KEYWORD_FAMILIES: tuple[dict, ...] = (
    {
        "family_id": "cyberbullying_insults",
        "risk_category": "cyberbullying",
        "base_score": 0.62,
        "patterns": (
            r"\byou('re| are)\s+(so\s+)?(stupid|dumb|pathetic|worthless)\b",
            r"\b(nobody|no one)\s+likes\s+you\b",
            r"\bkill\s+yourself\b",
            r"\bkys\b",
        ),
    },
    {
        "family_id": "hate_speech_identity_attack",
        "risk_category": "hate_speech",
        "base_score": 0.72,
        "patterns": (
            r"\bhate\s+all\s+\w+\b",
            r"\bgo\s+back\s+to\s+your\s+country\b",
            r"\b(people|kids)\s+like\s+you\s+shouldn'?t\s+exist\b",
        ),
    },
    {
        "family_id": "grooming_secrecy",
        "risk_category": "grooming_language",
        "base_score": 0.78,
        "patterns": (
            r"\bdon'?t\s+tell\s+(your\s+)?(mom|dad|parents|anyone)\b",
            r"\bour\s+little\s+secret\b",
            r"\bkeep\s+this\s+between\s+us\b",
        ),
    },
    {
        "family_id": "self_harm_language",
        "risk_category": "self_harm_language",
        "base_score": 0.8,
        "patterns": (
            r"\bi\s+want\s+to\s+die\b",
            r"\bi\s+('?m|am)\s+going\s+to\s+hurt\s+myself\b",
            r"\bend\s+it\s+all\b",
        ),
    },
)

_COMPILED: list[tuple[str, str, float, re.Pattern[str]]] = [
    (
        family["family_id"],
        family["risk_category"],
        float(family["base_score"]),
        re.compile(pattern, re.IGNORECASE),
    )
    for family in KEYWORD_FAMILIES
    for pattern in family["patterns"]
]


@dataclass(frozen=True)
class KeywordScreenResult:
    """Result of the keyword screen. Safe to log: no raw matched spans."""

    matched: bool
    risk_category: str | None
    confidence_score: float
    matched_families: tuple[str, ...]


def screen_keywords(text: str) -> KeywordScreenResult:
    """
    Score text with the auditable keyword layer.

    Returns matched=False when nothing hits (caller should not raise a C3 trigger).
    When several families hit, the highest base_score wins the category.
    """
    if not text or not text.strip():
        return KeywordScreenResult(
            matched=False,
            risk_category=None,
            confidence_score=0.0,
            matched_families=(),
        )

    hits: dict[str, tuple[str, float]] = {}
    for family_id, risk_category, base_score, pattern in _COMPILED:
        if pattern.search(text):
            previous = hits.get(family_id)
            if previous is None or base_score > previous[1]:
                hits[family_id] = (risk_category, base_score)

    if not hits:
        return KeywordScreenResult(
            matched=False,
            risk_category=None,
            confidence_score=0.0,
            matched_families=(),
        )

    # Highest-scoring family decides the outbound risk_category.
    ranked = sorted(hits.items(), key=lambda item: item[1][1], reverse=True)
    best_family_id, (best_category, best_score) = ranked[0]
    # Mild boost when multiple independent families fire (still capped at 1.0).
    multi_boost = min(0.1 * (len(hits) - 1), 0.15)
    confidence = min(best_score + multi_boost, 1.0)
    families = tuple(sorted(hits.keys()))

    return KeywordScreenResult(
        matched=True,
        risk_category=best_category,
        confidence_score=round(confidence, 3),
        matched_families=families,
    )


def list_family_ids() -> tuple[str, ...]:
    """Expose family IDs for tests and documentation."""
    return tuple(family["family_id"] for family in KEYWORD_FAMILIES)

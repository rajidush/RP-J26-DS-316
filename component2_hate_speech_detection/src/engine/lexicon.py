"""Stage-1 lexicon: explainable, instant, no weights (Guardian M-LEX).

Ported from the J26-DS-316 MVP (`analyst/stage1/lexicon.py`); the rules now
live in `packs/policy/lexicon.json`. Score bands are chosen so the persona
thresholds actually discriminate: the MID band straddles 0.55 / 0.65 / 0.75,
so the same phrase can escalate for an 8-year-old and clear for a 15-year-old,
while the HIGH band (slurs, self-harm directives) sits above every threshold.

A CUE band (0.45) sits above the Stage-1 gate but below every alert
threshold: the hit is recorded as evidence and the text is re-read in Stage
2, but a cue never alerts on its own. Grooming secrecy uses it, because
grooming is a conversation-level pattern rather than one message.

Every hit names the phrase or family that fired, so a parent or reviewer can
read the exact rule behind a detection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .policy import Policy, default_policy

# "kiiill" -> "kiill" (keep doubles: "kill", "gg"); 3+ runs are obfuscation.
_RUNS = re.compile(r"(.)\1{2,}")
# "k y s" -> "kys": 3+ single letters separated by whitespace.
_SPACED = re.compile(r"\b(?:[a-z]\s+){2,}[a-z]\b")


def _phrase_pattern(needle: str) -> re.Pattern:
    r"""Word-boundary anchored, tolerant of missing or repeated whitespace.

    `\s*` rather than `\s+`, because OCR swallows spaces ("go back toyour
    country") and despace() joins letter-spaced text ("killyourself"). The
    `\b` anchors stop a phrase firing inside a longer word.
    """
    parts = [re.escape(part) for part in needle.split()]
    return re.compile(rf"\b{r'\s*'.join(parts)}\b")


@dataclass(frozen=True)
class Family:
    id: str
    category: str
    band: str
    suppressed_in: Tuple[str, ...]
    patterns: Tuple[re.Pattern, ...]

    def matches(self, views: Tuple[str, ...]) -> bool:
        return any(p.search(v) for p in self.patterns for v in views)


@dataclass(frozen=True)
class LexiconHit:
    score: float
    category: str
    hits: Tuple[str, ...]
    gaming: bool


class Lexicon:
    def __init__(self, policy: Optional[Policy] = None) -> None:
        pack = (policy or default_policy()).lexicon
        self.version = pack["version"]
        self._leet = str.maketrans(pack["leet"])
        self._high = tuple(pack["phrases"]["high"])
        self._mid = tuple(pack["phrases"]["mid"])
        self._gaming_benign = tuple(pack["phrases"]["gaming_benign"])
        self._literal_categories = {k: tuple(v) for k, v in pack["literal_categories"].items()}
        self._phrase_re = {p: _phrase_pattern(p) for p in self._high + self._mid + self._gaming_benign}
        self._contexts = {name: re.compile(rx) for name, rx in pack["contexts"].items()}
        self._families = tuple(
            Family(
                id=f["id"],
                category=f["category"],
                band=f["band"],
                suppressed_in=tuple(f.get("suppressed_in", ())),
                patterns=tuple(re.compile(p) for p in f["patterns"]),
            )
            for f in pack["families"]
        )
        bands = pack["bands"]
        self._high_band = bands["high"]
        self._mid_band = bands["mid"]
        self._cue_score = float(bands["cue"])
        self._gaming_score = float(bands["gaming"])
        self._neutral_score = float(bands["neutral"])
        self._empty_score = float(bands["empty"])

    # -- normalisation ------------------------------------------------------

    def normalize(self, text: str) -> str:
        lowered = (text or "").lower().translate(self._leet)
        collapsed = re.sub(r"[^a-z0-9\s']+", " ", lowered)
        return re.sub(r"\s+", " ", collapsed).strip()

    @staticmethod
    def _views(blob: str) -> Tuple[str, ...]:
        doubled = _RUNS.sub(lambda m: m.group(1) * 2, blob)
        singled = _RUNS.sub(lambda m: m.group(1), blob)
        despace = lambda s: _SPACED.sub(lambda m: re.sub(r"\s+", "", m.group(0)), s)  # noqa: E731
        return (blob, doubled, singled, despace(blob), despace(singled))

    def in_context(self, name: str, text: str) -> bool:
        pattern = self._contexts.get(name)
        return bool(pattern and pattern.search(self.normalize(text)))

    def in_gaming_context(self, text: str) -> bool:
        """Exported because the model heads need it too (gaming cap, invariant I-6)."""
        return self.in_context("gaming", text)

    # -- scoring --------------------------------------------------------------

    def _phrase_hits(self, views: Tuple[str, ...], phrases: Tuple[str, ...]) -> List[str]:
        return [p for p in phrases if any(self._phrase_re[p].search(v) for v in views)]

    def _category_for_literals(self, high: List[str]) -> str:
        for category in ("threat", "hate_identity", "sexual_harassment"):
            if any(marker in high for marker in self._literal_categories.get(category, ())):
                return category
        return "bullying"

    def score(self, text: str) -> LexiconHit:
        blob = self.normalize(text)
        if not blob:
            return LexiconHit(self._empty_score, "none", (), False)

        views = self._views(blob)
        high = self._phrase_hits(views, self._high)
        mid = self._phrase_hits(views, self._mid)
        active = {name for name, rx in self._contexts.items() if rx.search(blob)}
        gaming = "gaming" in active
        families = [
            f for f in self._families
            if not (active & set(f.suppressed_in)) and f.matches(views)
        ]
        by_band = {band: [f for f in families if f.band == band] for band in ("high", "mid", "cue")}

        # Literal slurs outrank patterns for category: they are the most
        # specific evidence; patterns fill the categories literals cannot see.
        if high or by_band["high"]:
            hits = high + [f.id for f in by_band["high"]]
            band = self._high_band
            score = round(min(band["cap"], band["base"] + band["step"] * (len(hits) - 1)), 4)
            category = self._category_for_literals(high) if high else by_band["high"][0].category
            return LexiconHit(score, category, tuple(hits), gaming)

        if mid or by_band["mid"]:
            hits = mid + [f.id for f in by_band["mid"]]
            band = self._mid_band
            score = round(min(band["cap"], band["base"] + band["step"] * len(hits)), 4)
            category = by_band["mid"][0].category if by_band["mid"] and not mid else "bullying"
            return LexiconHit(score, category, tuple(hits), gaming)

        if by_band["cue"]:
            return LexiconHit(self._cue_score, by_band["cue"][0].category,
                              tuple(f.id for f in by_band["cue"]), gaming)

        if self._phrase_hits(views, self._gaming_benign):
            return LexiconHit(self._gaming_score, "none", (), gaming)

        return LexiconHit(self._neutral_score, "none", (), gaming)

"""C2 Analyst: one Observation in, one Verdict out (proposal Table 4.1, stages 3-6).

    spans --score each (cached)--> top span --gate (theta1)--> stage-2 re-read
          --> fusion (I-1, I-2) --> decide(persona) --> Verdict with evidence

An Observation is what a reader produced from one moment of input: typed
chat, the text blocks OCR found on screen (with their positions), or one
transcribed utterance. Every span is scored by the same text scorer (FR1).

Resource behaviour (proposal NFR1-3):
- Scores are cached by text. Screen text is mostly unchanged between checks,
  so a chat window costs model time only for its *new* lines.
- Scoring is serialised behind one lock, so the screen and audio readers never
  run the heads at the same time and CPU use stays bounded.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .decide import decide, persona_for_age
from .fusion import FullTextReread, Signal, fuse
from .policy import Policy, default_policy
from .text_scorer import ScoreDetail, TextScorer
from .verdict import CHILD_SAFE_SUMMARIES, CategoryScore, Evidence, Verdict, redact


@dataclass
class TextSpan:
    text: str
    source: str = "typed"                 # typed | ocr | asr
    box: Optional[List[float]] = None     # normalised x0, y0, x1, y1 (screen spans)
    conversation_id: str = ""


@dataclass
class Observation:
    spans: List[TextSpan]
    source: str = "typed"                 # typed | screen | audio
    app: Dict[str, str] = field(default_factory=dict)
    vision: Optional[Signal] = None       # evidence only until calibrated (I-1)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    read_ms: Dict[str, float] = field(default_factory=dict)  # reader timings to carry through


@dataclass
class _Scored:
    span: TextSpan
    detail: ScoreDetail


class ScoreCache:
    """LRU of text -> ScoreDetail. Scoring is deterministic for a fixed policy."""

    def __init__(self, size: int = 4096) -> None:
        self._size = size
        self._data: "OrderedDict[str, ScoreDetail]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(text: str) -> str:
        return " ".join((text or "").split())

    def get(self, text: str) -> Optional[ScoreDetail]:
        key = self.key(text)
        detail = self._data.get(key)
        if detail is None:
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return detail

    def put(self, text: str, detail: ScoreDetail) -> None:
        self._data[self.key(text)] = detail
        self._data.move_to_end(self.key(text))
        while len(self._data) > self._size:
            self._data.popitem(last=False)


class Analyzer:
    def __init__(self, policy: Optional[Policy] = None, scorer: Optional[TextScorer] = None) -> None:
        self.policy = policy or default_policy()
        self.scorer = scorer or TextScorer(self.policy)
        self.cache = ScoreCache()
        self.reread = FullTextReread(lambda chunk: self._score(chunk).score)
        self.stage1_theta = self.policy.decision["stage1_theta"]
        self._lock = threading.Lock()
        self.model_calls = 0

    @property
    def model_pack_version(self) -> str:
        return self.scorer.name

    def _score(self, text: str) -> ScoreDetail:
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        detail = self.scorer.score(text)
        if detail.model_score is not None:
            self.model_calls += 1
        self.cache.put(text, detail)
        return detail

    def analyze_text(self, text: str, age: int, *, source: str = "typed", app: Optional[dict] = None) -> Verdict:
        span_source = {"typed": "typed", "screen": "ocr", "audio": "asr"}.get(source, source)
        return self.analyze(Observation([TextSpan(text, span_source)], source=source, app=app or {}), age)

    def analyze(self, obs: Observation, age: int) -> Verdict:
        t0 = time.perf_counter()
        persona = persona_for_age(age)
        latency = dict(obs.read_ms)
        spans = [s for s in obs.spans if (s.text or "").strip()]

        with self._lock:
            t_score = time.perf_counter()
            scored = [_Scored(s, self._score(s.text)) for s in spans]
            latency["score_ms"] = round((time.perf_counter() - t_score) * 1000, 1)

            top = max(scored, key=lambda s: s.detail.score, default=None)
            text_score = top.detail.score if top else 0.0
            category = top.detail.category if top else "none"
            escalated = text_score >= self.stage1_theta

            full_score = text_score
            if escalated and len(spans) > 1:
                t_s2 = time.perf_counter()
                joined = " ".join(s.text for s in spans)
                full_score = self.reread.score(joined, text_score)
                latency["stage2_ms"] = round((time.perf_counter() - t_s2) * 1000, 1)

        signals = [Signal("text", full_score, True, self.scorer.name)]
        if obs.vision is not None:
            signals.append(obs.vision)
        fusion = fuse(signals)
        p = fusion.fused if escalated else round(text_score, 4)
        decision = decide(category if escalated else "none", p, persona, self.policy)

        flagged = [s for s in scored if s.detail.score >= self.stage1_theta]
        evidence = self._evidence(flagged, persona)
        if obs.vision is not None and obs.vision.score > 0:
            evidence.append(Evidence(producer=obs.vision.producer or "vision", score=round(obs.vision.score, 4),
                                     rule_id="" if obs.vision.calibrated else "uncalibrated:evidence_only"))
        regions = [s.span.box for s in flagged if s.span.box and s.detail.score >= decision.theta > 0]
        framing_reason = top.detail.framing_reason if top else ""

        latency["total_ms"] = round((time.perf_counter() - t0) * 1000 + sum(obs.read_ms.values()), 1)
        verdict = Verdict(
            persona=persona,
            top_category=decision.category,
            p=round(p, 4),
            rung=decision.rung,
            urgent=decision.urgent,
            child_safe_summary=CHILD_SAFE_SUMMARIES.get(decision.category, CHILD_SAFE_SUMMARIES["none"]),
            scores=[CategoryScore(category, round(full_score, 4), True, self.scorer.name)] if escalated else [],
            evidence=evidence,
            regions=regions,
            framing_reason=framing_reason,
            app=dict(obs.app),
            observation_id=obs.id,
            latency_ms=latency,
            model_pack_version=self.model_pack_version,
            policy_version=self.policy.version,
            theta=decision.theta,
            source=obs.source,
        )
        verdict.explanation = explain(verdict, top.detail if top else None, escalated, fusion.ignored)
        return verdict

    @staticmethod
    def _evidence(flagged: List[_Scored], persona: str) -> List[Evidence]:
        out: List[Evidence] = []
        for s in sorted(flagged, key=lambda s: s.detail.score, reverse=True)[:8]:
            snippet = "" if persona == "P3_RESPECT" else redact(s.span.text)
            for rule in s.detail.hits or [""]:
                out.append(Evidence(producer="lexicon" if rule else "heads", score=s.detail.lexicon_score if rule else (s.detail.model_score or 0.0),
                                    rule_id=rule, box=s.span.box, redacted_snippet=snippet, source=s.span.source))
            for head, score in s.detail.head_scores.items():
                out.append(Evidence(producer=head, score=score, box=s.span.box, source=s.span.source))
        return out


def explain(v: Verdict, detail: Optional[ScoreDetail], escalated: bool, ignored: List[str]) -> str:
    """One plain sentence per fact that moved the decision (proposal FR7)."""
    if detail is None:
        return "Nothing to check."
    parts: List[str] = []
    if v.rung != "L0":
        parts.append(f"Flagged as {v.top_category.replace('_', ' ')} (rung {v.rung}).")
    elif escalated:
        parts.append(f"Looked risky, but stayed below this age group's threshold ({v.theta:.2f}).")
    else:
        parts.append("Nothing risky found.")
    if detail.hits:
        parts.append("Matched rule(s): " + ", ".join(detail.hits[:4]) + ".")
    if detail.head_scores:
        heads = ", ".join(f"{k.split(':')[-1]} {s:.2f}" for k, s in detail.head_scores.items())
        parts.append(f"Language models: {heads} ({detail.corroboration.replace('_', ' ')}).")
    if detail.gaming_capped:
        parts.append("Game context: model score capped so trash talk doesn't trigger.")
    if detail.framing_reason:
        parts.append("Score held down: the text reads like reporting or quoting harm, not doing it.")
    if ignored:
        parts.append(f"{', '.join(ignored)} read but not calibrated, so it did not affect the score.")
    if v.urgent:
        parts.append("Urgent: a parent should be told now.")
    parts.append(f"Final risk {v.p:.2f}.")
    return " ".join(parts)

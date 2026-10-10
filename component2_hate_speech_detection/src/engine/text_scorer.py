"""Stage-1 text screen: lexicon + two corroborating heads + framing guard (SO1, SO2).

Ported from the J26-DS-316 MVP (`analyst/stage1/text_fast.py`). Every text
source (typed chat, OCR'd screen text, speech transcript) goes through this
one scorer (FR1), so how the words arrived never changes how they are judged.

Layering, and why each layer is here:

    lexicon      owns explicit abuse, threats, bullying and obfuscation. ~1 ms,
                 and every hit names the rule that fired.
    two heads    read *implicit* harm the lexicon scores at zero. Combined by
                 corroboration (below), not max().
    gaming cap   competitive trash talk is the heads' worst blind spot
                 ("you're trash at this game" scored 0.93 on toxic-bert).
    framing      applied last, to the combined score, so no layer routes around it.

Corroboration (proposal design rule 1, FR3). Combining heads with max() lets
either head's false positive through: 156 of 165 sampled Berkeley false
positives came from a model. Measured on train splits (recall / FP rate):

                     berkeley       davidson
    max()            94% / 43%      96% / 18%
    corroboration    91% / 36%      95% /  9%

So: one head is trusted alone at >= 0.90; otherwise the other must be >= 0.50;
otherwise the reading is damped to 70%. The screen is re-checked every few
seconds, so a miss gets another chance while a false alert interrupts a child
immediately. Precision has no second chance; recall does.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .framing import FramingGuard
from .heads import DEFAULT_HEADS, TextHead
from .lexicon import Lexicon
from .policy import Policy, default_policy

Reading = Tuple[float, Optional[str], Dict[str, float]]


@dataclass
class ScoreDetail:
    """Everything needed to explain one text score."""

    score: float
    category: str
    hits: List[str] = field(default_factory=list)
    lexicon_score: float = 0.0
    model_score: Optional[float] = None
    head_scores: Dict[str, float] = field(default_factory=dict)
    model_category: Optional[str] = None
    model_labels: Dict[str, float] = field(default_factory=dict)
    corroboration: str = ""  # solo_trusted | corroborated | damped | single_head | ""
    gaming_capped: bool = False
    framing_reason: str = ""
    discounted_from: Optional[float] = None

    @property
    def discounted(self) -> bool:
        return self.discounted_from is not None


class TextScorer:
    def __init__(
        self,
        policy: Optional[Policy] = None,
        heads: Optional[Sequence[TextHead]] = None,
        model_fn: Optional[Callable[[str], float]] = None,
        use_heads: bool = True,
        use_corroboration: bool = True,
        use_framing: bool = True,
        use_gaming_cap: bool = True,
    ) -> None:
        """`model_fn` injects a fake head for tests; `use_*` flags exist for ablation (proposal §3.5)."""
        policy = policy or default_policy()
        self.lexicon = Lexicon(policy)
        self.framing = FramingGuard(policy)
        d = policy.decision
        self.solo_trust = d["corroboration"]["solo_trust"]
        self.corroboration_floor = d["corroboration"]["floor"]
        self.solo_damp = d["corroboration"]["solo_damp"]
        self.gaming_cap = d["gaming_model_cap"]
        self.lexicon_decides_alone = d["lexicon_decides_alone"]
        self.model_category_min = d["model_category_min"]
        self.alert_floor = min(d["persona_theta"].values())

        self._model_fn = model_fn
        self.use_corroboration = use_corroboration
        self.use_framing = use_framing
        self.use_gaming_cap = use_gaming_cap
        if model_fn is not None or not use_heads:
            self.heads: List[TextHead] = []
        else:
            self.heads = list(heads) if heads is not None else [TextHead(m) for m in DEFAULT_HEADS]
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.heads)), thread_name_prefix="c2-head")

    @property
    def name(self) -> str:
        if self._model_fn is not None:
            return "lexicon+injected"
        live = [h.name for h in self.heads if h.loaded]
        return "lexicon+" + "+".join(live) if live else "lexicon"

    def warm_up(self) -> None:
        """Load the heads now rather than on the first real message."""
        for head in self.heads:
            head.read("warm up")

    # -- scoring --------------------------------------------------------------

    def score(self, text: str) -> ScoreDetail:
        lex = self.lexicon.score(text)
        category = lex.category

        # At or above the HIGH band the heads cannot change the outcome, so they
        # are not run: explicit abuse is the *cheapest* path through Stage 1.
        if lex.score >= self.lexicon_decides_alone:
            model_score, model_category, labels, head_scores, corr = None, None, {}, {}, ""
        else:
            model_score, model_category, labels, head_scores, corr = self._model_reading(text)

        gaming_capped = False
        if (self.use_gaming_cap and lex.gaming and model_score is not None
                and model_score > self.gaming_cap):
            model_score, model_category, gaming_capped = self.gaming_cap, None, True

        combined = lex.score
        if model_score is not None:
            combined = max(lex.score, model_score)
            if not lex.hits and model_score >= self.model_category_min:
                # A binary head says "harmful" without saying what kind, so fall
                # back to the least specific label rather than claim hate.
                category = model_category or "bullying"

        detail = ScoreDetail(
            score=round(combined, 4),
            category=category,
            hits=list(lex.hits),
            lexicon_score=round(lex.score, 4),
            model_score=model_score,
            head_scores=head_scores,
            model_category=model_category,
            model_labels=labels,
            corroboration=corr,
            gaming_capped=gaming_capped,
        )

        if self.use_framing:
            capped, mark = self.framing.apply(detail.score, text, category=detail.category)
            if capped < detail.score:
                detail.discounted_from = detail.score
                detail.score = round(capped, 4)
                detail.framing_reason = mark.reason
                if detail.score < self.alert_floor:
                    detail.category = "none"
            elif mark.reporting:
                detail.framing_reason = mark.reason
        return detail

    def _model_reading(self, text: str):
        if not (text or "").strip():
            return None, None, {}, {}, ""
        if self._model_fn is not None:
            try:
                return float(self._model_fn(text)), None, {}, {}, "single_head"
            except Exception:
                return None, None, {}, {}, ""
        if not self.heads:
            return None, None, {}, {}, ""

        futures = [(h, self._pool.submit(h.read, text)) for h in self.heads]
        readings: List[Reading] = []
        head_scores: Dict[str, float] = {}
        for head, future in futures:
            try:
                score, category, labels = future.result()
            except Exception:
                continue
            if score is not None:
                readings.append((float(score), category, labels))
                head_scores[head.name] = round(float(score), 4)
        if not readings:
            return None, None, {}, {}, ""

        best_score, best_category, best_labels = max(readings, key=lambda r: r[0])
        if len(readings) < 2:
            return best_score, best_category, best_labels, head_scores, "single_head"
        if not self.use_corroboration:
            return best_score, best_category, best_labels, head_scores, "max"
        if best_score >= self.solo_trust:
            return best_score, best_category, best_labels, head_scores, "solo_trusted"
        if min(r[0] for r in readings) >= self.corroboration_floor:
            return best_score, best_category, best_labels, head_scores, "corroborated"
        # One head alone, only moderately confident: damp rather than trust.
        return round(best_score * self.solo_damp, 4), best_category, best_labels, head_scores, "damped"

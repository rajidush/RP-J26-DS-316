"""
EVALUATE state: rule-based emotional_state classifier.

Replaces the word-count placeholder (which could never output `distressed`
and labelled "It's not my fault." as calm because it has 4 words). Still a
transparent baseline, not a trained model: every label can be explained by
the exact phrases that fired, which is also what Component 4's XAI layer
needs. A Gemma EVALUATE classifier can replace classify() later without
changing the EvaluateOutput contract.

Rules were written from development children C001-C020 of the simulated
dataset only; children C021-C040 are held out for evaluation
(evidence/evaluate_emotion_rules.py).

Signals, scored over every reply the child gave in the session:
  reflective  the child plans a safer next step     -> evidence for calm
  distress    fear, sadness, loneliness, overwhelm  -> distressed
  defensive   denial, deflection, minimising        -> defensive
  withdrawn   non-answers, opting out               -> unclear
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import EmotionalState


def _rx(*phrases: str, prefix: str = "") -> re.Pattern:
    return re.compile(prefix + r"\b(" + "|".join(phrases) + r")\b")


# Softened feelings ("kinda scary", "a little upset") are not counted as
# distress; distressed children intensify ("so scared", "really heavy").
_NOT_SOFTENED = r"(?<!a little )(?<!a bit )(?<!kind of )(?<!kinda )(?<!sort of )"


_APOS = "['’]?"  # "don't", "dont", "don’t"

REFLECTIVE = _rx(
    r"maybe (i|we)", r"i think i", r"i guess i (should|need)", r"i should", r"i need to",
    rf"i{_APOS}ll", r"i will", rf"i{_APOS}m going to", r"i (can|could) (ask|tell|talk|try|draw|write|look|block|be)",
    r"try (to|and)", r"talk to (someone|my|a|you)", r"tell (someone|a|my)", r"ask (a|my|someone)",
    r"grown-?up", r"trusted adult", r"next time", r"instead", r"be (more |extra )?careful",
)
DISTRESS = _rx(
    r"scared", r"scary", r"afraid", r"frightened", r"(?<!leave me )(?<!leave it )alone", r"lonely",
    r"lost", r"upset(ting)?", r"awful", r"terrible", r"sadness",
    r"hurts?", r"heavy", r"my heart", r"breathe", r"cry(ing)?", r"anymore", r"disappear",
    r"overwhelmed", r"falling apart", r"wish i could", r"miss (my|you)", r"judg(ed|ing)",
    r"bad dream", r"want it (all )?to stop", r"so wrong", r"feels? wrong", r"dark", r"sick",
    r"worried", r"panic", r"ugh",
    prefix=_NOT_SOFTENED,
)
DEFENSIVE = _rx(
    rf"not (really )?my (fault|choice|thing)", rf"my fault", rf"wasn{_APOS}t (really )?(me|my)", r"not me",
    rf"i didn{_APOS}t (really )?(do|see|mean|look)", rf"didn{_APOS}t do anything", rf"don{_APOS}t worry about it",
    r"whatever", r"so what", r"it was nothing", r"no big deal", r"just a (little|bit)",
    r"just a bit", r"bit of fun", r"no way", rf"didn{_APOS}t (bother|happen to) me",
    rf"wasn{_APOS}t (even )?looking", r"not (that |very )?important", r"leave it alone",
)
# A refusal: the child opts out of talking. INQUIRE stops asking (14 Oct: the
# withdrawn child was asked again after "leave me alone"), and every refusal is
# also a WITHDRAWN phrase, so EVALUATE scores it unclear, never calm.
_REFUSAL_PHRASES = (
    r"leave me alone", rf"(don{_APOS}t|do not) (really )?(want|feel like) (to )?talk(ing)?", r"stop asking",
    r"go away", r"i (need|want) (some )?space",
)
REFUSAL = _rx(*_REFUSAL_PHRASES)
# "I don't know": INQUIRE asks the same Socratic move again in its easier form, and
# every one is also a WITHDRAWN phrase. The uncertainty phrases were added 14 Oct from
# the 12 Oct end-to-end bug, not from held-out data: three "I'm not sure" replies had
# no signal and fell through to calm.
_DONT_KNOW_PHRASES = (
    rf"i don{_APOS}t (really )?know(?! (what to do|why|how))", r"idk", rf"i don{_APOS}t (get|understand)",
    r"not sure", r"dunno", r"no idea",
)
DONT_KNOW = _rx(*_DONT_KNOW_PHRASES)

WITHDRAWN = _rx(
    *_REFUSAL_PHRASES, *_DONT_KNOW_PHRASES,
    r"quiet", r"weird", r"nope", r"nothing", r"(some )?space", r"boring",
    r"a lot", rf"what{_APOS}s that", r"i guess(?! i (should|need))",
)
_SIGNALS = {"reflective": REFLECTIVE, "distress": DISTRESS, "defensive": DEFENSIVE, "withdrawn": WITHDRAWN}

# The PP1 self-regulation keywords behind `self_regulation_shown` (moved here
# from the controller on 26 Oct, unchanged). Whole words: "i could" must not
# fire inside "i couldn't".
SELF_REGULATION = re.compile(r"\b(instead|next time|should have|i could)\b")


@dataclass
class EmotionVerdict:
    state: EmotionalState
    hits: dict[str, list[str]] = field(default_factory=dict)  # signal -> phrases that fired

    @property
    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.hits.items()}


def is_refusal(reply: str) -> bool:
    return bool(REFUSAL.search((reply or "").lower()))


def states_something(reply: str) -> bool:
    """True if one reply states a feeling, a denial or a safer plan, however short
    ("I feel sick"). A withdrawn phrase ("idk") is not enough."""
    text = (reply or "").lower()
    return any(_SIGNALS[name].search(text) for name in ("distress", "defensive", "reflective"))


def is_dont_know(reply: str) -> bool:
    """An "I don't know" reply: INQUIRE asks the move again in its easier form. One
    that also states something ("idk, it was so scary") is answered by the next move."""
    return bool(DONT_KNOW.search((reply or "").lower())) and not states_something(reply)


def signal_hits(text: str) -> dict[str, list[str]]:
    """Phrases each signal fires on in `text`, for every signal (empty when none fired)."""
    text = (text or "").lower()
    return {name: [m.group(0) for m in rx.finditer(text)] for name, rx in _SIGNALS.items()}


def classify(replies: list[str]) -> EmotionVerdict:
    text = " ".join(r for r in replies if r).lower()
    if not text.strip():
        return EmotionVerdict(EmotionalState.UNCLEAR)

    hits = signal_hits(text)
    n = {k: len(v) for k, v in hits.items()}

    # A child who names a feeling AND plans a safer next step is regulating:
    # calm. Distress only wins when it outweighs the planning.
    if n["reflective"] and n["reflective"] >= n["distress"]:
        state = EmotionalState.CALM
    elif n["distress"] or n["defensive"] or n["withdrawn"]:
        # Ties go distressed > defensive > unclear: missing distress is the costliest error.
        best = max(("distress", "defensive", "withdrawn"), key=lambda k: (n[k], -("distress", "defensive", "withdrawn").index(k)))
        state = {"distress": EmotionalState.DISTRESSED, "defensive": EmotionalState.DEFENSIVE,
                 "withdrawn": EmotionalState.UNCLEAR}[best]
    elif all(len(r.split()) <= 2 for r in replies if r.strip()):
        # Only minimal replies and no signal at all ("ok", "no"): not enough to judge.
        # (A bare refusal is not treated as defensive.)
        state = EmotionalState.UNCLEAR
    else:
        state = EmotionalState.CALM  # a plain explanation, e.g. "I clicked a link by accident"
    return EmotionVerdict(state, {k: v for k, v in hits.items() if v})

"""Self-harm specialist: is this the writer's own risk? (proposal FR2 `self_harm`)

Why it exists: the two text heads are hate / toxicity models, so self-harm
rested on a short keyword list. On the self-harm set
(`evaluation/self_harm_set.py`, written before this fix) C2 caught 6 of 20 dev
and 0 of 20 held-out messages, all by keyword.

A suicidality model (sentinet/suicidality: ELECTRA, CC0) reads suicide
language wherever it appears. It scored "kill yourself already loser" 0.99
and a child reporting "someone in the group chat told me to kill myself and
i'm scared" 0.98. So its reading counts only for the writer's own statement:

    first person        "i / me / my"                   else it is about someone else
    not other-directed  not "yourself", "you should"    else it is abuse aimed at someone
    not reported        no framing attribution          else it is someone else's words
    not hyperbole       not gaming / figurative context as the lexicon's ambiguous family,
                        and not the idiom "kill / die for X" ("i'd kill for some sleep")

and alone only at the corroboration rule's solo_trust (0.90), like any single
head. The rules live in `packs/policy/decision.json` (`self_harm_model`).

A gate decides whether the reading is about the writer; it never lowers a
self-harm score that is (invariant I-3). Explicit first-person intent is still
caught by the lexicon's `self_harm:intent` family, which no gate touches.

Measured with `evaluation/self_harm.py` (18 Oct 2026), the same at ages 9, 12
and 15, and no change to any of the 321 decisions on the 107 existing cases:

    dev self-harm set         18 / 20 caught (was 6),  0 / 17 false alerts
    held-out, first report    16 / 20 caught (was 0),  1 / 16 ("i'd kill for some sleep")
    held-out, after idiom     16 / 20 caught,          0 / 16 (that case moved to dev)

The model ships as ONNX with only its embedding table in INT8: full MatMul
INT8 flipped decisions in the export gate, the embedding-only variant none.
"""
from __future__ import annotations

import re
from typing import Optional

from .framing import FramingGuard
from .lexicon import Lexicon
from .policy import Policy, default_policy


class SelfHarmGate:
    def __init__(self, policy: Optional[Policy] = None, lexicon: Optional[Lexicon] = None,
                 framing: Optional[FramingGuard] = None) -> None:
        policy = policy or default_policy()
        rules = policy.decision["self_harm_model"]
        self.model_id = rules["model"]
        self.trust = float(rules["trust"])
        self._first = re.compile(rules["first_person"], re.IGNORECASE)
        self._other = re.compile(rules["other_directed"], re.IGNORECASE)
        self._hyperbole = re.compile(rules["hyperbole"], re.IGNORECASE)
        self._contexts = tuple(rules["suppressed_in"])
        self.lexicon = lexicon or Lexicon(policy)
        self.framing = framing or FramingGuard(policy)

    def blocked_by(self, text: str) -> str:
        """Why a self-harm reading is not about the writer, or "" when it is."""
        blob = self.lexicon.normalize(text)
        if not self._first.search(blob):
            return "not_first_person"
        if self._other.search(blob):
            return "aimed_at_someone_else"
        if "attribution" in self.framing.detect(text).signals:
            return "reported_speech"
        if self._hyperbole.search(blob):
            return "idiom"
        for context in self._contexts:
            if self.lexicon.in_context(context, text):
                return f"{context}_context"
        return ""

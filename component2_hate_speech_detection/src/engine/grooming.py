"""Grooming specialist: does this message use a grooming tactic? (proposal FR2 `grooming`)

Why it exists: grooming is polite. The hate and toxicity heads score it near
zero, and the lexicon's tactic families are word patterns. Written on 20 dev
lines, they caught 18 / 20 there but 4 / 20 of the held-out lines, which say
the same tactics in other words (`evaluation/grooming_set.py`).

A zero-shot NLI model (MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33,
MIT) is asked one question per tactic, each written from the online-grooming
stages (O'Connell 2003; Winters & Jeglic 2017), not from the test lines:
hiding the contact from parents, isolation, age flattery, a risk check, a
sexual or image request, a reward, a secret meeting. Its reading counts when

    on topic       the message mentions a parent, a secret, meeting, an image,
                   a gift or how the child feels; otherwise the model is not
                   run at all (it costs ~160 ms a message)
    not a surprise not a birthday or party being kept quiet: the model read
                   "don't tell mum about her birthday present" as secrecy (0.66)

and the strongest tactic reaches the policy's trust (0.6: the highest reading
on an ordinary dev line was 0.46). The rules live in
`packs/policy/decision.json` (`grooming_model`). A gate decides only whether
the model is asked or listened to; it never lowers a grooming score the
lexicon gave (framing never discounts grooming either).

Measured with `evaluation/grooming.py` (18 Oct 2026), the same at ages 9, 12
and 15, every alert urgent:

                          rules only    rules + model
    dev (40 / 32)         22 / 40       27 / 40        1 / 32 false alerts (the hate head, not grooming)
    fresh held-out        2 / 20        6 / 20         1 / 16 false alerts
    107 existing cases    the model changes one decision: "keep this between us dont mention
                          it to mum" (labelled grooming) is now caught

The held-out false alert is a rule, not the model: "dont tell your mom about
the cake we're making for her" fires grooming:hide_contact (it was a sexual
harassment alert before). Per-message grooming detection stays weak: 14 of
20 held-out lines are still missed, e.g. "send me a pic of you in your pjs,
our secret" (model 0.13). Fine-tuning on grooming data, or a
conversation-level model, is the next step.

The model ships as ONNX with only its embedding table in INT8 (the MatMul
INT8 changed 6 decisions); with it every C2 model loaded peaks at 1,854 MB.
A message on a grooming topic takes ~160 ms (median), others ~40 ms.

Sony's grooming-risk classifier (trained on NCMEC-reported cases) was looked
at and not used here: its card says single messages are out of distribution,
and at 1.2 GB it would break the memory budget. It fits a conversation-level
check later.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

from .lexicon import Lexicon
from .policy import Policy, default_policy


class GroomingGate:
    def __init__(self, policy: Optional[Policy] = None, lexicon: Optional[Lexicon] = None) -> None:
        policy = policy or default_policy()
        rules = policy.decision["grooming_model"]
        self.model_id = rules["model"]
        self.trust = float(rules["trust"])
        self.hypotheses: Dict[str, str] = dict(rules["hypotheses"])
        self._topics = re.compile(rules["topics"], re.IGNORECASE)
        self._contexts = tuple(rules["suppressed_in"])
        self.lexicon = lexicon or Lexicon(policy)

    def on_topic(self, text: str) -> bool:
        """Worth asking the model at all."""
        return bool(self._topics.search(self.lexicon.normalize(text)))

    def blocked_by(self, text: str) -> str:
        """Why a grooming reading does not count, or "" when it does."""
        if not self.on_topic(text):
            return "off_topic"
        for context in self._contexts:
            if self.lexicon.in_context(context, text):
                return f"{context}_context"
        return ""

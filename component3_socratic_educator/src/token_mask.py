"""
Finite-automaton token masking for the INTERCEPT controller grammar.

The child-facing path (grammar_decoder, ADR 0004) delegates masking to LM
Studio's llama.cpp engine through a JSON schema. This module is Component 3's
own implementation of the same idea, written to show the mechanism step by
step (evidence/trace_token_masking.py); it is not on the child-facing path.

    InterceptAutomaton   a finite automaton over characters that accepts exactly
                         the INTERCEPT line grammar: a fixed opener, one space,
                         then [A-Z][A-Za-z ,']{5,98}?
    TokenMask            for an automaton state, the set of vocabulary tokens
                         whose text keeps the automaton alive; end-of-generation
                         only once the line is complete
    MaskingLogitsProcessor  a transformers logits processor that sets every
                         other token's score to -inf before sampling, and
                         records what it blocked

The automaton works on characters and the model on tokens, so a token is
allowed when the automaton survives every character of it (a token such as
" What" crosses from the opener's trailing space into the question).
"""
from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Iterable, Optional, Sequence

from .grammar_decoder import INTERCEPT_OPENERS

# The pieces of grammar_decoder._QUESTION: [A-Z][A-Za-z ,']{5,98}\?
# tests/test_token_mask.py checks the automaton against that regex, so the two
# cannot drift apart unnoticed.
_FIRST = frozenset(string.ascii_uppercase)
_BODY = frozenset(string.ascii_letters + " ,'")
_BODY_MIN, _BODY_MAX = 5, 98

# States: ("opener", prefix-typed-so-far), ("gap",) after a full opener,
# ("question", characters-of-question-so-far), ("end",) after the "?".
State = tuple


class InterceptAutomaton:
    start: State = ("opener", "")

    def step(self, state: State, ch: str) -> Optional[State]:
        """The state after reading `ch`, or None when no valid line continues this way."""
        kind = state[0]
        if kind == "opener":
            typed = state[1] + ch
            if typed in INTERCEPT_OPENERS:
                return ("gap",)
            return ("opener", typed) if any(o.startswith(typed) for o in INTERCEPT_OPENERS) else None
        if kind == "gap":
            return ("question", 0) if ch == " " else None
        if kind == "question":
            n = state[1]
            if n == 0:
                return ("question", 1) if ch in _FIRST else None
            if ch == "?" and n - 1 >= _BODY_MIN:
                return ("end",)
            if ch in _BODY and n - 1 < _BODY_MAX:
                return ("question", n + 1)
            return None
        return None   # nothing may follow the "?"

    def run(self, text: str, state: Optional[State] = None) -> Optional[State]:
        state = self.start if state is None else state
        for ch in text:
            state = self.step(state, ch)
            if state is None:
                return None
        return state

    def is_accepting(self, state: State) -> bool:
        return state == ("end",)


_BYTE_PIECE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def piece_text(piece: str, special: bool = False) -> Optional[str]:
    """The text a SentencePiece vocabulary piece stands for: "▁" is a space and
    "<0x41>" is the byte 0x41. Special tokens (<start_of_turn> ...) have none."""
    if special:
        return None
    byte = _BYTE_PIECE.fullmatch(piece)
    if byte:
        return chr(int(byte.group(1), 16))
    return piece.replace("▁", " ")


class TokenMask:
    """Which token ids may come next. `token_texts[i]` is token i's text (None
    for tokens with no text); `end_ids` are the end-of-generation tokens."""

    def __init__(self, token_texts: Sequence[Optional[str]], end_ids: Iterable[int],
                 automaton: Optional[InterceptAutomaton] = None):
        self.automaton = automaton or InterceptAutomaton()
        self.texts = token_texts
        self.end_ids = frozenset(end_ids)
        # Bucket by first character: one automaton step rules out a whole bucket,
        # which keeps a 262k-token vocabulary fast enough for a trace.
        self._by_first: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for i, text in enumerate(token_texts):
            if text and i not in self.end_ids:
                self._by_first[text[0]].append((i, text))

    def allowed(self, state: State) -> set[int]:
        out = set(self.end_ids) if self.automaton.is_accepting(state) else set()
        for first, tokens in self._by_first.items():
            if self.automaton.step(state, first) is None:
                continue
            out.update(i for i, text in tokens if self.automaton.run(text, state) is not None)
        return out

    def state_after(self, token_ids: Iterable[int]) -> Optional[State]:
        return self.automaton.run("".join(self.texts[i] or "" for i in token_ids))


class MaskingLogitsProcessor:
    """A transformers LogitsProcessor (batch size 1). Before each token is
    chosen it keeps only the scores of tokens the automaton allows, and appends
    one step to `trace`: the model's top choices before masking, which of them
    were blocked, and the probability mass the mask removed."""

    def __init__(self, mask: TokenMask, prompt_len: int, top_k: int = 5):
        self.mask = mask
        self.prompt_len = prompt_len
        self.top_k = top_k
        self.trace: list[dict] = []

    def __call__(self, input_ids, scores):
        import torch

        generated = input_ids[0, self.prompt_len:].tolist()
        state = self.mask.state_after(generated)
        allowed = sorted(self.mask.allowed(state)) if state is not None else []
        if not allowed:
            raise RuntimeError(f"no token can continue the line after {generated!r}")

        probs = torch.softmax(scores[0].float(), dim=-1)
        allowed_set = set(allowed)
        idx = torch.tensor(allowed, dtype=torch.long)
        top = torch.topk(scores[0], min(self.top_k, scores.shape[-1])).indices.tolist()
        self.trace.append({
            "step": len(generated),
            "text_so_far": "".join(self.mask.texts[i] or "" for i in generated),
            "allowed_count": len(allowed),
            "blocked_probability": round(max(0.0, 1.0 - float(probs[idx].sum())), 6),
            "top_before_mask": [{"token_id": i, "text": self.mask.texts[i], "prob": round(float(probs[i]), 6),
                                 "allowed": i in allowed_set} for i in top],
        })
        masked = torch.full_like(scores, float("-inf"))
        masked[:, idx] = scores[:, idx]
        return masked

"""Pretrained text heads (proposal SO1, FR3; Guardian M-TXT teachers).

Two heads, chosen by measurement in the MVP because they fail on different
cases (held-out set, theta 0.35):

    roberta-hate-speech-dynabench-r4   implicit identity hate 40%, bullying  0%
    toxic-bert (Jigsaw)                implicit identity hate 20%, bullying 60%

Ported from the MVP (`analyst/stage1/hf_model.py`). The label logic is shared
by every backend; the backend that runs the network is swappable:

    torch   transformers pipeline (reference; ~1.9 GB resident for both heads)
    onnx    INT8 ONNX Runtime session, no torch import (the shipping form)

Labels are matched by *name*, never by index, and an unrecognised vocabulary
returns None rather than a guess: a fabricated 0.5 from an unknown head
silently poisons the cascade.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Protocol, Tuple

DYNABENCH = "facebook/roberta-hate-speech-dynabench-r4-target"
TOXIC_BERT = "unitary/toxic-bert"
DEFAULT_HEADS = (DYNABENCH, TOXIC_BERT)

HATE_LABELS = {
    "hate", "hateful", "hate_speech", "hatespeech", "toxic", "severe_toxic",
    "obscene", "threat", "insult", "identity_hate", "identity_attack",
    "offensive", "abusive", "harassment", "sexual_explicit", "label_1", "1",
}
SAFE_LABELS = {
    "nothate", "not_hate", "non_toxic", "nontoxic", "neutral", "benign",
    "normal", "clean", "ok", "label_0", "0",
}
# Multi-label heads carry the category directly, which is better evidence than
# guessing one from a binary score.
LABEL_TO_CATEGORY = {
    "identity_hate": "hate_identity",
    "identity_attack": "hate_identity",
    "hate": "hate_identity",
    "hate_speech": "hate_identity",
    "threat": "threat",
    "insult": "bullying",
    "harassment": "bullying",
    "sexual_explicit": "sexual_harassment",
    "obscene": "profanity",
    "severe_toxic": "bullying",
    "toxic": "bullying",
}
CATEGORY_MIN_CONFIDENCE = 0.50

# A 128-token head cannot see a whole screen, and OCR chrome (tab titles, long
# URLs) fills the first hundreds of characters. Scoring overlapping windows and
# keeping the most harmful one stops a slur at char 700 being invisible (the
# MVP measured 0.9959 at char 36 falling to 0.08 at char 740 under truncation).
WINDOW_CHARS = int(os.environ.get("C2_TEXT_WINDOW", "300"))
WINDOW_OVERLAP = int(os.environ.get("C2_TEXT_WINDOW_OVERLAP", "60"))
MAX_WINDOWS = int(os.environ.get("C2_TEXT_MAX_WINDOWS", "16"))

LabelRows = List[dict]  # [{"label": str, "score": float}, ...]


def normalize_label(label) -> str:
    return str(label or "").strip().lower().replace(" ", "_").replace("-", "_")


def hate_score_from_labels(rows: LabelRows) -> Optional[float]:
    """Max over harmful labels (multi-label sigmoids), else 1 - max(safe) (softmax)."""
    hate = [float(r["score"]) for r in rows if normalize_label(r.get("label")) in HATE_LABELS]
    if hate:
        return round(max(hate), 4)
    safe = [float(r["score"]) for r in rows if normalize_label(r.get("label")) in SAFE_LABELS]
    if safe:
        return round(1.0 - max(safe), 4)
    return None


def category_from_labels(rows: LabelRows) -> Optional[str]:
    best_name, best_score = None, 0.0
    for row in rows:
        name = normalize_label(row.get("label"))
        if name in LABEL_TO_CATEGORY and float(row.get("score") or 0.0) > best_score:
            best_name, best_score = name, float(row.get("score") or 0.0)
    if best_name is None or best_score < CATEGORY_MIN_CONFIDENCE:
        return None
    # A bare toxic win names no kind of harm; prefer a specific sibling close behind.
    if best_name in ("toxic", "severe_toxic"):
        for row in rows:
            name = normalize_label(row.get("label"))
            if name in LABEL_TO_CATEGORY and name not in ("toxic", "severe_toxic"):
                if float(row.get("score") or 0.0) >= CATEGORY_MIN_CONFIDENCE:
                    return LABEL_TO_CATEGORY[name]
    return LABEL_TO_CATEGORY[best_name]


def windows(text: str) -> List[str]:
    text = (text or "").strip()
    if len(text) <= WINDOW_CHARS:
        return [text]
    step = WINDOW_CHARS - WINDOW_OVERLAP
    out: List[str] = []
    for start in range(0, len(text), step):
        chunk = text[start:start + WINDOW_CHARS]
        if chunk.strip():
            out.append(chunk)
        if len(out) >= MAX_WINDOWS:
            break
    return out


class Backend(Protocol):
    """Runs one model over a batch of texts and returns full label distributions."""

    name: str

    def __call__(self, texts: List[str]) -> List[LabelRows]: ...


class TorchBackend:
    """transformers pipeline on CPU. The reference the ONNX backend is checked against."""

    def __init__(self, model_id: str, max_length: int = 128) -> None:
        from transformers import pipeline

        self.name = f"torch:{model_id.split('/')[-1]}"
        self._pipe = pipeline(
            "text-classification",
            model=model_id,
            tokenizer=model_id,
            device=-1,  # CPU only: the whole project is a CPU claim
            truncation=True,
            max_length=max_length,
            top_k=None,  # full distribution, so multi-label heads work
        )

    def __call__(self, texts: List[str]) -> List[LabelRows]:
        out = self._pipe(texts)
        return [entry if isinstance(entry, list) else [entry] for entry in out]


class TextHead:
    """One pretrained head. Loads lazily and never raises into the cascade."""

    def __init__(self, model_id: str, backend: str = "auto") -> None:
        self.model_id = model_id
        self.backend_kind = backend
        self._backend: Optional[Backend] = None
        self._load_attempted = False
        self.last_error = ""

    @property
    def name(self) -> str:
        return self._backend.name if self._backend else f"unloaded:{self.model_id.split('/')[-1]}"

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    def _ensure_loaded(self) -> bool:
        if self._backend is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        try:
            self._backend = make_backend(self.model_id, self.backend_kind)
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            return False

    def read(self, text: str) -> Tuple[Optional[float], Optional[str], Dict[str, float]]:
        """(hate score, category, top labels) from one batched pass over all windows.

        Labels ride along from the same inference that produced the score, so
        the evidence shown to a parent is what actually drove the decision.
        """
        if not (text or "").strip() or not self._ensure_loaded():
            return None, None, {}
        try:
            per_window = self._backend(windows(text))  # type: ignore[misc]
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            return None, None, {}
        if not per_window:
            return None, None, {}
        rows = max(per_window, key=lambda r: hate_score_from_labels(r) or 0.0)
        ranked = sorted(rows, key=lambda r: float(r.get("score") or 0.0), reverse=True)
        labels = {normalize_label(r.get("label")): round(float(r.get("score") or 0.0), 4) for r in ranked[:3]}
        return hate_score_from_labels(rows), category_from_labels(rows), labels


def make_backend(model_id: str, kind: str = "auto") -> Backend:
    """`onnx` if an exported model exists (or is requested), else `torch`."""
    kind = os.environ.get("C2_HEAD_BACKEND", kind)
    if kind in ("onnx", "auto"):
        from .onnx_backend import OnnxBackend, onnx_model_dir

        if kind == "onnx" or onnx_model_dir(model_id).exists():
            return OnnxBackend(model_id)
    return TorchBackend(model_id)

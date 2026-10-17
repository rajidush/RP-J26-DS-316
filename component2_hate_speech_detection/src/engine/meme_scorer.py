"""Meme scorer: one image + its caption -> P(hateful) (proposal SO3, FR5).

Runs the exported meme model (frozen CLIP ViT-B/32 + the trained align-fusion
head; see `meme_model/README.md`) on ONNX Runtime with the `tokenizers`
library, so the running app never imports torch, like the text heads.

Export and parity-check with:
    python -m component2_hate_speech_detection.tools.export_meme_onnx

The score becomes a `vision` Signal. Whether it may move a decision is not
decided here: `calibrated` comes from the I-1 gate result that the export
copied into meme.json (`meme_model/calibrate.py`). Today it is False, so the
score is evidence only.

Like OCR, the scorer never raises into the caller: a missing or broken model
returns no signal and the rest of C2 keeps running (proposal NFR8). The model
loads on first use, so an agent that never checks an image pays no memory.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from .fusion import Signal
from .onnx_backend import INTRA_OP_THREADS, MODELS_DIR

MODEL_NAME = "meme-clip-b32-align"


def meme_model_dir() -> Path:
    return MODELS_DIR / MODEL_NAME


def preprocess(image: Image.Image, size: int, mean: List[float], std: List[float]) -> np.ndarray:
    """CLIP preprocessing: shortest side to `size` (bicubic), centre crop, normalise -> (3, size, size).

    Matches transformers' CLIPImageProcessor, which the training embeddings
    used: the long side is floored, not rounded. Rounding shifted the crop by
    a pixel and moved one val meme's score by 0.19.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width <= height:
        new_size = (size, int(size * height / width))
    else:
        new_size = (int(size * width / height), size)
    rgb = rgb.resize(new_size, Image.BICUBIC)
    left, top = (rgb.width - size) // 2, (rgb.height - size) // 2
    rgb = rgb.crop((left, top, left + size, top + size))
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    pixels = (pixels - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return pixels.transpose(2, 0, 1)


class MemeScorer:
    def __init__(self, root: Optional[Path] = None, quantized: Optional[bool] = None) -> None:
        """`quantized=None` uses INT8 when the export kept it (it passed the export gate)."""
        self.root = root or meme_model_dir()
        self._quantized = quantized
        self._session = None
        self._lock = threading.Lock()
        self.meta: dict = {}
        self.name = MODEL_NAME
        self.calibrated = False
        self.last_error = ""

    @property
    def available(self) -> bool:
        return (self.root / "meme.json").exists() and (self.root / "model.onnx").exists()

    def _load(self) -> bool:
        if self._session is not None:
            return True
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            meta = json.loads((self.root / "meme.json").read_text(encoding="utf-8"))
            quantized = (self.root / "model.int8.onnx").exists() if self._quantized is None else self._quantized
            tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
            tokenizer.enable_truncation(max_length=int(meta["max_length"]))
            options = ort.SessionOptions()
            options.intra_op_num_threads = INTRA_OP_THREADS
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            model_file = self.root / ("model.int8.onnx" if quantized else "model.onnx")
            self._session = ort.InferenceSession(str(model_file), options, providers=["CPUExecutionProvider"])
            self._tok, self.meta = tokenizer, meta
            self.calibrated = bool(meta.get("calibrated", False))
            self.name = f"onnx-int8:{MODEL_NAME}" if quantized else f"onnx:{MODEL_NAME}"
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            return False

    def probability(self, image: Optional[Image.Image], caption: str = "") -> Optional[float]:
        """Temperature-scaled P(hateful) for one meme, or None when the model is unavailable."""
        if image is None:
            return None
        with self._lock:
            if not self._load():
                return None
            try:
                spec = self.meta["image"]
                encoding = self._tok.encode(" ".join((caption or "").split()))
                feeds = {
                    "pixel_values": preprocess(image, spec["size"], spec["mean"], spec["std"])[None],
                    "input_ids": np.array([encoding.ids], dtype=np.int64),
                    "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
                }
                logit = float(self._session.run(None, feeds)[0][0])
                return float(1.0 / (1.0 + np.exp(-logit / float(self.meta["temperature"]))))
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"[:200]
                return None

    def score(self, image: Optional[Image.Image], caption: str = "") -> Optional[Signal]:
        p = self.probability(image, caption)
        return None if p is None else Signal("vision", round(p, 4), self.calibrated, self.name)

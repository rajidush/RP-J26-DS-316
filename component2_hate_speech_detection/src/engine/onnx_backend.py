"""ONNX Runtime backend for the text heads (Guardian GS-010, FR-C2-02 ship form).

Runs an exported INT8 model with onnxruntime + the `tokenizers` library, so
the running app never imports torch. That is where most of the memory goes:
both heads under torch cost ~1.9 GB resident on the reference laptop.

Export and parity-check with:
    python -m component2_hate_speech_detection.tools.export_onnx

Activation follows the model config, matching what the transformers pipeline
does: sigmoid for multi-label heads (toxic-bert), softmax otherwise.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "onnx"
INTRA_OP_THREADS = int(os.environ.get("C2_ORT_THREADS", "2"))


def onnx_model_dir(model_id: str) -> Path:
    return MODELS_DIR / model_id.split("/")[-1]


class OnnxBackend:
    def __init__(self, model_id: str, max_length: int = 128, quantized: bool = True) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        root = onnx_model_dir(model_id)
        model_file = root / ("model.int8.onnx" if quantized else "model.onnx")
        if not model_file.exists():
            raise FileNotFoundError(f"{model_file} not found; run tools/export_onnx.py first")

        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        self._labels = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
        self._multi_label = config.get("problem_type") == "multi_label_classification"

        self._tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        self._tok.enable_truncation(max_length=max_length)
        pad_id = int(config.get("pad_token_id") or 0)
        self._tok.enable_padding(pad_id=pad_id, pad_token=self._tok.id_to_token(pad_id) or "[PAD]")

        options = ort.SessionOptions()
        options.intra_op_num_threads = INTRA_OP_THREADS
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(model_file), options, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}
        self.name = f"onnx-int8:{root.name}" if quantized else f"onnx:{root.name}"

    def __call__(self, texts: List[str]) -> List[List[dict]]:
        encodings = self._tok.encode_batch(texts)
        feeds = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        logits = self._session.run(None, feeds)[0].astype(np.float64)
        if self._multi_label:
            probs = 1.0 / (1.0 + np.exp(-logits))
        else:
            shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = shifted / shifted.sum(axis=1, keepdims=True)
        return [
            [{"label": label, "score": float(p)} for label, p in zip(self._labels, row)]
            for row in probs
        ]

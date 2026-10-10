"""Export the text heads to ONNX, quantise where it is safe, and prove it (Guardian GS-010).

    python -m component2_hate_speech_detection.tools.export_onnx

For each head, `models/onnx/<name>/` (git-ignored) gets:
    model.onnx        fp32 export (must match torch: max score diff <= 1e-4)
    model.int8.onnx   only if INT8 passes the decision gate below
    tokenizer.json, config.json

The decision gate. A head's score moving a little does not matter; a child's
outcome changing does. INT8 is kept for a head only if, with that head
quantised, the full engine gives the same rung and category as fp32 for every
gate text at ages 9, 12 and 15. Measured on 10 Oct 2026 (132 texts x 3 ages):

    dynamic INT8, default             dynabench 10 head flips, toxic-bert 12
    MatMul-only + per-channel         dynabench  4 head flips -> 6 engine changes  (rejected)
                                      toxic-bert 7 head flips -> 0 engine changes  (kept)

So toxic-bert ships INT8 and Dynabench ships fp32 ONNX. The result is written
to `evaluation/results/onnx_parity.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from component2_hate_speech_detection.evaluation.dev_set import all_cases as dev_cases
from component2_hate_speech_detection.evaluation.heldout_set import all_cases as heldout_cases
from component2_hate_speech_detection.src.engine.analyzer import Analyzer
from component2_hate_speech_detection.src.engine.heads import DEFAULT_HEADS, TextHead, hate_score_from_labels
from component2_hate_speech_detection.src.engine.onnx_backend import OnnxBackend, onnx_model_dir
from component2_hate_speech_detection.src.engine.text_scorer import TextScorer

RESULTS = Path(__file__).resolve().parents[1] / "evaluation" / "results" / "onnx_parity.json"
INT8_SETTINGS = dict(op_types_to_quantize=["MatMul"], per_channel=True)
GATE_AGES = (9, 12, 15)

PARITY_TEXTS = [
    "you should kys", "nobody likes you go back to your country", "they breed like vermin",
    "all of them are animals and should be caged", "people like you shouldn't be allowed in this country",
    "you're so pathetic, everyone laughs at you", "send me pics and don't tell your parents",
    "i'm going to hurt you after school", "gg ez you're trash lol", "im gonna kill you in this match",
    "this homework is killing me", "want to play minecraft after school?",
    "miss he keeps calling me a retard what do i do", "we learned about hate speech in class today",
    "can you help me with my science project", "their kind always ruins everything they touch",
    "i hate all vegetables honestly", "shut up you idiot", "your mom is so fat lol",
    "nice shot, well played everyone", "she is so stupid and ugly nobody wants her here",
    "muslims are terrorists", "the article discusses racism in schools",
    "wait till i catch you outside", "lol you're such a noob",
]


def gate_texts() -> list[str]:
    return PARITY_TEXTS + [c.text for c in dev_cases()] + [c.text for c in heldout_cases()]


def export_fp32(model_id: str, opset: int = 17) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = onnx_model_dir(model_id)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
    tokenizer.save_pretrained(out)      # tokenizer.json (fast tokenizer)
    model.config.save_pretrained(out)   # id2label + problem_type decide softmax vs sigmoid

    sample = tokenizer(["export sample text", "a second, longer sample sentence"], padding=True, return_tensors="pt")
    names = ["input_ids", "attention_mask"] + (["token_type_ids"] if "token_type_ids" in sample else [])
    dynamic = {n: {0: "batch", 1: "sequence"} for n in names}
    dynamic["logits"] = {0: "batch"}
    torch.onnx.export(
        model, tuple(sample[n] for n in names), str(out / "model.onnx"),
        input_names=names, output_names=["logits"], dynamic_axes=dynamic,
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    for extra in out.iterdir():
        if extra.name not in {"model.onnx", "tokenizer.json", "config.json"}:
            extra.unlink()
    return out


def quantize_int8(model_id: str) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    root = onnx_model_dir(model_id)
    target = root / "model.int8.onnx"
    quantize_dynamic(str(root / "model.onnx"), str(target), weight_type=QuantType.QInt8, **INT8_SETTINGS)
    return target


def _head(model_id: str, quantized: bool) -> TextHead:
    head = TextHead(model_id)
    head._backend = OnnxBackend(model_id, quantized=quantized)
    return head


def _decisions(int8_heads: set[str], texts: list[str]) -> list[tuple]:
    scorer = TextScorer(heads=[_head(m, m in int8_heads) for m in DEFAULT_HEADS])
    analyzer = Analyzer(scorer=scorer)
    return [(analyzer.analyze_text(t, age).rung, analyzer.analyze_text(t, age).top_category)
            for t in texts for age in GATE_AGES]


def _changes(ref: list[tuple], got: list[tuple], texts: list[str]) -> list[dict]:
    return [
        {"text": texts[i // len(GATE_AGES)], "age": GATE_AGES[i % len(GATE_AGES)], "fp32": list(r), "int8": list(g)}
        for i, (r, g) in enumerate(zip(ref, got)) if r != g
    ]


def fp32_matches_torch(model_id: str, texts: list[str]) -> float:
    from component2_hate_speech_detection.src.engine.heads import TorchBackend

    torch_scores = [hate_score_from_labels(r) for r in TorchBackend(model_id)(texts)]
    onnx_scores = [hate_score_from_labels(r) for r in OnnxBackend(model_id, quantized=False)(texts)]
    return max(abs(a - b) for a, b in zip(torch_scores, onnx_scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-export", action="store_true", help="reuse model.onnx files, redo quantisation + gate")
    args = parser.parse_args()
    texts = gate_texts()
    report: dict = {"date": time.strftime("%Y-%m-%d"), "gate_texts": len(texts), "gate_ages": list(GATE_AGES),
                    "int8_settings": INT8_SETTINGS, "heads": {}}

    for model_id in DEFAULT_HEADS:
        if not args.skip_export:
            print(f"exporting {model_id} ...", file=sys.stderr)
            export_fp32(model_id)
        diff = fp32_matches_torch(model_id, texts)
        if diff > 1e-4:
            raise SystemExit(f"{model_id}: fp32 ONNX differs from torch by {diff:.6f}; export is wrong")
        quantize_int8(model_id)
        report["heads"][model_id] = {"fp32_vs_torch_max_diff": round(diff, 6)}

    # Gate each head's INT8 on its own, holding the other head at fp32.
    reference = _decisions(set(), texts)
    keep: set[str] = set()
    for model_id in DEFAULT_HEADS:
        changes = _changes(reference, _decisions({model_id}, texts), texts)
        passed = not changes
        report["heads"][model_id].update({"int8_engine_changes": len(changes), "int8_kept": passed,
                                          "examples": changes[:6]})
        if passed:
            keep.add(model_id)
        else:
            (onnx_model_dir(model_id) / "model.int8.onnx").unlink()

    combined = _changes(reference, _decisions(keep, texts), texts)
    report["shipped"] = {m: ("int8" if m in keep else "fp32") for m in DEFAULT_HEADS}
    report["shipped_engine_changes_vs_fp32"] = len(combined)
    report["size_mb"] = {
        m: {f.name: round(f.stat().st_size / 2**20, 1) for f in onnx_model_dir(m).glob("*.onnx")}
        for m in DEFAULT_HEADS
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

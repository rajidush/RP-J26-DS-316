"""Export the text heads (and the two specialists) to ONNX, quantise where safe, and prove it (GS-010).

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

So toxic-bert ships INT8 and Dynabench ships fp32 ONNX.

A second, gentler INT8 is tried for a model whose MatMul INT8 fails: only the
embedding table (the Gather op) is quantised. That table is most of a small
model's weights (DeBERTa-v3's 128k-token vocabulary is ~190 MB of its 270 MB),
and a lookup's rounding error is far smaller than a matrix product's. It must
pass the same gate. The first variant that passes is kept as model.int8.onnx.
Measured on 18 Oct 2026 (313 texts x 3 ages): embedding-only INT8 changed 0
decisions for Dynabench, the self-harm and the grooming models (MatMul INT8:
9, 3 and 6), so all four now ship some INT8 and peak memory fell to 1,854 MB
with every C2 model loaded.

The self-harm
specialist (sentinet/suicidality, `engine/self_harm.py`) and the grooming
specialist (an NLI model read on text + hypothesis pairs, `engine/grooming.py`)
go through the same export and the same gate; the gate texts add their test
sets. The result is written to `evaluation/results/onnx_parity.json`.
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
from component2_hate_speech_detection.evaluation.grooming_set import dev_cases as grooming_dev
from component2_hate_speech_detection.evaluation.grooming_set import heldout_cases as grooming_heldout
from component2_hate_speech_detection.evaluation.heldout_set import all_cases as heldout_cases
from component2_hate_speech_detection.evaluation.self_harm_set import dev_cases as self_harm_dev
from component2_hate_speech_detection.evaluation.self_harm_set import heldout_cases as self_harm_heldout
from component2_hate_speech_detection.src.engine.analyzer import Analyzer
from component2_hate_speech_detection.src.engine.heads import (DEFAULT_HEADS, TextHead, ZeroShotHead,
                                                                hate_score_from_labels, normalize_label)
from component2_hate_speech_detection.src.engine.onnx_backend import OnnxBackend, onnx_model_dir
from component2_hate_speech_detection.src.engine.policy import default_policy
from component2_hate_speech_detection.src.engine.text_scorer import TextScorer

RESULTS = Path(__file__).resolve().parents[1] / "evaluation" / "results" / "onnx_parity.json"
INT8_VARIANTS = {
    "matmul": dict(op_types_to_quantize=["MatMul"], per_channel=True),
    "embedding": dict(op_types_to_quantize=["Gather"]),
}
GATE_AGES = (9, 12, 15)
SELF_HARM = default_policy().decision["self_harm_model"]["model"]
GROOMING = default_policy().decision["grooming_model"]["model"]
HYPOTHESES = default_policy().decision["grooming_model"]["hypotheses"]
MODELS = (*DEFAULT_HEADS, SELF_HARM, GROOMING)

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
    return (PARITY_TEXTS + [c.text for c in dev_cases()] + [c.text for c in heldout_cases()]
            + [c.text for c in self_harm_dev() + self_harm_heldout()]
            + [c.text for c in grooming_dev() + grooming_heldout()])


def export_fp32(model_id: str, opset: int = 17) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = onnx_model_dir(model_id)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # fp32 explicitly: some checkpoints (the grooming NLI model) are stored fp16.
    model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=torch.float32).eval()
    tokenizer.save_pretrained(out)      # tokenizer.json (fast tokenizer)
    model.config.save_pretrained(out)   # id2label + problem_type decide softmax vs sigmoid
    # transformers 5 leaves default label names (LABEL_0, LABEL_1) out of config.json;
    # the runtime matches labels by name, so write them explicitly.
    saved = json.loads((out / "config.json").read_text(encoding="utf-8"))
    if "id2label" not in saved:
        saved["id2label"] = {str(i): name for i, name in model.config.id2label.items()}
        saved["label2id"] = {name: i for i, name in model.config.id2label.items()}
        (out / "config.json").write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")

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


def quantize_int8(model_id: str, variant: str) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    root = onnx_model_dir(model_id)
    target = root / "model.int8.onnx"
    quantize_dynamic(str(root / "model.onnx"), str(target), weight_type=QuantType.QInt8, **INT8_VARIANTS[variant])
    return target


def _head(model_id: str, quantized: bool) -> TextHead:
    head = ZeroShotHead(model_id, HYPOTHESES) if model_id == GROOMING else TextHead(model_id)
    head._backend = OnnxBackend(model_id, quantized=quantized)
    return head


def _decisions(int8_heads: set[str], texts: list[str]) -> list[tuple]:
    scorer = TextScorer(heads=[_head(m, m in int8_heads) for m in DEFAULT_HEADS],
                        self_harm_head=_head(SELF_HARM, SELF_HARM in int8_heads),
                        grooming_head=_head(GROOMING, GROOMING in int8_heads))
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

    if model_id == GROOMING:  # NLI: every text against every hypothesis, compared on entailment
        firsts = [t for t in texts for _ in HYPOTHESES]
        seconds = [h for _ in texts for h in HYPOTHESES.values()]

        def score(rows):
            return [next(float(x["score"]) for x in r if normalize_label(x["label"]) == "entailment") for r in rows]

        torch_scores = score(TorchBackend(model_id)(firsts, seconds))
        onnx_scores = score(OnnxBackend(model_id, quantized=False)(firsts, seconds))
    else:
        torch_scores = [hate_score_from_labels(r) for r in TorchBackend(model_id)(texts)]
        onnx_scores = [hate_score_from_labels(r) for r in OnnxBackend(model_id, quantized=False)(texts)]
    return max(abs(a - b) for a, b in zip(torch_scores, onnx_scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-export", action="store_true", help="reuse model.onnx files, redo quantisation + gate")
    args = parser.parse_args()
    texts = gate_texts()
    report: dict = {"date": time.strftime("%Y-%m-%d"), "gate_texts": len(texts), "gate_ages": list(GATE_AGES),
                    "int8_variants": INT8_VARIANTS, "heads": {}}

    for model_id in MODELS:
        if not args.skip_export:
            print(f"exporting {model_id} ...", file=sys.stderr)
            export_fp32(model_id)
        diff = fp32_matches_torch(model_id, texts)
        if diff > 1e-4:
            raise SystemExit(f"{model_id}: fp32 ONNX differs from torch by {diff:.6f}; export is wrong")
        report["heads"][model_id] = {"fp32_vs_torch_max_diff": round(diff, 6)}

    # Gate each model's INT8 variants on their own, holding the others at fp32.
    reference = _decisions(set(), texts)
    keep: dict[str, str] = {}
    for model_id in MODELS:
        tried = {}
        for variant in INT8_VARIANTS:
            quantize_int8(model_id, variant)
            changes = _changes(reference, _decisions({model_id}, texts), texts)
            tried[variant] = {"engine_changes": len(changes), "examples": changes[:6]}
            if not changes:
                keep[model_id] = variant
                break
        else:
            (onnx_model_dir(model_id) / "model.int8.onnx").unlink()
        report["heads"][model_id].update({"int8_tried": tried, "int8_kept": keep.get(model_id, "")})

    combined = _changes(reference, _decisions(set(keep), texts), texts)
    report["shipped"] = {m: (f"int8 ({keep[m]})" if m in keep else "fp32") for m in MODELS}
    report["shipped_engine_changes_vs_fp32"] = len(combined)
    report["size_mb"] = {
        m: {f.name: round(f.stat().st_size / 2**20, 1) for f in onnx_model_dir(m).glob("*.onnx")}
        for m in MODELS
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

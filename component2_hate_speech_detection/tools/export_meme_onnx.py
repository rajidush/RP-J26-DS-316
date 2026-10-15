"""Export the meme model to ONNX, quantise if it is safe, and prove it (Guardian GS-010).

    python -m component2_hate_speech_detection.tools.export_meme_onnx --run meme_clip-b32_align

One graph: pixels + caption tokens -> logit (frozen CLIP ViT-B/32 + the trained
align head). `models/onnx/meme-clip-b32-align/` (git-ignored) gets:
    model.onnx        fp32 export (must match torch: max logit diff <= 1e-3)
    model.int8.onnx   only if INT8 passes the gate below
    tokenizer.json    CLIP BPE tokenizer for the `tokenizers` library
    meme.json         temperature, preprocessing, and the I-1 gate verdict from
                      meme_model/calibrate.py (`calibrated`)

The runtime (src/engine/meme_scorer.py) does its own image preprocessing and
tokenisation, so two more checks run on val through that runtime path,
against the stored-embedding head the results were reported with:
    runtime fp32 vs reference   preprocessing / tokeniser parity
    INT8 gate (set before it was run): kept only if val AUROC drops by
        <= 0.005 and the mean probability moves by <= 0.02 against fp32.
Test (dev_seen) is scored once for the shipped form and chooses nothing.

Measured on val, 15 Oct 2026 (MatMul-only, per-channel, dynamic INT8):

    all MatMuls                AUROC -0.030  mean shift 0.087  (rejected)
    all but the projections    AUROC -0.037  mean shift 0.089  (rejected)
    text tower only            AUROC -0.035  mean shift 0.087  (rejected)
    vision tower only          AUROC +0.000  mean shift 0.012  351 MB (kept)

The text tower is what breaks; its activations have outliers that dynamic
per-tensor activation scales clip. So only the vision tower is quantised.

The result is written to `evaluation/results/meme_onnx_parity.json`.
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

import numpy as np

from component2_hate_speech_detection.meme_model.calibrate import load_head
from component2_hate_speech_detection.meme_model.embed import ENCODERS, IMAGE_ROOT, MANIFEST_DIR, read_manifest
from component2_hate_speech_detection.meme_model.train import RESULTS_DIR, load_splits, predict_logits, sigmoid
from component2_hate_speech_detection.src.engine.meme_scorer import MemeScorer, meme_model_dir

RESULTS = RESULTS_DIR / "meme_onnx_parity.json"
INT8_SETTINGS = dict(op_types_to_quantize=["MatMul"], per_channel=True)
INT8_ONLY_UNDER = "/vision_model/"  # see the measurement above
INT8_MAX_AUROC_DROP = 0.005
INT8_MAX_MEAN_SHIFT = 0.02
TORCH_MAX_LOGIT_DIFF = 1e-3
BANDS = (0.35, 0.55, 0.65, 0.75)  # stage 1 / agreement floor and persona thresholds


def _graph(clip, head):
    import torch

    class MemeGraph(torch.nn.Module):
        """Pixels + caption tokens -> head logit, the same path embed.py + train.py take."""

        def __init__(self) -> None:
            super().__init__()
            self.clip, self.head = clip, head

        def forward(self, pixel_values, input_ids, attention_mask):
            image = self.clip.visual_projection(self.clip.vision_model(pixel_values=pixel_values).pooler_output)
            words = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            return self.head(image, self.clip.text_projection(words))

    return MemeGraph().eval()


def export_fp32(run: str, opset: int = 17) -> tuple[Path, float]:
    """Write model.onnx + tokenizer.json; return the max |logit| difference to torch."""
    import onnxruntime as ort
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    head, checkpoint = load_head(run)
    model_id = ENCODERS[checkpoint["encoder"]]
    out = meme_model_dir()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    clip = CLIPModel.from_pretrained(model_id, local_files_only=True).eval()
    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    processor.tokenizer.save_pretrained(out)
    graph = _graph(clip, head)

    rows = read_manifest(MANIFEST_DIR / "val.jsonl")[:8]
    sample = processor(text=[r["text"] for r in rows], images=[Image.open(IMAGE_ROOT / r["img"]).convert("RGB") for r in rows],
                       return_tensors="pt", padding=True, truncation=True, max_length=77)
    names = ["pixel_values", "input_ids", "attention_mask"]
    dynamic = {"pixel_values": {0: "batch"}, "input_ids": {0: "batch", 1: "sequence"},
               "attention_mask": {0: "batch", 1: "sequence"}, "logit": {0: "batch"}}
    with torch.inference_mode():
        torch.onnx.export(graph, tuple(sample[n] for n in names), str(out / "model.onnx"), input_names=names,
                          output_names=["logit"], dynamic_axes=dynamic, opset_version=opset,
                          do_constant_folding=True, dynamo=False)
        reference = graph(*(sample[n] for n in names)).numpy()
    for extra in out.iterdir():
        if extra.name not in {"model.onnx", "tokenizer.json"}:
            extra.unlink()

    session = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    got = session.run(None, {n: sample[n].numpy().astype(np.float32 if n == "pixel_values" else np.int64)
                             for n in names})[0]
    return out, float(np.max(np.abs(got - reference)))


def write_meta(run: str) -> dict:
    """meme.json: what the runtime needs, and whether the score may fuse (I-1)."""
    _, checkpoint = load_head(run)
    model_id = ENCODERS[checkpoint["encoder"]]
    from transformers import CLIPImageProcessor

    image = CLIPImageProcessor.from_pretrained(model_id, local_files_only=True)
    calibration_file = RESULTS_DIR / f"{run}_calibration.json"
    gate = json.loads(calibration_file.read_text(encoding="utf-8"))["gate"] if calibration_file.exists() else None
    meta = {
        "name": meme_model_dir().name,
        "run": run,
        "encoder": model_id,
        "seed": checkpoint["seed"],
        "temperature": round(float(checkpoint["temperature"]), 4),
        "max_length": 77,
        "image": {"size": int(image.crop_size["height"]), "mean": list(image.image_mean), "std": list(image.image_std)},
        # I-1: only a score that passed the calibration gate may move a decision.
        "calibrated": bool(gate and gate["passes"]),
        "calibration_gate": gate or "not run: python -m component2_hate_speech_detection.meme_model.calibrate",
        "exported": time.strftime("%Y-%m-%d"),
    }
    (meme_model_dir() / "meme.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def quantize_int8() -> Path:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    root = meme_model_dir()
    target = root / "model.int8.onnx"
    graph = onnx.load(str(root / "model.onnx"), load_external_data=False).graph
    vision = [n.name for n in graph.node if n.op_type == "MatMul" and n.name.startswith(INT8_ONLY_UNDER)]
    quantize_dynamic(str(root / "model.onnx"), str(target), weight_type=QuantType.QInt8,
                     nodes_to_quantize=vision, **INT8_SETTINGS)
    return target


def runtime_probabilities(split: str, quantized: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """Score a split through the runtime path; returns (p, y, median ms per meme)."""
    from PIL import Image

    scorer = MemeScorer(quantized=quantized)
    rows = read_manifest(MANIFEST_DIR / f"{split}.jsonl")
    p, ms = [], []
    for r in rows:
        with Image.open(IMAGE_ROOT / r["img"]) as image:
            started = time.perf_counter()
            p.append(scorer.probability(image, r["text"]))
            ms.append((time.perf_counter() - started) * 1000)
    if any(v is None for v in p):
        raise SystemExit(f"runtime scorer failed: {scorer.last_error}")
    return np.array(p), np.array([float(r["label"]) for r in rows]), float(np.median(ms))


def reference_probabilities(run: str, split: str) -> np.ndarray:
    head, checkpoint = load_head(run)
    return sigmoid(predict_logits(head, load_splits(checkpoint["encoder"])[split]) / checkpoint["temperature"])


def band_changes(a: np.ndarray, b: np.ndarray) -> int:
    """Memes whose score lands in a different decision band."""
    return int(np.sum(np.digitize(a, BANDS) != np.digitize(b, BANDS)))


def compare(p: np.ndarray, ref: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score

    return {"auroc": round(float(roc_auc_score(y, p)), 4), "reference_auroc": round(float(roc_auc_score(y, ref)), 4),
            "max_abs_diff": round(float(np.max(np.abs(p - ref))), 4),
            "mean_abs_diff": round(float(np.mean(np.abs(p - ref))), 4),
            "band_changes": band_changes(p, ref)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="meme_clip-b32_align", help="checkpoint name in models/meme/")
    args = parser.parse_args(argv)

    print("exporting fp32 ...", file=sys.stderr)
    root, torch_diff = export_fp32(args.run)
    if torch_diff > TORCH_MAX_LOGIT_DIFF:
        raise SystemExit(f"fp32 ONNX differs from torch by {torch_diff:.6f}; export is wrong")
    meta = write_meta(args.run)
    quantize_int8()

    print("scoring val through the runtime (fp32, int8) ...", file=sys.stderr)
    ref_val = reference_probabilities(args.run, "val")
    fp32_val, y_val, fp32_ms = runtime_probabilities("val", quantized=False)
    int8_val, _, int8_ms = runtime_probabilities("val", quantized=True)
    fp32_cmp = compare(fp32_val, ref_val, y_val)
    int8_cmp = compare(int8_val, fp32_val, y_val)
    int8_cmp["auroc_drop"] = round(fp32_cmp["auroc"] - int8_cmp["auroc"], 4)
    keep = int8_cmp["auroc_drop"] <= INT8_MAX_AUROC_DROP and int8_cmp["mean_abs_diff"] <= INT8_MAX_MEAN_SHIFT
    if not keep:
        (root / "model.int8.onnx").unlink()

    print("scoring test once in the shipped form ...", file=sys.stderr)
    shipped_test, y_test, _ = runtime_probabilities("test", quantized=keep)
    test_cmp = compare(shipped_test, reference_probabilities(args.run, "test"), y_test)

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "run": args.run,
        "fp32_vs_torch_max_logit_diff": round(torch_diff, 6),
        "val_runtime_fp32_vs_reference": fp32_cmp,
        "val_int8_vs_fp32": int8_cmp,
        "int8_gate": {"max_auroc_drop": INT8_MAX_AUROC_DROP, "max_mean_shift": INT8_MAX_MEAN_SHIFT,
                      "settings": {**INT8_SETTINGS, "only_under": INT8_ONLY_UNDER}, "kept": keep},
        "shipped": "int8" if keep else "fp32",
        "test_shipped_vs_reference": test_cmp,
        "median_ms_per_meme": {"fp32": round(fp32_ms, 1), "int8": round(int8_ms, 1)},
        "size_mb": {f.name: round(f.stat().st_size / 2**20, 1) for f in root.glob("*.onnx")},
        "calibrated": meta["calibrated"],
    }
    RESULTS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

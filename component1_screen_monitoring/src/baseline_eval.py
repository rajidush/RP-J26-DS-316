"""
baseline_eval.py — ViT-based evaluation for Component 1 (Screen Monitoring).

Checkpoint: jaranohaal/vit-base-violence-detection
Architecture: vit_base_patch16_224 (timm), trained with timm-style weight keys.

Why timm instead of transformers Auto* classes
-----------------------------------------------
The checkpoint's config.json has no `model_type` key (breaks AutoConfig) and
its safetensors weights use timm key names (`blocks.*`, `patch_embed.*`,
`head.*`) rather than the HuggingFace ViTForImageClassification names
(`vit.layers.*`, `classifier.*`).  Loading via the Auto* API results in every
weight being MISSING / randomly initialised.  Using timm directly avoids the
key remapping entirely.

Label auto-detection
--------------------
The checkpoint's config.json has no label names (only generic LABEL_0 /
LABEL_1).  This script probes a handful of images from each class folder and
assigns the integer index whose mean logit is higher for each class.

Usage
-----
    python baseline_eval.py
    python baseline_eval.py --val-dir ../data/val --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
VAL_DIR     = ROOT / "data" / "val"
RESULTS_DIR = ROOT / "results"
OUT_JSON    = RESULTS_DIR / "baseline_accuracy.json"

CHECKPOINT  = "jaranohaal/vit-base-violence-detection"
TIMM_ARCH   = "vit_base_patch16_224"   # architecture that matches the weight keys
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Human-readable class names (subfolder names in data/val/)
CLASSES = ["violence", "non_violence"]


# ---------------------------------------------------------------------------
# Model loading (timm)
# ---------------------------------------------------------------------------

def _load_model_timm(device: str):
    """
    Load the checkpoint into a timm VisionTransformer.

    Returns (model, transform) where transform is the standard
    ImageNet preprocessing pipeline used during training.
    """
    import timm
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as load_safetensors

    print(f"  Downloading / locating weights for {CHECKPOINT} …")
    weights_path = hf_hub_download(CHECKPOINT, "model.safetensors")
    print(f"  Weights path : {weights_path}")

    # Create a bare timm ViT with 2 output classes
    model = timm.create_model(TIMM_ARCH, num_classes=2, pretrained=False)

    state_dict = load_safetensors(weights_path, device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"  ⚠️  Missing keys  ({len(missing)}): {missing[:3]} …")
    if unexpected:
        print(f"  ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:3]} …")
    if not missing and not unexpected:
        print("  ✅ All weights loaded perfectly (strict match).")

    model.to(device).eval()

    # Standard timm preprocessing for this architecture
    data_config = timm.data.resolve_model_data_config(model)
    transform   = timm.data.create_transform(**data_config, is_training=False)

    return model, transform


# ---------------------------------------------------------------------------
# Label auto-detection
# ---------------------------------------------------------------------------

def _detect_label_order(
    model,
    transform,
    val_dir: Path,
    device: str,
    probe_n: int = 10,
) -> dict[str, int]:
    """
    Probe `probe_n` images from each class to learn which output index
    (0 or 1) corresponds to which folder name.

    Returns
    -------
    dict mapping class folder name → predicted class index
    e.g. {"violence": 1, "non_violence": 0}
    """
    import torch
    from PIL import Image

    class_to_idx: dict[str, int] = {}
    print(f"\n  Auto-detecting label order (probing {probe_n} images/class) …")

    for cls_dir in sorted(val_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        probe = imgs[:probe_n]

        logit_sums = [0.0, 0.0]
        count = 0
        for img_path in probe:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)[0]          # shape (2,)
            logit_sums[0] += logits[0].item()
            logit_sums[1] += logits[1].item()
            count += 1

        if count == 0:
            print(f"  ⚠️  No readable images in {cls_dir.name}, skipping.")
            continue

        mean_logits = [s / count for s in logit_sums]
        dominant_idx = int(mean_logits[1] > mean_logits[0])   # index with higher mean
        class_to_idx[cls_dir.name] = dominant_idx
        print(f"  {cls_dir.name:>20s}  →  index {dominant_idx}  "
              f"(mean logits: [0]={mean_logits[0]:.3f}, [1]={mean_logits[1]:.3f})")

    return class_to_idx


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------

def _collect_images(val_dir: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for cls_dir in sorted(val_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTS:
                pairs.append((img_path, cls_dir.name))
    return pairs


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------

def _run_inference(
    pairs: list[tuple[Path, str]],
    model,
    transform,
    device: str,
    batch_size: int,
    class_to_idx: dict[str, int],
) -> list[dict]:
    import torch
    from PIL import Image

    results = []
    n = len(pairs)

    for start in range(0, n, batch_size):
        batch = pairs[start : start + batch_size]
        tensors, gt_classes = [], []

        for img_path, gt_cls in batch:
            try:
                img = Image.open(img_path).convert("RGB")
                tensors.append(transform(img))
                gt_classes.append(gt_cls)
            except Exception as exc:
                print(f"\n  ⚠️  {img_path.name}: {exc}")

        if not tensors:
            continue

        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(x)                          # (B, 2)
            probs  = torch.softmax(logits, dim=-1)
            pred_idxs = logits.argmax(dim=-1).tolist()

        for i, (pred_idx, gt_cls) in enumerate(zip(pred_idxs, gt_classes)):
            gt_idx  = class_to_idx.get(gt_cls, -1)
            correct = (pred_idx == gt_idx)
            conf    = float(probs[i, pred_idx].item())
            results.append({
                "file":      batch[i][0].name,
                "gt_class":  gt_cls,
                "gt_idx":    gt_idx,
                "pred_idx":  pred_idx,
                "confidence": round(conf, 4),
                "correct":   correct,
            })

        done = min(start + batch_size, n)
        print(f"  [{done:>4d}/{n}] processed …", end="\r")

    print()
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(results: list[dict], class_to_idx: dict[str, int]) -> dict:
    per_class: dict[str, dict] = {}
    for r in results:
        cls = r["gt_class"]
        per_class.setdefault(cls, {"correct": 0, "total": 0})
        per_class[cls]["total"]   += 1
        per_class[cls]["correct"] += int(r["correct"])

    per_class_acc = {
        cls: {
            "correct":  v["correct"],
            "total":    v["total"],
            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
        }
        for cls, v in per_class.items()
    }

    total_correct = sum(v["correct"] for v in per_class.values())
    total_images  = sum(v["total"]   for v in per_class.values())
    overall_acc   = round(total_correct / total_images, 4) if total_images else 0.0

    return {
        "checkpoint":        CHECKPOINT,
        "timm_arch":         TIMM_ARCH,
        "val_dir":           str(VAL_DIR),
        "evaluated_at":      datetime.now(timezone.utc).isoformat(),
        "label_mapping":     class_to_idx,
        "overall_accuracy":  overall_acc,
        "total_correct":     total_correct,
        "total_images":      total_images,
        "per_class":         per_class_acc,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--val-dir",    type=Path, default=VAL_DIR)
    p.add_argument("--batch-size", type=int,  default=8)
    p.add_argument("--out",        type=Path, default=OUT_JSON)
    p.add_argument("--probe-n",    type=int,  default=10,
                   help="Images per class used for label auto-detection")
    return p.parse_args()


def run_eval(val_dir: Path, batch_size: int, out: Path, probe_n: int) -> dict:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Baseline Evaluation — {CHECKPOINT}")
    print(f"  Architecture : {TIMM_ARCH}  (timm)")
    print(f"  Device       : {device}")
    print(f"  Val dir      : {val_dir}")
    print(f"{'='*60}\n")

    # 1. Load model
    model, transform = _load_model_timm(device)

    # 2. Auto-detect label order
    class_to_idx = _detect_label_order(model, transform, val_dir, device, probe_n)

    # 3. Collect all val images
    pairs = _collect_images(val_dir)
    print(f"\n  Found {len(pairs)} validation images across "
          f"{len(set(c for _, c in pairs))} classes.\n")

    # 4. Run inference
    t0 = time.time()
    results = _run_inference(pairs, model, transform, device, batch_size, class_to_idx)
    elapsed = round(time.time() - t0, 1)
    print(f"  Inference time: {elapsed}s  ({elapsed/max(len(results),1):.2f}s/image)\n")

    # 5. Metrics + write JSON
    metrics = _compute_metrics(results, class_to_idx)
    metrics["inference_seconds"] = elapsed

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  ✅ Results written → {out}\n")

    # 6. Summary table
    print(f"{'─'*60}")
    print(f"  Overall accuracy : {metrics['overall_accuracy']*100:.1f}%  "
          f"({metrics['total_correct']}/{metrics['total_images']})")
    for cls, v in metrics["per_class"].items():
        print(f"  {cls:>20s} : {v['accuracy']*100:.1f}%  "
              f"({v['correct']}/{v['total']})")
    print(f"{'─'*60}\n")

    return metrics


if __name__ == "__main__":
    args = _parse_args()
    run_eval(
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        out=args.out,
        probe_n=args.probe_n,
    )

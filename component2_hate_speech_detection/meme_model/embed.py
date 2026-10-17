"""Embed every meme once with a frozen encoder, so heads train in minutes on a CPU.

    python -m component2_hate_speech_detection.meme_model.embed --encoder clip-b32
    python -m component2_hate_speech_detection.meme_model.embed --encoder clip-l14 --allow-download

The encoder never changes during training (Hate-CLIPper and RGCL both freeze
CLIP), so its outputs are computed here once and reused by every head, seed
and ablation. Writes `data/embeddings/<encoder>.npz` (git-ignored) with:

    ids    int64  (N,)       meme ids, in manifest order
    image  float32 (N, d)    CLIP image embedding (joint space, not normalised)
    text   float32 (N, d)    CLIP text embedding of the caption (joint space)

Offline by default: models load from the local Hugging Face cache only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = COMPONENT_ROOT / "data"
IMAGE_ROOT = DATA_DIR / "hateful_memes"
MANIFEST_DIR = DATA_DIR / "labels" / "finetune"
EMBED_DIR = DATA_DIR / "embeddings"
MANIFESTS = ("train", "val", "test", "unlabelled_test_seen")

# The laptop target is B/32; the larger ones are GPU upper bounds for RQ3.
ENCODERS = {
    "clip-b32": "openai/clip-vit-base-patch32",
    "clip-b16": "openai/clip-vit-base-patch16",
    "clip-l14": "openai/clip-vit-large-patch14",
    "clip-l14-336": "openai/clip-vit-large-patch14-336",
}


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def all_memes(manifest_dir: Path = MANIFEST_DIR) -> list[dict]:
    """Every meme across the fine-tuning manifests, once each, in a stable order."""
    seen, rows = set(), []
    for name in MANIFESTS:
        for row in read_manifest(manifest_dir / f"{name}.jsonl"):
            if row["id"] not in seen:
                seen.add(row["id"])
                rows.append(row)
    return rows


def batches(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def embed(rows: list[dict], model_id: str, image_root: Path, batch_size: int = 64,
          allow_download: bool = False) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id, local_files_only=not allow_download).eval()
    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=not allow_download)

    images, texts = [], []
    done, started = 0, time.perf_counter()
    with torch.inference_mode():
        for batch in batches(rows, batch_size):
            pictures = [Image.open(image_root / r["img"]).convert("RGB") for r in batch]
            inputs = processor(text=[r["text"] for r in batch], images=pictures, return_tensors="pt",
                               padding=True, truncation=True, max_length=77)
            # Pooled output -> CLIP's own projection, spelled out because
            # get_*_features changed its return type in transformers 5.
            vision = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            words = model.text_model(input_ids=inputs["input_ids"],
                                     attention_mask=inputs["attention_mask"]).pooler_output
            images.append(model.visual_projection(vision).numpy())
            texts.append(model.text_projection(words).numpy())
            done += len(batch)
            rate = done / (time.perf_counter() - started)
            print(f"\r  {done}/{len(rows)} memes  {rate:.1f}/s", end="", flush=True)
    print()
    return np.concatenate(images).astype(np.float32), np.concatenate(texts).astype(np.float32)


def load_embeddings(encoder: str, embed_dir: Path = EMBED_DIR) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """id -> (image vector, text vector)."""
    data = np.load(embed_dir / f"{encoder}.npz")
    return {int(i): (img, txt) for i, img, txt in zip(data["ids"], data["image"], data["text"])}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", choices=sorted(ENCODERS), default="clip-b32")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="embed only the first N memes (smoke test)")
    parser.add_argument("--allow-download", action="store_true", help="fetch the model if it is not cached")
    args = parser.parse_args(argv)

    rows = all_memes()[: args.limit]
    print(f"{args.encoder}: {ENCODERS[args.encoder]}  memes={len(rows)}")
    started = time.perf_counter()
    image, text = embed(rows, ENCODERS[args.encoder], IMAGE_ROOT, args.batch_size, args.allow_download)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"-first{args.limit}" if args.limit else ""
    out = EMBED_DIR / f"{args.encoder}{suffix}.npz"
    np.savez(out, ids=np.array([r["id"] for r in rows], dtype=np.int64), image=image, text=text)
    print(f"wrote {out}  dim={image.shape[1]}  {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()

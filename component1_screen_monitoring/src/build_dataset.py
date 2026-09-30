"""
build_dataset.py — Train / val splitter for Component 1 (Screen Monitoring).

Reads class-labelled images from data/raw/<class>/ and copies them into
data/train/<class>/ and data/val/<class>/ using a configurable split ratio.

Usage
-----
    python build_dataset.py                      # defaults: 80/20 split, seed 42
    python build_dataset.py --val-ratio 0.25 --seed 7

Directory contract
------------------
    data/raw/
        violence/        *.jpg
        non_violence/    *.jpg

    data/train/
        violence/
        non_violence/

    data/val/
        violence/
        non_violence/
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[1]
RAW_DIR   = ROOT / "data" / "raw"
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR   = ROOT / "data" / "val"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Core split function
# ---------------------------------------------------------------------------

def split_dataset(
    raw_dir:   Path = RAW_DIR,
    train_dir: Path = TRAIN_DIR,
    val_dir:   Path = VAL_DIR,
    val_ratio: float = 0.20,
    seed:      int   = 42,
    overwrite: bool  = False,
) -> dict:
    """
    Randomly split images in *raw_dir* into train and val sets.

    Parameters
    ----------
    raw_dir   : source root containing one subfolder per class
    train_dir : destination root for training images
    val_dir   : destination root for validation images
    val_ratio : fraction of images to reserve for validation (default 0.20)
    seed      : random seed for reproducibility
    overwrite : if True, clear existing train/val dirs before copying

    Returns
    -------
    dict with counts per split and class, e.g.:
        {
          "violence":     {"train": 120, "val": 30},
          "non_violence": {"train": 120, "val": 30},
          "totals":       {"train": 240, "val": 60}
        }
    """
    rng = random.Random(seed)
    summary: dict = {}
    total_train = total_val = 0

    class_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories found in {raw_dir}")

    for cls_dir in class_dirs:
        cls = cls_dir.name
        images = sorted(
            f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )
        if not images:
            print(f"  ⚠️  {cls}: no images found, skipping.")
            continue

        rng.shuffle(images)
        n_val   = max(1, round(len(images) * val_ratio))
        n_train = len(images) - n_val
        val_imgs   = images[:n_val]
        train_imgs = images[n_val:]

        for split, imgs, dst_root in [
            ("train", train_imgs, train_dir),
            ("val",   val_imgs,   val_dir),
        ]:
            dst = dst_root / cls
            if overwrite and dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dst / img.name)

        summary[cls] = {"train": n_train, "val": n_val}
        total_train += n_train
        total_val   += n_val
        print(f"  {cls:>20s}  →  train: {n_train:>4d}  |  val: {n_val:>4d}")

    summary["totals"] = {"train": total_train, "val": total_val}
    return summary


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split data/raw/ into train / val.")
    p.add_argument("--val-ratio",  type=float, default=0.20,
                   help="Fraction for validation (default: 0.20)")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--overwrite",  action="store_true",
                   help="Clear existing train/val dirs before copying")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"\nSplitting {RAW_DIR}  (val_ratio={args.val_ratio}, seed={args.seed})\n")
    summary = split_dataset(
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"\n{'─'*50}")
    print(f"  Total  →  train: {summary['totals']['train']:>4d}  |  "
          f"val: {summary['totals']['val']:>4d}")
    print(f"{'─'*50}")
    print(f"\nManifest:\n{json.dumps(summary, indent=2)}")
    print("\n✅  Dataset split complete.")

"""
pack_train_set.py — Zip the screen TRAINING set for Colab (notebook section 15).

Packs ONLY what the notebook needs, with paths relative to data/screens/ so the
`path` column of screen_train.csv works after unzipping:

    train_composites/composites/<class>/*.png     every composite listed in screen_train.csv
    train_composites/tiles/<class>/*.png          every tile listed in screen_train.csv
    screen_train.csv                              labels, groups, train/val split
    rlvs_frames.csv                               RLVS split + near_test flags (the RLVS
                                                  frames themselves are already in Colab)
    manifest.json                                 counts, so the notebook can check the unzip

Never packed: the TEST set (composite/, scraped_pages/, recorded/, urls.txt,
sources.csv, composites.csv), train_pages/, and model weights. Only files listed in
screen_train.csv are packed, and every archive name is checked against these rules.

Usage
-----
    python component1_screen_monitoring/src/pack_train_set.py
        → component1_screen_monitoring/dist/screen_train_v1.zip
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from .config import COMPONENT_ROOT, SCREENS_DIR
except ImportError:
    from config import COMPONENT_ROOT, SCREENS_DIR

DIST_ZIP = COMPONENT_ROOT / "dist" / "screen_train_v1.zip"
TRAIN_CSV, FRAMES_CSV = "screen_train.csv", "rlvs_frames.csv"
IMAGES_PREFIX = "train_composites/"
#: Top-level names under data/screens/ that must never end up in the zip (TEST set, raw pages).
FORBIDDEN_TOP = {"composite", "scraped_pages", "recorded", "train_pages", "urls.txt",
                 "sources.csv", "composites.csv"}
WEIGHT_EXTS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".h5"}


def check_arcname(name: str) -> None:
    """Raise ValueError unless *name* is allowed in the training zip."""
    parts = Path(name).parts
    if not parts or name.startswith("/") or ".." in parts:
        raise ValueError(f"unsafe archive path: {name!r}")
    if parts[0] in FORBIDDEN_TOP:
        raise ValueError(f"{name!r} belongs to the TEST set / raw pages — never packed")
    if Path(name).suffix.lower() in WEIGHT_EXTS:
        raise ValueError(f"{name!r} looks like model weights — never packed")
    if name not in (TRAIN_CSV, FRAMES_CSV, "manifest.json") and not name.startswith(IMAGES_PREFIX):
        raise ValueError(f"{name!r} is not part of the training set")


def pack(screens_dir: Path = SCREENS_DIR, out_zip: Path = DIST_ZIP) -> dict:
    """Write *out_zip*; returns the manifest (counts, size)."""
    screens_dir, out_zip = Path(screens_dir), Path(out_zip)
    rows = list(csv.DictReader(open(screens_dir / TRAIN_CSV)))
    if not rows:
        raise ValueError(f"{screens_dir / TRAIN_CSV} is empty")
    for name in (TRAIN_CSV, FRAMES_CSV):
        if not (screens_dir / name).exists():
            raise FileNotFoundError(screens_dir / name)

    images = sorted({r["path"] for r in rows})
    missing = [p for p in images if not (screens_dir / p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image(s) in {TRAIN_CSV} not on disk, e.g. {missing[:3]}")
    for name in images:
        check_arcname(name)

    is_tile = {r["path"]: r["is_tile"] in ("1", "True") for r in rows}
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "images": len(images),
        "composites": sum(not is_tile[p] for p in images),
        "tiles": sum(is_tile[p] for p in images),
        "rows_by_split_label": {f"{s}/{l}": n for (s, l), n in
                                sorted(Counter((r["split"], r["label"]) for r in rows).items())},
    }

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_zip.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w") as zf:
        for name in (TRAIN_CSV, FRAMES_CSV):
            zf.write(screens_dir / name, name, compress_type=zipfile.ZIP_DEFLATED)
        for i, name in enumerate(images, 1):
            zf.write(screens_dir / name, name, compress_type=zipfile.ZIP_STORED)  # PNGs: already compressed
            if i % 1000 == 0:
                print(f"  {i}/{len(images)} images", flush=True)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    tmp.replace(out_zip)
    manifest["zip_bytes"] = out_zip.stat().st_size
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--screens", type=Path, default=SCREENS_DIR)
    ap.add_argument("--out", type=Path, default=DIST_ZIP)
    args = ap.parse_args()
    m = pack(args.screens, args.out)
    with zipfile.ZipFile(args.out) as zf:     # re-read: report what is actually in the zip
        names = zf.namelist()
    n_png = sum(n.endswith(".png") for n in names)
    print(f"{args.out}  {m['zip_bytes'] / 1e9:.2f} GB ({m['zip_bytes'] / 2**20:.0f} MiB)")
    print(f"{len(names)} files: {n_png} images ({m['composites']} composites + {m['tiles']} tiles), "
          f"{', '.join(n for n in names if not n.endswith('.png'))}")
    print("rows by split/label:", m["rows_by_split_label"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

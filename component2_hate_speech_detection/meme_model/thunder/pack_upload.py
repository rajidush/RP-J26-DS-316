"""Pack what the Thunder run needs into one tar (run on the laptop).

    python -m component2_hate_speech_detection.meme_model.thunder.pack_upload --kaggle
    python -m component2_hate_speech_detection.meme_model.thunder.pack_upload --kaggle --reuse-views
    python -m component2_hate_speech_detection.meme_model.thunder.pack_upload

With --kaggle (used for the 2026-10-22 runs) the images stay out: the instance downloads
the same Kaggle copy itself, and the tar carries every image's SHA-256 so thunder_go.sh
can prove they match byte for byte. Writes data/upload/meme_v2_small.tar (~60 MB).
Without it, the images go in too: data/upload/meme_v2_upload.tar (~3.3 GB).

Both carry the train, val and test manifests, the shipped v1 head (scored on the new
features too) and v1's stored CLIP features (to check the cloud reproduces them). Paths
are relative to component2_hate_speech_detection/. Each tar gets a .sha256 file for
`sha256sum -c`. Hateful Memes is research-only: nothing here is committed or shared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

from ..embed import COMPONENT_ROOT, DATA_DIR, IMAGE_ROOT, MANIFEST_DIR, read_manifest

UPLOAD_DIR = DATA_DIR / "upload"
OUT = UPLOAD_DIR / "meme_v2_upload.tar"
SMALL = UPLOAD_DIR / "meme_v2_small.tar"
SUMS = UPLOAD_DIR / "image_sha256.txt"
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kaggle", action="store_true",
                        help="leave the images out (the instance downloads them from Kaggle); add their SHA-256s")
    parser.add_argument("--reuse-views", action="store_true",
                        help="add data/views/captions.jsonl from an earlier run: the instance skips OCR and only "
                             "rebuilds the screen crops, so a new encoder sees exactly the same views")
    args = parser.parse_args()

    manifests = [MANIFEST_DIR / f"{split}.jsonl" for split in SPLITS]
    images = [IMAGE_ROOT / r["img"] for m in manifests if m.exists() for r in read_manifest(m)]
    files = manifests + [COMPONENT_ROOT / "models" / "meme" / "meme_clip-b32_align.pt", DATA_DIR / "embeddings" / "clip-b32.npz"]
    if args.reuse_views:
        files.append(DATA_DIR / "views" / "captions.jsonl")
    missing = [str(f) for f in files + images if not f.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} files missing, e.g. {missing[:3]}")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.kaggle:
        out = SMALL
        SUMS.write_text("".join(f"{sha256(f)}  {f.relative_to(COMPONENT_ROOT).as_posix()}\n" for f in images),
                        encoding="utf-8", newline="\n")
        files.append(SUMS)
    else:
        out = OUT
        files += images
    with tarfile.open(out, "w") as tar:
        for k, f in enumerate(files, 1):
            tar.add(f, arcname=f.relative_to(COMPONENT_ROOT).as_posix())
            if k % 2000 == 0:
                print(f"  {k}/{len(files)} files", flush=True)
    digest = sha256(out)
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8", newline="\n")
    info = {"file": out.name, "files": len(files), "images_inside": 0 if args.kaggle else len(images),
            "images_listed": len(images) if args.kaggle else 0, "bytes": out.stat().st_size, "sha256": digest,
            "packed": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.perf_counter() - started, 1)}
    out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"\nUpload {out.name} and {out.name}.sha256 to the instance's home folder (see thunder/README.md).")


if __name__ == "__main__":
    main()

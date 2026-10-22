"""Pack what the Thunder run needs into one tar (run on the laptop).

    python -m component2_hate_speech_detection.meme_model.thunder.pack_upload

Writes data/upload/meme_v2_upload.tar (git-ignored), with paths relative to
component2_hate_speech_detection/, so on the instance:

    tar -xf ~/meme_v2_upload.tar -C RP-J26-DS-316/component2_hate_speech_detection

Contents: the train, val and test meme images (not the 1,000 unlabelled ones), their
manifests, the shipped v1 head (scored on the new features too) and v1's stored CLIP
features (to check the cloud reproduces them). Hateful Memes is research-only: the tar
goes to your private instance only and is never committed or shared.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import time

from ..embed import COMPONENT_ROOT, DATA_DIR, IMAGE_ROOT, MANIFEST_DIR, read_manifest

OUT = DATA_DIR / "upload" / "meme_v2_upload.tar"
SPLITS = ("train", "val", "test")


def main() -> None:
    files = []
    for split in SPLITS:
        files.append(MANIFEST_DIR / f"{split}.jsonl")
        files += [IMAGE_ROOT / r["img"] for r in read_manifest(MANIFEST_DIR / f"{split}.jsonl")]
    files += [COMPONENT_ROOT / "models" / "meme" / "meme_clip-b32_align.pt", DATA_DIR / "embeddings" / "clip-b32.npz"]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} files missing, e.g. {missing[:3]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tarfile.open(OUT, "w") as tar:
        for k, f in enumerate(files, 1):
            tar.add(f, arcname=f.relative_to(COMPONENT_ROOT).as_posix())
            if k % 2000 == 0:
                print(f"  {k}/{len(files)} files", flush=True)
    h = hashlib.sha256()
    with open(OUT, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    info = {"file": OUT.name, "files": len(files), "bytes": OUT.stat().st_size, "sha256": h.hexdigest(),
            "packed": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.perf_counter() - started, 1)}
    OUT.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"\nUpload {OUT} to the instance's home folder. Check it there with: sha256sum ~/meme_v2_upload.tar")


if __name__ == "__main__":
    main()

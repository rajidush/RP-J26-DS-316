"""
rlvs_groups.py — Approximate RLVS "source video" groups, and frames near the TEST set.

RLVS frames here (~/Desktop/new_violence/<class>/<n>.jpg, listed in
results/01_rlvs_finetune/splits.csv) carry NO video id: files are numbered per class in shuffled
order and the frame-level split put frames of one video in train/val/test.
So videos are approximated by grouping near-identical frames:

    edge between two frames of the same class if
        cosine(ViT CLS features) >= GROUP_CLS   (0.97)   or
        cosine(24x24 grayscale thumbnails) >= GROUP_THUMB (0.85)
    video_group = connected component (union-find)

Checked visually: grouped frames are the same video (high precision), but
frames of one video that look different can land in different groups (recall
is limited) — so this is an approximation, documented as such.

To protect the TEST set, a frame is marked ``near_test`` when it is
    * one of the frames used in the test composites (byte-identical), or
    * in the same video_group as one, or
    * loosely similar to one: CLS >= NEAR_CLS (0.93) or thumbnail >= NEAR_THUMB (0.75)
      (any class) — deliberately over-inclusive.
Training composites never use a near_test frame.

Output: data/screens/rlvs_frames.csv
    path, class, rlvs_split, video_group, in_test_composites, near_test,
    max_cls_to_test, max_thumb_to_test

Usage
-----
    python component1_screen_monitoring/src/rlvs_groups.py            # features cached in data/screens/.cache
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .config import COMPONENT_ROOT, SCREENS_DIR, SPLITS_CSV
except ImportError:
    from config import COMPONENT_ROOT, SCREENS_DIR, SPLITS_CSV

RLVS_ROOT = Path.home() / "Desktop" / "new_violence"
TEST_FRAMES_ROOT = COMPONENT_ROOT / "data" / "raw"          # frames used by the TEST composites
OUT_CSV = SCREENS_DIR / "rlvs_frames.csv"
CACHE = SCREENS_DIR / ".cache"

GROUP_CLS, GROUP_THUMB = 0.97, 0.85
NEAR_CLS, NEAR_THUMB = 0.93, 0.75
CLASSES = ("violence", "non_violence")


def frame_key(cls: str, name: str) -> str:
    """The splits.csv path format: 'violence /123.jpg'."""
    return f"{cls} /{name}"


def key_to_path(key: str, root: Path = RLVS_ROOT) -> Path:
    cls, name = key.split(" /")
    return root / cls / name


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def thumb_features(paths: list[Path], size: int = 24) -> np.ndarray:
    t = np.stack([np.asarray(Image.open(p).convert("L").resize((size, size)), dtype=np.float32).ravel()
                  for p in paths])
    t = (t - t.mean(1, keepdims=True)) / (t.std(1, keepdims=True) + 1e-6)
    return _l2(np.nan_to_num(t))


def cls_features(paths: list[Path], batch: int = 64) -> np.ndarray:
    import torch
    try:
        from .detector import ViolenceDetector
    except ImportError:
        from detector import ViolenceDetector
    det = ViolenceDetector()
    out = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        x = det.processor(images=imgs, return_tensors="pt").to(det.device)
        with torch.inference_mode():
            out.append(det.model.vit(**x).last_hidden_state[:, 0].float().cpu().numpy())
        print(f"  CLS features {min(i + batch, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return _l2(np.concatenate(out))


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# Grouping (pure numpy — unit-tested)
# ---------------------------------------------------------------------------

def group_frames(feats: list[np.ndarray], thresholds: list[float], classes: np.ndarray) -> np.ndarray:
    """
    Union-find over frames of the SAME class; an edge exists if ANY feature's
    cosine similarity reaches its threshold. Returns a group id per frame
    (the smallest member index of its component).
    """
    n = len(classes)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        for f, thr in zip(feats, thresholds):
            sim = f[idx] @ f[idx].T
            ii, jj = np.where(np.triu(sim, 1) >= thr)
            for a, b in zip(idx[ii], idx[jj]):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def near_mask(feats: list[np.ndarray], thresholds: list[float], test_idx: np.ndarray):
    """For every frame: max similarity to any test frame per feature, and the near flag."""
    maxes = [(f @ f[test_idx].T).max(1) if len(test_idx) else np.zeros(len(f)) for f in feats]
    near = np.zeros(len(feats[0]), dtype=bool)
    for m, thr in zip(maxes, thresholds):
        near |= m >= thr
    return near, maxes


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def build(rlvs_root: Path = RLVS_ROOT, out_csv: Path = OUT_CSV) -> list[dict]:
    rows = list(csv.DictReader(open(SPLITS_CSV)))
    keys = [r["path"] for r in rows]
    paths = [key_to_path(k, rlvs_root) for k in keys]
    classes = np.array([k.split(" /")[0] for k in keys])

    CACHE.mkdir(parents=True, exist_ok=True)
    cls_cache, thumb_cache = CACHE / "rlvs_cls.npy", CACHE / "rlvs_thumb.npy"
    E = np.load(cls_cache) if cls_cache.exists() else cls_features(paths)
    T = np.load(thumb_cache) if thumb_cache.exists() else thumb_features(paths)
    np.save(cls_cache, E)
    np.save(thumb_cache, T)
    assert len(E) == len(T) == len(keys), "feature cache does not match splits.csv — delete .cache"

    groups = group_frames([E, T], [GROUP_CLS, GROUP_THUMB], classes)

    # frames used by the TEST composites: identify by content, not just filename
    key_index = {k: i for i, k in enumerate(keys)}
    test_idx = []
    for cls in CLASSES:
        for p in sorted((TEST_FRAMES_ROOT / cls).iterdir()):
            i = key_index.get(frame_key(cls, p.name))
            if i is None or _md5(p) != _md5(paths[i]):
                raise ValueError(f"test frame {p} not found byte-identical in {rlvs_root}")
            test_idx.append(i)
    test_idx = np.array(sorted(test_idx))
    near, (max_cls, max_thumb) = near_mask([E, T], [NEAR_CLS, NEAR_THUMB], test_idx)
    test_groups = set(groups[test_idx])
    in_test = np.zeros(len(keys), dtype=bool)
    in_test[test_idx] = True
    near |= np.isin(groups, list(test_groups))

    out = [{"path": k, "class": classes[i], "rlvs_split": rows[i]["split"],
            "video_group": f"{classes[i][0]}{groups[i]}", "in_test_composites": int(in_test[i]),
            "near_test": int(near[i]), "max_cls_to_test": round(float(max_cls[i]), 4),
            "max_thumb_to_test": round(float(max_thumb[i]), 4)} for i, k in enumerate(keys)]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[1]).parse_args()
    out = build()
    n_groups = len({r["video_group"] for r in out})
    usable = [r for r in out if r["rlvs_split"] == "train" and not r["near_test"]]
    print(f"{len(out)} frames, {n_groups} approximate video groups → {OUT_CSV}")
    print(f"frames in test composites: {sum(r['in_test_composites'] for r in out)} | "
          f"near_test (excluded from training): {sum(r['near_test'] for r in out)}")
    for c in CLASSES:
        print(f"usable for TRAIN composites ({c}): {sum(r['class'] == c for r in usable)} frames, "
              f"{len({r['video_group'] for r in usable if r['class'] == c})} groups")


if __name__ == "__main__":
    sys.exit(main())

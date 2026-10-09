"""
Download the two inputs for build_labels.py. Nothing lands in git:
  WOAH 2021 fine-grained annotations (~2.6 MB, Apache-2.0)
      -> component2_hate_speech_detection/data/woah/
  Kaggle parthplc/facebook-hateful-meme-dataset (~3.6 GB, research use only)
      -> kagglehub cache (path printed at the end)

Kaggle needs your own API token in ~/.kaggle/kaggle.json
(kaggle.com -> Settings -> API -> Create New Token).

    pip install kagglehub
    python -m component2_hate_speech_detection.annotation.fetch_data
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
WOAH_DIR = COMPONENT_ROOT / "data" / "woah"
WOAH_BASE = "https://raw.githubusercontent.com/facebookresearch/fine_grained_hateful_memes/main/data/annotations/"
WOAH_FILES = ("train.json", "dev_seen.json", "dev_unseen.json")
KAGGLE_DATASET = "parthplc/facebook-hateful-meme-dataset"


def fetch_woah(dest: Path = WOAH_DIR) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name in WOAH_FILES:
        target = dest / name
        if not target.exists():
            urllib.request.urlretrieve(WOAH_BASE + name, target)
        print(f"woah  {name:16s} {target.stat().st_size:>10,d} bytes")
    return dest


def fetch_memes() -> Path:
    import kagglehub  # optional dependency, only needed for this step

    root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    # The archive nests everything under data/; fall back to the root if not.
    data = root / "data" if (root / "data" / "train.jsonl").exists() else root
    print(f"memes {data}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-memes", action="store_true", help="only fetch the WOAH annotations")
    args = parser.parse_args()

    woah = fetch_woah()
    if args.skip_memes:
        return
    memes = fetch_memes()
    print("\nnext:\n  python -m component2_hate_speech_detection.annotation.build_labels "
          f'--memes-dir "{memes}" --woah-dir "{woah}"')


if __name__ == "__main__":
    main()

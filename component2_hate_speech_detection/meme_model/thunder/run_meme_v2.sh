#!/usr/bin/env bash
# Meme head v2 on a Thunder Compute GPU instance: views -> CLIP embeddings -> train v1 recipe + v2 -> evidence.
#
# thunder_go.sh sets up the images, code and upload and then runs this (see thunder/README.md). By hand:
#   git clone -b c2/finetune-pp1 https://github.com/rajidush/RP-J26-DS-316.git
#   cd RP-J26-DS-316
#   tar -xf ~/meme_v2_upload.tar -C component2_hate_speech_detection
#   bash component2_hate_speech_detection/meme_model/thunder/run_meme_v2.sh
#
# Everything it prints is also saved under evaluation/results/meme_v2/ as evidence.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
C2=component2_hate_speech_detection
EV="$C2/evaluation/results/meme_v2"
mkdir -p "$EV"
export PYTHONUNBUFFERED=1

echo "== 1/6 Python 3.14 environment (matches the laptop)"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --quiet --user uv 2>/dev/null \
    || python3 -m pip install --quiet --user --break-system-packages uv 2>/dev/null \
    || curl -LsSf https://astral.sh/uv/install.sh | sh
fi
[ -d .venv ] || uv venv --quiet --python 3.14 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --quiet pip torch "transformers>=5" numpy pillow scikit-learn rapidocr onnxruntime tokenizers jsonschema matplotlib
# rapidocr pulls in opencv-python, which needs libGL; a bare GPU image may not have it.
python -c "import cv2" 2>/dev/null || { uv pip uninstall --quiet opencv-python; uv pip install --quiet --reinstall opencv-python-headless; }
python -c "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')"

echo "== 2/6 Check the upload"
for f in data/labels/finetune/train.jsonl data/labels/finetune/val.jsonl data/labels/finetune/test.jsonl models/meme/meme_clip-b32_align.pt; do
  test -f "$C2/$f" || { echo "Missing $C2/$f: unpack meme_v2_upload.tar into $C2 first"; exit 1; }
done
echo "images: $(ls "$C2/data/hateful_memes/img" | wc -l)"

echo "== 3/6 Evidence: code version, machine, packages, data fingerprints"
python -m $C2.meme_model.evidence env --out "$EV"

echo "== 4/6 Views: OCR + screenshots on all $(nproc) cores, then CLIP embeddings"
python -m $C2.meme_model.views --workers "${WORKERS:-$(nproc)}" 2>&1 | tee "$EV/views.log"

echo "== 5/6 Train the v1 recipe and v2, 5 seeds each; compare on every view"
python -m $C2.meme_model.train_views 2>&1 | tee "$EV/train.log"

echo "== 6/6 Training card + one archive to download"
python -m $C2.meme_model.evidence card --out "$EV"
tar -czf "$HOME/meme_v2_results.tgz" "$EV" "$C2/models/meme/meme_clip-b32_align_v2views.pt" "$C2/data/predictions/meme_v2_views.csv" \
  "$C2/data/views/captions.jsonl" "$C2/data/views/views_stats.json" "$C2/data/embeddings/clip-b32-views.npz"
echo "Done. Download ~/meme_v2_results.tgz ($(du -h "$HOME/meme_v2_results.tgz" | cut -f1))"

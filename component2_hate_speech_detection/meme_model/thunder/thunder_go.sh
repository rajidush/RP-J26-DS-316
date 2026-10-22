#!/usr/bin/env bash
# One command on the Thunder instance: Kaggle images -> repo -> small tar -> image check -> run_meme_v2.sh.
#
# Upload meme_v2_small.tar, meme_v2_small.tar.sha256 (from pack_upload --kaggle) and this script to ~,
# put your Kaggle API token in ~/.kaggle/access_token (Kaggle -> Settings -> API shows the line), then:
#   bash ~/thunder_go.sh
# The token is deleted as soon as the images are down. Every step can be re-run.
set -euo pipefail
BRANCH="${BRANCH:-c2/finetune-pp1}"
cd ~
echo "== Check the upload"
sha256sum -c meme_v2_small.tar.sha256

echo "== Hateful Memes images from Kaggle (the same copy as the laptop)"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
if [ -z "$(ls -d ~/hm/data/img ~/hm/img 2>/dev/null)" ]; then
  test -f ~/.kaggle/access_token || { echo "No Kaggle token: put it in ~/.kaggle/access_token first, then rerun."; exit 1; }
  uvx kaggle datasets download -d parthplc/facebook-hateful-meme-dataset -p ~/hm --unzip
fi
rm -f ~/.kaggle/access_token

echo "== Code (branch $BRANCH) and the small upload"
[ -d RP-J26-DS-316 ] || git clone -b "$BRANCH" https://github.com/rajidush/RP-J26-DS-316.git
cd RP-J26-DS-316
C2=component2_hate_speech_detection
tar -xf ~/meme_v2_small.tar -C "$C2"
HM=~/hm/data; [ -d "$HM/img" ] || HM=~/hm
[ -e "$C2/data/hateful_memes" ] || ln -s "$HM" "$C2/data/hateful_memes"

echo "== Every image must match the laptop's copy byte for byte"
(cd "$C2" && sha256sum -c --quiet data/upload/image_sha256.txt)
echo "ALL $(wc -l < "$C2/data/upload/image_sha256.txt") IMAGES MATCH"

bash "$C2/meme_model/thunder/run_meme_v2.sh"

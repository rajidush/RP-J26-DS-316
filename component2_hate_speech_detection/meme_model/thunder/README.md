# Meme head v2 on Thunder Compute

Retrains C2's meme head on what C2 actually reads: the dataset caption, C2's own OCR
caption, and the meme cut out of a built screenshot. It compares v2 with the v1 recipe on
every view and records the evidence. Protocol and success rule: `meme_model/train_views.py`.

**Result of the 2026-10-22 run:** see [`evaluation/results/meme_v2/TRAINING_CARD.md`](../../evaluation/results/meme_v2/TRAINING_CARD.md).

| Step | Where | Time (measured 2026-10-22) |
|---|---|---|
| Pack the small upload | laptop | ~1 min |
| Upload ~60 MB | laptop → instance | ~2 min on a phone hotspot |
| Kaggle download of the images | instance | ~1 min |
| Views: OCR + screenshots | 12 vCPUs | 15 min |
| CLIP embeddings | L40 GPU | 5 min |
| Train v1 recipe + v2, 5 seeds each | L40 GPU | 4 min |
| Download results (~100 MB) | instance → laptop | ~2 min on a phone hotspot |

Cost: about $1 (L40 1×, 12 vCPU / 96 GB, 100 GB disk, $1.03/hr).

## 1. On the laptop: pack the upload

```bash
python -m component2_hate_speech_detection.meme_model.thunder.pack_upload --kaggle
```

This writes `component2_hate_speech_detection/data/upload/meme_v2_small.tar` and `meme_v2_small.tar.sha256`.
The images stay out: the instance downloads the same Kaggle copy (`parthplc/facebook-hateful-meme-dataset`),
and the tar carries every image's SHA-256 so the instance proves they match.
Without `--kaggle`, the images go in too (`meme_v2_upload.tar`, ~3.3 GB); then skip the Kaggle part and
unpack it into `component2_hate_speech_detection` before running `run_meme_v2.sh`.

## 2. Install the Thunder CLI and create the instance

- The `tnr` package on PyPI is retired. Install CLI v2 from
  [github.com/Thunder-Compute/thunder-cli/releases](https://github.com/Thunder-Compute/thunder-cli/releases)
  (on Windows: `tnr_<version>_windows_amd64.zip`, check it against `checksums.txt`, put `tnr.exe` on PATH).
- `tnr login` opens the browser.
- In the console: any GPU, **8 or more vCPUs** (the OCR runs on CPU, one process per core), ~100 GB disk.
  `tnr status` should then list the instance as running.

## 3. Upload, connect, run

```bash
tnr scp component2_hate_speech_detection/data/upload/meme_v2_small.tar component2_hate_speech_detection/data/upload/meme_v2_small.tar.sha256 component2_hate_speech_detection/meme_model/thunder/thunder_go.sh 0:/home/ubuntu/
```

```bash
tnr connect 0
```

On the instance:

```bash
sudo apt-get update -qq && sudo apt-get install -y -qq tmux
tmux new -s meme
```

Paste the `~/.kaggle/access_token` line that Kaggle shows under Settings → API, then:

```bash
bash ~/thunder_go.sh
```

It checks the upload, downloads the images, deletes the token, clones the branch, checks every image's
SHA-256 and runs `run_meme_v2.sh`. If the connection drops: `tnr connect 0`, then `tmux attach -t meme`.
A stopped run resumes the OCR where it left off.

### The same protocol on CLIP ViT-L/14

L/14 (the encoder behind Hate-CLIPper's published 0.82) runs as an upper bound: too heavy for
the laptop, but it shows what the laptop budget costs. Pack with the captions from the B/32 run,
so the instance skips OCR and L/14 sees exactly the same views:

```bash
python -m component2_hate_speech_detection.meme_model.thunder.pack_upload --kaggle --reuse-views
```

Upload as in step 3, then on the instance run `ENCODER=clip-l14 bash ~/thunder_go.sh`. The model
(~1.7 GB) downloads on the instance. Results go to `evaluation/results/meme_v2_clip_l14/` and the
archive is `~/meme_v2_clip_l14_results.tgz`.

## 4. Download the results and delete the instance

```bash
tnr scp 0:/home/ubuntu/meme_v2_results.tgz component2_hate_speech_detection/data/upload/
```

Delete the instance in the console (or `tnr delete 0`) so it stops billing. On the laptop, unpack at the repo root:

```bash
tar -xzf component2_hate_speech_detection/data/upload/meme_v2_results.tgz
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Connection timed out during banner exchange` | the network blocks SSH on non-standard ports (Thunder uses a high port such as 31922; port 22 still works) | use another network, e.g. a phone hotspot; once the run is going inside tmux you can switch back |
| `WARNING: UNPROTECTED PRIVATE KEY FILE` (Windows) | the key in `%USERPROFILE%\.thunder\keys\` inherited an `OWNER RIGHTS` entry | `icacls <key> /inheritance:r /grant:r "$env:USERDOMAIN\$($env:USERNAME):(R)"` |
| `tmux: command not found` | the Base template has no tmux | the `apt-get` line in step 3 |
| `No Kaggle token` | the token file is missing | paste Kaggle's `access_token` line, rerun `thunder_go.sh` |

## What comes back

| File | What it is | Committed? |
|---|---|---|
| `evaluation/results/meme_v2/TRAINING_CARD.md` | one-page summary: model, data, settings, techniques, machine, results, success rule | yes |
| `evaluation/results/meme_v2/meme_v2_views.json` | every number, per seed, with epoch-by-epoch histories | yes |
| `evaluation/results/meme_v2/env.json`, `pip_freeze.txt`, `nvidia_smi.txt` | code commit, GPU, packages, data SHA-256s | yes |
| `evaluation/results/meme_v2/views.log`, `train.log`, `views_stats.json` | full console output, OCR and picture-finder statistics | yes |
| `models/meme/meme_clip-b32_align_v2views.pt` | the trained v2 head | no (git-ignored) |
| `data/views/`, `data/embeddings/clip-b32-views.npz`, `data/predictions/meme_v2_views.csv` | captions, features, predictions | no (git-ignored) |

Hateful Memes is licensed for research only. The images stay on your private instance and are never committed.

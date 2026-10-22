# Meme head v2 on Thunder Compute

Retrains C2's meme head on what C2 actually reads: the dataset caption, C2's own OCR
caption, and the meme cut out of a built screenshot. It compares v2 with the v1 recipe on
every view and records the evidence. Protocol and success rule: `meme_model/train_views.py`.

| Step | Where | Time (estimate) |
|---|---|---|
| Pack the upload | laptop | ~5 min |
| Upload the tar | laptop → instance | depends on your upload speed (~3 GB) |
| Views: OCR + screenshots | instance CPUs | ~15–30 min with 8–16 vCPUs |
| CLIP embeddings | instance GPU | ~5 min |
| Train v1 recipe + v2, 5 seeds each | instance GPU | ~10 min |
| Download results | instance → laptop | seconds (~150 MB) |

## 1. On the laptop: pack the upload

```bash
python -m component2_hate_speech_detection.meme_model.thunder.pack_upload
```

This writes `component2_hate_speech_detection/data/upload/meme_v2_upload.tar` and prints its SHA-256.

## 2. Create the instance

In the Thunder console, or with the CLI (`pip install tnr`, then `tnr login`):

- any GPU (T4, A10 or A100);
- 8 or more vCPUs, because the OCR runs on CPU, one process per core;
- about 100 GB disk.

## 3. Upload and connect

```bash
tnr scp component2_hate_speech_detection/data/upload/meme_v2_upload.tar 0:~/
```

```bash
tnr connect 0
```

`0` is the instance ID from `tnr status`. If your CLI version uses different syntax, check `tnr --help`.

## 4. On the instance: run everything

```bash
sha256sum ~/meme_v2_upload.tar
git clone -b c2/finetune-pp1 https://github.com/rajidush/RP-J26-DS-316.git
cd RP-J26-DS-316
tar -xf ~/meme_v2_upload.tar -C component2_hate_speech_detection
bash component2_hate_speech_detection/meme_model/thunder/run_meme_v2.sh
```

The SHA-256 from `sha256sum` must match the one printed on the laptop.

If the connection may drop, run the last line inside `tmux` (`tmux new -s meme`; reattach with `tmux attach -t meme`).
A stopped run resumes the OCR where it left off.

## 5. Download the results and stop the instance

```bash
tnr scp 0:~/meme_v2_results.tgz .
```

Stop the instance in the console (or with `tnr stop 0`) so it stops billing.

On the laptop, unpack the archive at the repo root:

```bash
tar -xzf meme_v2_results.tgz
```

## What comes back

| File | What it is | Committed? |
|---|---|---|
| `evaluation/results/meme_v2/TRAINING_CARD.md` | one-page summary: model, data, settings, machine, results, success rule | yes |
| `evaluation/results/meme_v2/meme_v2_views.json` | every number, per seed, with epoch-by-epoch histories | yes |
| `evaluation/results/meme_v2/env.json`, `pip_freeze.txt`, `nvidia_smi.txt` | code commit, GPU, packages, data SHA-256s | yes |
| `evaluation/results/meme_v2/views.log`, `train.log` | full console output of the run | yes |
| `models/meme/meme_clip-b32_align_v2views.pt` | the trained v2 head | no (git-ignored) |
| `data/views/`, `data/embeddings/clip-b32-views.npz`, `data/predictions/meme_v2_views.csv` | captions, features, predictions | no (git-ignored) |

Hateful Memes is licensed for research only. The images stay on your private instance and are never committed.

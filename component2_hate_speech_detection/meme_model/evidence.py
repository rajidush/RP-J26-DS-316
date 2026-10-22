"""Training evidence for a cloud run: what ran, on what, with which data, and what came out.

    python -m component2_hate_speech_detection.meme_model.evidence env  --out <dir>
    python -m component2_hate_speech_detection.meme_model.evidence card --out <dir>

`env` records the code version (git commit), the machine (GPU, CUDA, Python, packages),
and fingerprints of the data (SHA-256 of the split manifests and of the shipped v1 head),
so anyone can check the run used exactly these inputs. `card` turns that plus the run's
results into TRAINING_CARD.md. Everything here is safe to commit: no data, no weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from .embed import COMPONENT_ROOT, IMAGE_ROOT, MANIFEST_DIR

# Fine-tuning techniques this run uses: (technique, what it does here, where in the code, source).
TECHNIQUES = (
    ("Transfer learning with a frozen backbone", "CLIP {model} turns each image and caption into {dim}-number features once; only the small head on top is trained",
     "`meme_model/views.py` `embed_views()`", "Radford et al., ICML 2021 (CLIP)"),
    ("Hate-CLIPper align fusion", "image and text features are projected and multiplied element-wise, so the head sees how picture and caption interact",
     "`meme_model/heads.py` `FusionHead.fuse()`", "Kumar & Nandakumar, NLP4PI @ EMNLP 2022"),
    ("Deployment-matched training views (data augmentation)", "each training meme is also shown with C2's own OCR caption and cut out of a built screenshot, the inputs C2 really gets",
     "`meme_model/views.py` `make_views()`", "Kirk et al., WOAH 2021 (OCR text hurts meme models in the wild)"),
    ("AdamW with weight decay 1e-4", "optimiser with decoupled weight decay, to limit overfitting",
     "`meme_model/train.py` `train_one()`", "Loshchilov & Hutter, ICLR 2019"),
    ("Dropout 0.4", "randomly drops units while training so the head can't memorise",
     "`meme_model/heads.py` `FusionHead`", "Srivastava et al., JMLR 2014"),
    ("Gradient clipping 0.1", "caps each update's size, keeping training stable",
     "`meme_model/train.py` `train_one()`", "Pascanu et al., ICML 2013"),
    ("Early stopping on validation AUROC (patience 5)", "keeps the epoch that scored best on validation, never on test",
     "`meme_model/train.py` `train_one()`", "Prechelt 1998 (foundation)"),
    ("Temperature scaling", "rescales the scores on validation so 0.7 means about 70 %",
     "`meme_model/train.py` `fit_temperature()`", "Guo et al., ICML 2017"),
    ("Threshold chosen on validation", "the decision cut-off is picked on validation memes only",
     "`meme_model/train.py` `best_threshold()`", "standard practice"),
    ("Five-seed ensemble", "five heads with different seeds; their scores are averaged, and mean ± sd shows the spread",
     "`meme_model/train_views.py` `run_recipe()`", "Lakshminarayanan et al., NeurIPS 2017"),
    ("Controlled baseline", "the v1 recipe is retrained with identical settings and seeds, so only the training data differs",
     "`meme_model/train_views.py` `RECIPES`", "ablation design"),
    ("Bootstrap 95 % confidence interval", "1,000 resamples of the test memes show how sure each AUROC is",
     "`meme_model/train.py` `bootstrap_auroc()`", "Efron & Tibshirani 1993 (foundation)"),
)

PACKAGES = ("torch", "transformers", "numpy", "scikit-learn", "pillow", "rapidocr", "onnxruntime", "tokenizers", "opencv-python-headless", "opencv-python")
MODEL_NAMES = {"clip-b32": "ViT-B/32", "clip-b16": "ViT-B/16", "clip-l14": "ViT-L/14", "clip-l14-336": "ViT-L/14 (336 px)"}
VIEW_LABELS = (("clean", "Clean caption (as in the dataset)"), ("ocr", "C2's OCR caption"), ("screen", "Cut from a screenshot"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=COMPONENT_ROOT).stdout.strip()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def env(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    gpu = {}
    try:
        import torch

        gpu = {"cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
               "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    except Exception as exc:
        gpu = {"error": str(exc)}
    info = {"recorded_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "python": sys.version.split()[0],
            "platform": platform.platform(), "cpu_count": __import__("os").cpu_count(), "gpu": gpu, "packages": versions,
            "git": {"commit": run(["git", "rev-parse", "HEAD"]), "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                    "remote": run(["git", "remote", "get-url", "origin"]), "uncommitted": run(["git", "status", "--short"]).splitlines()}}
    data = {"manifests": {}, "images": None, "shipped_v1_head": None}
    for split in ("train", "val", "test"):
        p = MANIFEST_DIR / f"{split}.jsonl"
        if p.exists():
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            data["manifests"][split] = {"rows": len(rows), "hateful": sum(int(r["label"]) for r in rows), "sha256": sha256(p)}
    img = IMAGE_ROOT / "img"
    if img.exists():
        data["images"] = sum(1 for _ in img.glob("*.png"))
    v1 = COMPONENT_ROOT / "models" / "meme" / "meme_clip-b32_align.pt"
    if v1.exists():
        data["shipped_v1_head"] = {"file": v1.name, "sha256": sha256(v1)}
    info["data"] = data
    (out / "env.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (out / "pip_freeze.txt").write_text(run([sys.executable, "-m", "pip", "freeze"]) + "\n", encoding="utf-8")
    (out / "nvidia_smi.txt").write_text(run(["nvidia-smi"]) + "\n", encoding="utf-8")
    print(f"wrote {out / 'env.json'}, pip_freeze.txt, nvidia_smi.txt")


def card(out: Path, encoder: str = "clip-b32") -> None:
    from .views import stats_file

    e = json.loads((out / "env.json").read_text(encoding="utf-8"))
    r = json.loads((out / "meme_v2_views.json").read_text(encoding="utf-8"))
    vs_path = stats_file(encoder)
    vs = json.loads(vs_path.read_text(encoding="utf-8")) if vs_path.exists() else {}
    if vs:
        (out / "views_stats.json").write_text(json.dumps(vs, indent=2) + "\n", encoding="utf-8")
    model, dim = MODEL_NAMES.get(encoder, encoder), r.get("feature_dim", 512)
    rec = r["recipes"]
    shipped = r.get("shipped_v1_head") or {}
    f = lambda x: f"{x['ensemble']['auroc']:.3f} ({x['ensemble']['auroc_ci95'][0]:.3f}–{x['ensemble']['auroc_ci95'][1]:.3f}); seeds {x['auroc_mean_sd'][0]:.3f} ± {x['auroc_mean_sd'][1]:.3f}"
    rows = []
    for v, label in VIEW_LABELS:
        s = shipped.get("views", {}).get(v, {}).get("auroc", "–")
        a, b = rec["v1_recipe"]["test"][v], rec["v2_views"]["test"][v]
        rows.append(f"| {label} | {s if isinstance(s, str) else f'{s:.3f}'} | {f(a)} | {f(b)} |")
    compare = []  # the same protocol with B/32, when this run used another encoder
    b32_path = COMPONENT_ROOT / "evaluation" / "results" / "meme_v2" / "meme_v2_views.json"
    if encoder != "clip-b32" and b32_path.exists():
        b = json.loads(b32_path.read_text(encoding="utf-8"))["recipes"]
        auc = lambda rep, v: rep[v]["ensemble"]["auroc"]
        compare = [f"### Against CLIP ViT-B/32 (same views, settings and seeds; `evaluation/results/meme_v2/`)", "",
                   f"| Test view | B/32 v1 recipe | {model} v1 recipe | B/32 v2 | {model} v2 |", "|---|---|---|---|---|",
                   *[f"| {label} | {auc(b['v1_recipe']['test'], v):.3f} | {auc(rec['v1_recipe']['test'], v):.3f} | "
                     f"{auc(b['v2_views']['test'], v):.3f} | {auc(rec['v2_views']['test'], v):.3f} |" for v, label in VIEW_LABELS], ""]
    rule = r["success_rule"]
    ev = vs.get("embedding", {}).get("check_vs_v1_features", {})
    rebuilt = vs.get("views", {}).get("crops_rebuilt_on_same_page")
    g = e["gpu"]
    lines = [
        f"# Training card: meme head v2 (trained on what C2 actually reads){'' if encoder == 'clip-b32' else f', CLIP {model}'}", "",
        f"Run {r['date']} · branch `{e['git']['branch']}` · commit `{e['git']['commit'][:10]}` · {r['hardware']} · {r['seconds'] / 60:.0f} min of training",
        "",
        "## What was trained",
        f"- **Model:** frozen CLIP {model} (`{r['encoder']}`) + Hate-CLIPper align-fusion head, {r['trainable_parameters']:,} trainable parameters.",
        "- **v1 recipe:** clean captions only, as the shipped head was trained. **v2:** clean + C2's OCR caption + the meme cut out of a built screenshot.",
        f"- **Data:** Hateful Memes, {r['memes_per_split']['train']:,} train / {r['memes_per_split']['val']} val / {r['memes_per_split']['test']} test (dev_seen) memes; "
        f"{r['train_rows']['v2_views']:,} training rows for v2.",
        f"- **Settings (v1's, unchanged):** `{json.dumps(r['settings'])}`; seeds {r['seeds']}.",
        "- **Protocol, fixed before results:** validation chooses everything; test reported once per view; a meme the finder misses scores 0 on the screen view.",
        "",
        "## Fine-tuning techniques used",
        "| Technique | What it does here | Code (at the commit above) | Source |", "|---|---|---|---|",
        *[f"| {t} | {w.format(model=model, dim=dim)} | {c} | {s} |" for t, w, c, s in TECHNIQUES],
        "",
        f"## Results (test AUROC, {r['memes_per_split']['test']} memes; ensemble of {len(r['seeds'])} seeds with 95 % CI)", "",
        "| Test view | Shipped v1 head (B/32) | v1 recipe, retrained | v2 (three views) |", "|---|---|---|---|", *rows, "",
        f"**Success rule** (v2 beats v1 on OCR and screen, loses ≤ 0.01 on clean): OCR {rule['ocr_gain']:+.3f}, "
        f"screen {rule['screen_gain']:+.3f}, clean {rule['clean_change']:+.3f} → **{'PASS' if rule['pass'] else 'FAIL'}**.",
        "",
        *compare,
        "## Where and how it ran",
        f"- **Machine:** {', '.join(g.get('devices') or ['CPU'])} (CUDA {g.get('torch_cuda')}), {e['cpu_count']} vCPUs, {e['platform']}, Python {e['python']}.",
        f"- **Packages:** " + ", ".join(f"{k} {v}" for k, v in e["packages"].items()) + " (full list: `pip_freeze.txt`).",
        f"- **Code:** {e['git']['remote']} at commit `{e['git']['commit']}`.",
        f"- **Commands:** {'' if encoder == 'clip-b32' else f'`ENCODER={encoder}` '}`meme_model/thunder/run_meme_v2.sh`, "
        "which runs `meme_model.views` then `meme_model.train_views`.",
        "",
        "## Data fingerprints",
        *[f"- `{k}.jsonl`: {v['rows']} memes ({v['hateful']} hateful), SHA-256 `{v['sha256'][:16]}…`" for k, v in e["data"]["manifests"].items()],
        f"- Images on the instance: {e['data']['images']}. Shipped v1 head SHA-256 `{(e['data']['shipped_v1_head'] or {}).get('sha256', '–')[:16]}…`",
        (f"- Clean-view features vs the stored {model} features: image cosine {ev.get('image_mean_cosine')}, text cosine "
         f"{ev.get('text_mean_cosine')} (1.0 = identical inputs)." if ev else f"- No stored {model} features to compare the clean view with."),
        *([f"- OCR and screen captions reused from the B/32 run; {rebuilt * 100:.1f} % of the rebuilt screen crops came from "
           "the same page (layout and box) as that run."] if rebuilt is not None else []),
        *([f"- Picture finder found the meme on {vs['view_stats']['test']['finder_found'] * 100:.1f} % of test pages; "
           f"OCR read no text on {vs['view_stats']['test']['ocr_caption_empty'] * 100:.1f} % of test memes."] if vs.get("view_stats", {}).get("test") else []),
        "",
        "## Evidence files in this folder",
        "- `env.json`, `pip_freeze.txt`, `nvidia_smi.txt`: machine and package versions.",
        "- `views.log`, `train.log`: full console output, including every seed's epoch-by-epoch validation AUROC.",
        "- `meme_v2_views.json`: every number above, per seed, with training histories.",
        "- `views_stats.json`: OCR and picture-finder statistics per split.",
        "",
        "Not committed (git-ignored): the images, captions, embeddings, predictions and the trained weights "
        f"(`models/meme/meme_{encoder}_align_v2views.pt`). Hateful Memes is licensed for research only.",
    ]
    (out / "TRAINING_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'TRAINING_CARD.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("env", "card"))
    parser.add_argument("--out", type=Path, default=COMPONENT_ROOT / "evaluation" / "results" / "meme_v2")
    parser.add_argument("--encoder", default="clip-b32", choices=sorted(MODEL_NAMES))
    a = parser.parse_args()
    env(a.out) if a.what == "env" else card(a.out, a.encoder)


if __name__ == "__main__":
    main()

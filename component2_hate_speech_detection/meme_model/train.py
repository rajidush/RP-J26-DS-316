"""Train, calibrate and evaluate a meme fusion head on frozen CLIP embeddings.

    python -m component2_hate_speech_detection.meme_model.train --encoder clip-b32 --fusion align
    python -m component2_hate_speech_detection.meme_model.train --fusion align --rgcl 1 --knn 10 --tag rgcl
    python -m component2_hate_speech_detection.meme_model.train --logreg        # MVP-style baselines

Protocol (CODEBOOK / DATASHEET splits; test is never used for any choice):
    train  8,000   gradient steps
    val      500   early stopping, temperature, threshold, every hyper-parameter
    test     500   dev_seen, reported once per configuration

Optimiser settings follow Hate-CLIPper and RGCL (AdamW, lr 1e-4, weight decay
1e-4, gradient clip 0.1, batch 64). Five seeds; the report gives mean +/- sd
over seeds and a bootstrap 95 % CI for the seed ensemble.

Note on comparisons: published HMC figures are mostly on test_seen, whose
labels were never released. The comparable published number for our test
split (dev_seen) is Hate-CLIPper's 81.2-82.6 AUROC with CLIP ViT-L/14.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn

from .embed import COMPONENT_ROOT, DATA_DIR, MANIFEST_DIR, load_embeddings, read_manifest
from .heads import FUSIONS, FusionHead, count_parameters

RESULTS_DIR = COMPONENT_ROOT / "evaluation" / "results"
MODEL_DIR = COMPONENT_ROOT / "models" / "meme"           # git-ignored
PREDICTION_DIR = DATA_DIR / "predictions"                # git-ignored
SEEDS = (316, 317, 318, 319, 320)
BOOTSTRAP_ROUNDS = 1000

# Published reference points (AUROC x100), for context in the report only.
REFERENCES = {
    "kiela2020_human_test_seen": 82.65,
    "hateclipper2022_align_L14_dev_seen": 81.55,
    "hateclipper2022_cross_L14_dev_seen": 82.62,
    "hateclipper2022_align_L14_test_seen": 85.8,
    "rgcl2024_L14_test_seen": 87.0,
    "mvp_logreg_B32_1200rows": 65.79,
}


# ---------------------------------------------------------------- data

def split_arrays(rows: list[dict], embeddings: dict) -> dict:
    ids = [r["id"] for r in rows]
    return {
        "ids": np.array(ids, dtype=np.int64),
        "image": np.stack([embeddings[i][0] for i in ids]).astype(np.float32),
        "text": np.stack([embeddings[i][1] for i in ids]).astype(np.float32),
        "y": np.array([int(r["label"]) for r in rows], dtype=np.float32),
        "conflict": np.array([bool(r.get("label_conflict")) for r in rows]),
    }


def load_splits(encoder: str, exclude_conflicts: bool = False, manifest_dir: Path = MANIFEST_DIR) -> dict:
    embeddings = load_embeddings(encoder)
    splits = {}
    for name in ("train", "val", "test"):
        rows = read_manifest(manifest_dir / f"{name}.jsonl")
        if exclude_conflicts and name != "test":
            rows = [r for r in rows if not r.get("label_conflict")]
        splits[name] = split_arrays(rows, embeddings)
    return splits


# ---------------------------------------------------------------- metrics

def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15,
                               weights: Optional[np.ndarray] = None) -> float:
    """ECE with equal-width bins (Guo et al., 2017). `weights` reweights the
    sample, e.g. to another base rate (see calibrate.class_weights)."""
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            wb = w[mask]
            ece += wb.sum() / w.sum() * abs(np.average(p[mask], weights=wb) - np.average(y[mask], weights=wb))
    return float(ece)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling on validation logits (Guo et al., 2017). Returns T > 0."""
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(z / log_t.exp(), t)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_t.detach().exp())


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Accuracy-maximising threshold, chosen on validation only."""
    candidates = np.unique(np.round(p, 4))
    scores = [accuracy_score(y, p >= c) for c in candidates]
    return float(candidates[int(np.argmax(scores))])


def summarise(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    return {
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "accuracy@0.5": round(float(accuracy_score(y, p >= 0.5)), 4),
        "accuracy@val_threshold": round(float(accuracy_score(y, p >= threshold)), 4),
        "f1_hateful@0.5": round(float(f1_score(y, p >= 0.5)), 4),
        "ece": round(expected_calibration_error(y, p), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
    }


def bootstrap_auroc(y: np.ndarray, p: np.ndarray, rounds: int = BOOTSTRAP_ROUNDS, seed: int = 316) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            values.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- training

def resolve_device(name: str = "auto") -> torch.device:
    """"auto" = the GPU when there is one. Training runs on a cloud GPU; the laptop only runs the model."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def tensors(split: dict, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.from_numpy(split[k]).to(device) for k in ("image", "text", "y"))


@torch.no_grad()
def predict_logits(model: FusionHead, split: dict, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    image, text, _ = tensors(split, model_device(model))
    out = [model(image[i:i + batch_size], text[i:i + batch_size]) for i in range(0, len(image), batch_size)]
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def embed_split(model: FusionHead, image: torch.Tensor, text: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    """Unit-normalised last-hidden-layer embeddings (dropout off)."""
    model.eval()
    image, text = image.to(model_device(model)), text.to(model_device(model))
    out = [F.normalize(model.features(image[i:i + batch_size], text[i:i + batch_size]), dim=-1)
           for i in range(0, len(image), batch_size)]
    return torch.cat(out)


def rgcl_loss(model: FusionHead, image: torch.Tensor, text: torch.Tensor, y: torch.Tensor,
              idx: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """Retrieval-guided contrastive loss (Mei et al., 2024, ACL).

    For each anchor, retrieved from this epoch's snapshot of the training set:
    the most similar *same-label* meme is the pseudo-gold positive and the most
    similar *opposite-label* meme is the hard negative; opposite-label memes in
    the batch are further negatives. Loss = -log softmax of the positive's
    cosine similarity (no temperature, as in the paper). It pulls a hateful
    meme towards other hateful memes and away from its benign look-alikes,
    which is exactly the confounder structure of this dataset.
    """
    anchor = F.normalize(model.features(image[idx], text[idx]), dim=-1)
    labels = y[idx]
    with torch.no_grad():
        sims = anchor @ bank.T
        sims[torch.arange(len(idx), device=idx.device), idx] = float("-inf")  # never retrieve yourself
        same = y[None, :] == labels[:, None]
        positive_idx = sims.masked_fill(~same, float("-inf")).argmax(1)
        negative_idx = sims.masked_fill(same, float("-inf")).argmax(1)
    positive = F.normalize(model.features(image[positive_idx], text[positive_idx]), dim=-1)
    negative = F.normalize(model.features(image[negative_idx], text[negative_idx]), dim=-1)
    s_positive = (anchor * positive).sum(-1)
    s_negative = (anchor * negative).sum(-1)
    s_batch = (anchor @ anchor.T).masked_fill(labels[None, :] == labels[:, None], float("-inf"))
    logits = torch.cat([s_positive[:, None], s_negative[:, None], s_batch], dim=1)
    return (torch.logsumexp(logits, dim=1) - s_positive).mean()


@torch.no_grad()
def knn_scores(model: FusionHead, train_split: dict, split: dict, k: int = 10) -> np.ndarray:
    """RGCL's retrieval classifier: similarity-weighted vote of the k nearest training memes.

    Returns a logit-like score (positive = hateful); temperature scaling on
    validation turns it into a probability. New labelled memes can be added to
    the bank without retraining.
    """
    device = model_device(model)
    train_image, train_text, train_y = tensors(train_split, device)
    bank = embed_split(model, train_image, train_text)
    image, text, _ = tensors(split, device)
    sims, neighbours = (embed_split(model, image, text) @ bank.T).topk(k, dim=1)
    votes = torch.where(train_y[neighbours] == 1, 1.0, -1.0)
    return (sims * votes).sum(1).cpu().numpy()


def train_one(splits: dict, args: argparse.Namespace, seed: int) -> tuple[FusionHead, dict]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    device = resolve_device(getattr(args, "device", "cpu"))
    dim = splits["train"]["image"].shape[1]
    model = FusionHead(dim, args.fusion, args.proj_dim, args.pre_layers, args.hidden, args.dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    image, text, y = tensors(splits["train"], device)
    pos_weight = None
    if args.class_weight:
        pos_weight = ((len(y) - y.sum()) / y.sum()).reshape(1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rgcl_weight = getattr(args, "rgcl", 0.0)

    best = {"auroc": -1.0, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, args.epochs + 1):
        bank = embed_split(model, image, text) if rgcl_weight else None  # refreshed every epoch
        model.train()
        order = torch.randperm(len(y), generator=generator).to(device)  # CPU generator: same order on any device
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(image[idx], text[idx]), y[idx])
            if rgcl_weight:
                loss = loss + rgcl_weight * rgcl_loss(model, image, text, y, idx, bank)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimiser.step()
            total += loss.item() * len(idx)
        val_auroc = roc_auc_score(splits["val"]["y"], predict_logits(model, splits["val"]))
        history.append({"epoch": epoch, "train_loss": round(total / len(y), 4), "val_auroc": round(val_auroc, 4)})
        if val_auroc > best["auroc"]:
            best = {"auroc": val_auroc, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= args.patience:
            break
    model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "epochs_run": len(history), "history": history}


def run_seed(splits: dict, args: argparse.Namespace, seed: int) -> dict:
    started = time.perf_counter()
    model, info = train_one(splits, args, seed)
    val_logits = predict_logits(model, splits["val"])
    test_logits = predict_logits(model, splits["test"])
    temperature = fit_temperature(val_logits, splits["val"]["y"])
    val_p = sigmoid(val_logits / temperature)
    test_raw = sigmoid(test_logits)
    test_p = sigmoid(test_logits / temperature)
    threshold = best_threshold(splits["val"]["y"], val_p)
    y = splits["test"]["y"]
    knn = {}
    if getattr(args, "knn", 0):
        val_s = knn_scores(model, splits["train"], splits["val"], args.knn)
        test_s = knn_scores(model, splits["train"], splits["test"], args.knn)
        knn_t = fit_temperature(val_s, splits["val"]["y"])
        knn_val_p, knn_test_p = sigmoid(val_s / knn_t), sigmoid(test_s / knn_t)
        knn = {"test_knn": summarise(y, knn_test_p, best_threshold(splits["val"]["y"], knn_val_p)),
               "val_knn_auroc": round(float(roc_auc_score(splits["val"]["y"], knn_val_p)), 4),
               "test_knn_p": knn_test_p}
    return {
        **knn,
        "seed": seed,
        "model": model,
        "temperature": round(temperature, 4),
        "val_threshold": round(threshold, 4),
        "val_auroc": round(float(roc_auc_score(splits["val"]["y"], val_p)), 4),
        "test": summarise(y, test_p, threshold),
        "test_uncalibrated_ece": round(expected_calibration_error(y, test_raw), 4),
        "test_p": test_p,
        "seconds": round(time.perf_counter() - started, 1),
        **{k: v for k, v in info.items() if k != "history"},
        "history": info["history"],
    }


def mean_sd(values: list[float]) -> list[float]:
    return [round(float(np.mean(values)), 4), round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4)]


def run_head(args: argparse.Namespace) -> dict:
    splits = load_splits(args.encoder, args.exclude_conflicts)
    sizes = {k: int(len(v["y"])) for k, v in splits.items()}
    device = resolve_device(args.device)
    print(f"{args.encoder} / {args.fusion}  sizes={sizes}  hate share train={splits['train']['y'].mean():.3f}"
          f"  device={device}", flush=True)
    runs = []
    for seed in SEEDS[: args.seeds]:
        r = run_seed(splits, args, seed)
        runs.append(r)
        t = r["test"]
        print(f"  seed {seed}: val {r['val_auroc']:.4f}  test AUROC {t['auroc']:.4f}  acc {t['accuracy@0.5']:.4f}"
              f"  ECE {r['test_uncalibrated_ece']:.3f}->{t['ece']:.3f}  epoch {r['best_epoch']}  {r['seconds']}s"
              + (f"  | kNN val {r['val_knn_auroc']:.4f} test {r['test_knn']['auroc']:.4f}" if "test_knn" in r else ""),
              flush=True)

    y = splits["test"]["y"]
    ensemble_p = np.mean([r["test_p"] for r in runs], axis=0)
    ensemble_threshold = float(np.mean([r["val_threshold"] for r in runs]))
    report = {
        "run": run_name(args),
        "date": time.strftime("%Y-%m-%d"),
        "encoder": args.encoder,
        "config": {k: getattr(args, k) for k in ("fusion", "proj_dim", "pre_layers", "hidden", "dropout", "lr",
                                                  "weight_decay", "grad_clip", "batch_size", "epochs", "patience",
                                                  "class_weight", "exclude_conflicts", "rgcl", "knn")},
        "trainable_parameters": count_parameters(runs[0]["model"]),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "sizes": sizes,
        "seeds": [r["seed"] for r in runs],
        "test_mean_sd": {m: mean_sd([r["test"][m] for r in runs]) for m in runs[0]["test"]},
        "val_auroc_mean_sd": mean_sd([r["val_auroc"] for r in runs]),
        "ensemble": {**summarise(y, ensemble_p, ensemble_threshold), "auroc_ci95": bootstrap_auroc(y, ensemble_p)},
        "test_without_conflicts_auroc": round(float(roc_auc_score(y[~splits["test"]["conflict"]],
                                                                  ensemble_p[~splits["test"]["conflict"]])), 4),
        "per_seed": [{k: v for k, v in r.items() if k not in ("model", "test_p", "test_knn_p")} for r in runs],
        "references_auroc_x100": REFERENCES,
    }
    if "test_knn" in runs[0]:
        knn_p = np.mean([r["test_knn_p"] for r in runs], axis=0)
        report["knn_test_mean_sd"] = {m: mean_sd([r["test_knn"][m] for r in runs]) for m in runs[0]["test_knn"]}
        report["knn_val_auroc_mean_sd"] = mean_sd([r["val_knn_auroc"] for r in runs])
        report["knn_ensemble"] = {**summarise(y, knn_p, 0.5), "auroc_ci95": bootstrap_auroc(y, knn_p)}
    save_artifacts(args, report, runs, splits["test"]["ids"], ensemble_p)
    return report


def run_name(args: argparse.Namespace) -> str:
    tag = f"_{args.tag}" if args.tag else ""
    return f"meme_{args.encoder}_{args.fusion}{tag}"


def save_artifacts(args, report: dict, runs: list[dict], test_ids: np.ndarray, ensemble_p: np.ndarray) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{report['run']}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Best seed by *validation* AUROC is the shippable head; test never picks it.
    best = max(runs, key=lambda r: r["val_auroc"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"config": best["model"].config, "encoder": args.encoder, "seed": best["seed"],
                "temperature": best["temperature"], "val_threshold": best["val_threshold"],
                "state_dict": best["model"].state_dict()}, MODEL_DIR / f"{report['run']}.pt")
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREDICTION_DIR / f"{report['run']}.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "p_hateful_ensemble"] + [f"p_seed{r['seed']}" for r in runs])
        for i, row_id in enumerate(test_ids):
            writer.writerow([int(row_id), round(float(ensemble_p[i]), 5)] + [round(float(r["test_p"][i]), 5) for r in runs])
    print(f"wrote {RESULTS_DIR / (report['run'] + '.json')}")


# ---------------------------------------------------------------- baselines

def run_logreg(args: argparse.Namespace) -> dict:
    """The MVP's logistic-regression probe, now on all 8,000 training memes."""
    from sklearn.linear_model import LogisticRegression

    splits = load_splits(args.encoder, args.exclude_conflicts)

    def features(split: dict, which: str) -> np.ndarray:
        img = split["image"] / np.linalg.norm(split["image"], axis=1, keepdims=True)
        txt = split["text"] / np.linalg.norm(split["text"], axis=1, keepdims=True)
        return {"image": img, "text": txt, "concat": np.hstack([img, txt]), "align": img * txt}[which]

    report = {"run": f"meme_{args.encoder}_logreg", "date": time.strftime("%Y-%m-%d"), "encoder": args.encoder,
              "results": {}, "references_auroc_x100": REFERENCES}
    y = splits["test"]["y"]
    for which in ("image", "text", "concat", "align"):
        # C picked on validation only.
        best_c, best_val = None, -1.0
        for c in (0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=c, max_iter=5000).fit(features(splits["train"], which), splits["train"]["y"])
            val = roc_auc_score(splits["val"]["y"], clf.predict_proba(features(splits["val"], which))[:, 1])
            if val > best_val:
                best_c, best_val = c, val
        clf = LogisticRegression(C=best_c, max_iter=5000).fit(features(splits["train"], which), splits["train"]["y"])
        p = clf.predict_proba(features(splits["test"], which))[:, 1]
        report["results"][which] = {"C": best_c, "val_auroc": round(best_val, 4),
                                    **summarise(y, p, 0.5), "auroc_ci95": bootstrap_auroc(y, p)}
        print(f"  logreg {which:7s} C={best_c:<5} val {best_val:.4f}  test AUROC {report['results'][which]['auroc']:.4f}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{report['run']}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="clip-b32")
    parser.add_argument("--fusion", choices=FUSIONS, default="align")
    parser.add_argument("--logreg", action="store_true", help="run the logistic-regression baselines instead")
    parser.add_argument("--proj-dim", type=int, default=1024)
    parser.add_argument("--pre-layers", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--class-weight", action="store_true", help="weight the loss by the class ratio")
    parser.add_argument("--exclude-conflicts", action="store_true",
                        help="drop train/val memes whose Kaggle and WOAH labels disagree")
    parser.add_argument("--rgcl", type=float, default=0.0,
                        help="weight of the retrieval-guided contrastive loss (RGCL default 1.0; 0 = off)")
    parser.add_argument("--knn", type=int, default=0, help="also evaluate RGCL's k-NN retrieval classifier (k)")
    parser.add_argument("--device", default="auto", help="auto (GPU if present), cuda or cpu")
    parser.add_argument("--tag", default="", help="suffix for the run name")
    args = parser.parse_args(argv)
    if args.logreg:
        run_logreg(args)
    else:
        run_head(args)


if __name__ == "__main__":
    main()

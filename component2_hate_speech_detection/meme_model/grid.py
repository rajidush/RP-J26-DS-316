"""Choose the head's hyper-parameters on validation only, then report the winner once.

    python -m component2_hate_speech_detection.meme_model.grid --encoder clip-b32

Every configuration below is trained with 3 seeds and ranked by mean
validation AUROC (ties: fewer parameters). Test AUROC is recorded for the
appendix but never used to choose. The winner is then re-run by `train.py`
with 5 seeds for the reported figure. Writes
`evaluation/results/meme_grid_<encoder>.json`.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from typing import Optional

from . import train
from .heads import count_parameters

BASE = {"proj_dim": 1024, "hidden": 1024, "lr": 1e-4, "weight_decay": 1e-4, "grad_clip": 0.1,
        "batch_size": 64, "epochs": 20, "patience": 5, "class_weight": False, "rgcl": 0.0, "knn": 0}


def configurations() -> list[dict]:
    configs = []
    for fusion, layers, dropout in itertools.product(("concat", "align"), (1, 3), (0.2, 0.4)):
        configs.append({**BASE, "fusion": fusion, "pre_layers": layers, "dropout": dropout})
    # The full interaction matrix is n^2 wide, so it only fits with a small projection.
    for layers, dropout in itertools.product((1, 3), (0.2, 0.4)):
        configs.append({**BASE, "fusion": "cross", "proj_dim": 128, "pre_layers": layers, "dropout": dropout})
    return configs


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="clip-b32")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="auto", help="auto (GPU if present), cuda or cpu")
    args = parser.parse_args(argv)

    splits = train.load_splits(args.encoder)
    rows = []
    for config in configurations():
        ns = argparse.Namespace(**config, device=args.device)
        runs = [train.run_seed(splits, ns, seed) for seed in train.SEEDS[: args.seeds]]
        row = {**config,
               "trainable_parameters": count_parameters(runs[0]["model"]),
               "val_auroc": train.mean_sd([r["val_auroc"] for r in runs]),
               "test_auroc": train.mean_sd([r["test"]["auroc"] for r in runs]),
               "best_epochs": [r["best_epoch"] for r in runs]}
        rows.append(row)
        print(f"  {config['fusion']:6s} layers={config['pre_layers']} dropout={config['dropout']} "
              f"proj={config['proj_dim']:4d}  val {row['val_auroc'][0]:.4f}+/-{row['val_auroc'][1]:.4f}  "
              f"test {row['test_auroc'][0]:.4f}  params {row['trainable_parameters']:,}", flush=True)

    rows.sort(key=lambda r: (-r["val_auroc"][0], r["trainable_parameters"]))
    winner = {k: rows[0][k] for k in ("fusion", "proj_dim", "pre_layers", "hidden", "dropout")}
    out = train.RESULTS_DIR / f"meme_grid_{args.encoder}.json"
    out.write_text(json.dumps({"date": time.strftime("%Y-%m-%d"), "encoder": args.encoder,
                               "rule": "max mean val AUROC over seeds; ties -> fewer parameters; test not used",
                               "seeds": list(train.SEEDS[: args.seeds]), "winner": winner,
                               "ranked": rows}, indent=2), encoding="utf-8")
    print(f"winner (by val): {winner}\nwrote {out}")


if __name__ == "__main__":
    main()

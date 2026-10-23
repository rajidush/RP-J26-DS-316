"""Fine-tune the self-harm specialist (sentinet/suicidality, ELECTRA-base) on the v2 data, 5 seeds, chosen on val.

    python -m component2_hate_speech_detection.self_harm_model.train            # GPU (Colab T4) or CPU
    python -m component2_hate_speech_detection.self_harm_model.train --smoke    # 1 seed, 64 rows: a code check

Recipe (fixed before training; one run, no search):
    full fine-tune from the shipped checkpoint, so the head starts where C2's engine is today;
    AdamW lr 2e-5, weight decay 0.01 (not on bias / LayerNorm), linear warm-up 10 %, 3 epochs,
    batch 32, max 128 tokens, fp16 autocast on GPU, gradient clipping 1.0; seeds 316-320.
Selection, on validation only: per seed, the epoch with the best macro-F1 at the engine's own decision
point (score >= 0.90, the self-harm solo_trust); across seeds, the best of those. Tests are scored for
every seed (mean +- sd) and for the chosen model, and never steer anything.

Two recipes, both with this recipe and these seeds: v2 (all three sources, the candidate to ship) and
v2_public_only (replay + Civil Comments, no C2-written pairs: what the pairs add).

Writes evaluation/results/self_harm_v2/model_level.json (every number) and saves each recipe's chosen
model to models/self_harm_v2/suicidality-c2v2[-public]/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Optional

from .data import C2, OUT as DATA, load
from ..evaluation.other_directed_set import SELF_HARM_BUCKETS, dev_cases as od_dev, heldout_cases as od_heldout
from ..evaluation.self_harm_set import dev_cases as sh_dev, heldout_cases as sh_heldout

BASE = "sentinet/suicidality"
NAME = "suicidality-c2v2"
MODEL_OUT = C2 / "models" / "self_harm_v2" / NAME
RESULTS = C2 / "evaluation" / "results" / "self_harm_v2"
SEEDS = (316, 317, 318, 319, 320)
TRUST = 0.90  # packs/policy/decision.json: self_harm_model.trust
CFG = {"lr": 2e-5, "weight_decay": 0.01, "warmup": 0.1, "epochs": 3, "batch_size": 32, "max_length": 128,
       "grad_clip": 1.0}
SH_HARM = ("direct", "ideation", "injury", "plan")
# v2 is the candidate to ship. v2_public_only is the ablation: the same recipe without the C2-written pairs,
# so the run shows what the pairs add over public data alone. Each is chosen on its own sources' val rows.
RECIPES = {"v2": None, "v2_public_only": ("replay", "threat")}
RECIPE_SUFFIX = {"v2": "", "v2_public_only": "-public"}


def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scores(model, tokenizer, texts: list[str], batch: int = 128) -> list[float]:
    """P(label 1) for each text, as the engine reads it (softmax over the two logits)."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tokenizer(texts[i:i + batch], padding=True, truncation=True, max_length=CFG["max_length"],
                            return_tensors="pt").to(model.device)
            with torch.autocast(model.device.type, dtype=torch.float16, enabled=model.device.type == "cuda"):
                logits = model(**enc).logits.float()
            out += torch.softmax(logits, dim=-1)[:, 1].tolist()
    return out


def auroc(labels: list[int], probs: list[float]) -> Optional[float]:
    pos = [p for p, y in zip(probs, labels) if y == 1]
    neg = [p for p, y in zip(probs, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def at_trust(labels: list[int], probs: list[float]) -> dict:
    tp = sum(p >= TRUST and y == 1 for p, y in zip(probs, labels))
    fp = sum(p >= TRUST and y == 0 for p, y in zip(probs, labels))
    fn = sum(p < TRUST and y == 1 for p, y in zip(probs, labels))
    tn = sum(p < TRUST and y == 0 for p, y in zip(probs, labels))
    f1_pos = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if tn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "macro_f1": round((f1_pos + f1_neg) / 2, 4)}


def val_metrics(rows: list[dict], probs: list[float]) -> dict:
    labels = [r["label"] for r in rows]
    out = {"all": {**at_trust(labels, probs), "auroc": auroc(labels, probs)}}
    for key in sorted({f"{r['source']}/{r['kind']}" for r in rows}):
        idx = [i for i, r in enumerate(rows) if f"{r['source']}/{r['kind']}" == key]
        out[key] = at_trust([labels[i] for i in idx], [probs[i] for i in idx])
    return out


def test_sets(external: Optional[dict] = None) -> dict[str, list[dict]]:
    """Every test set as rows {text, label, group}. Label 1 = should read as self-harm."""
    sets = {
        "c2_self_harm_dev": [{"text": c.text, "label": int(c.hate), "group": c.bucket} for c in sh_dev()],
        "c2_self_harm_heldout": [{"text": c.text, "label": int(c.hate), "group": c.bucket} for c in sh_heldout()],
        "c2_other_directed_dev": [{"text": c.text, "label": int(c.hate), "group": c.bucket} for c in od_dev()],
        "c2_other_directed_heldout": [{"text": c.text, "label": int(c.hate), "group": c.bucket} for c in od_heldout()],
    }
    return {**sets, **(external or {})}


def test_metrics(sets: dict[str, list[dict]], score_fn) -> dict:
    out = {}
    for name, rows in sets.items():
        probs = score_fn([r["text"] for r in rows])
        labels = [r["label"] for r in rows]
        groups = {}
        for g in sorted({r["group"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["group"] == g]
            groups[g] = {"n": len(idx), "label_1": sum(labels[i] for i in idx),
                         "at_or_above_trust": sum(probs[i] >= TRUST for i in idx),
                         "mean_score": round(statistics.fmean(probs[i] for i in idx), 4)}
        out[name] = {"n": len(rows), "label_1": sum(labels), "auroc": auroc(labels, probs),
                     **at_trust(labels, probs), "by_group": groups}
    return out


def headline(m: dict) -> dict:
    """The few numbers the acceptance rule reads, from one model's test metrics."""
    od = m["c2_other_directed_heldout"]["by_group"]
    other = [g for g in od if g not in SELF_HARM_BUCKETS]
    sh = [m[k]["by_group"] for k in ("c2_self_harm_dev", "c2_self_harm_heldout")]
    out = {
        "other_directed_heldout_read_as_self_harm": f"{sum(od[g]['at_or_above_trust'] for g in other)}/{sum(od[g]['n'] for g in other)}",
        "other_directed_heldout_self_harm_kept": f"{sum(od[g]['at_or_above_trust'] for g in SELF_HARM_BUCKETS)}/{sum(od[g]['n'] for g in SELF_HARM_BUCKETS)}",
        "known_film_lines_read_as_self_harm": f"{m['c2_other_directed_dev']['fp']}/{m['c2_other_directed_dev']['n']}",
        "c2_self_harm_sets_kept": f"{sum(b[g]['at_or_above_trust'] for b in sh for g in b if g in SH_HARM)}/{sum(b[g]['n'] for b in sh for g in b if g in SH_HARM)}",
        "c2_self_harm_sets_benign_above_trust": f"{sum(b[g]['at_or_above_trust'] for b in sh for g in b if g not in SH_HARM)}/{sum(b[g]['n'] for b in sh for g in b if g not in SH_HARM)}",
    }
    if "openai_moderation" in m:
        om = m["openai_moderation"]
        out["openai_moderation_self_harm_auroc"] = om["auroc"]
        out["openai_moderation_violence_above_trust"] = f"{om['by_group']['violence']['at_or_above_trust']}/{om['by_group']['violence']['n']}"
    if "xstest" in m:
        out["xstest_safe_above_trust"] = f"{m['xstest']['fp']}/{m['xstest']['n']}"
    if "hatecheck" in m:
        out["hatecheck_above_trust"] = f"{m['hatecheck']['fp']}/{m['hatecheck']['n']}"
    return out


def train_one(seed: int, train: list[dict], val: list[dict], log) -> tuple[dict, list[dict], dict]:
    """One seed. Returns the best epoch's weights (on CPU), the per-epoch curve and its val metrics."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed)
    torch.manual_seed(seed)
    dev = device()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, dtype=torch.float32).to(dev)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": CFG["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=CFG["lr"])
    steps = CFG["epochs"] * math.ceil(len(train) / CFG["batch_size"])
    sched = get_linear_schedule_with_warmup(opt, int(CFG["warmup"] * steps), steps)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    rng = random.Random(seed)
    curve, best, best_state = [], None, None
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses, t0 = [], time.time()
        for i in range(0, len(order), CFG["batch_size"]):
            batch = [train[j] for j in order[i:i + CFG["batch_size"]]]
            enc = tok([r["text"] for r in batch], padding=True, truncation=True, max_length=CFG["max_length"],
                      return_tensors="pt").to(dev)
            labels = torch.tensor([r["label"] for r in batch], device=dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        vm = val_metrics(val, scores(model, tok, [r["text"] for r in val]))
        point = {"seed": seed, "epoch": epoch, "train_loss": round(statistics.fmean(losses), 4),
                 "val_macro_f1": vm["all"]["macro_f1"], "val_auroc": vm["all"]["auroc"],
                 "seconds": round(time.time() - t0, 1)}
        curve.append(point)
        log(f"seed {seed} epoch {epoch}: loss {point['train_loss']:.4f}  val macro-F1@{TRUST} {point['val_macro_f1']:.4f}"
            f"  val AUROC {point['val_auroc']}  ({point['seconds']} s)")
        if best is None or vm["all"]["macro_f1"] > best["all"]["macro_f1"]:
            best, best_epoch = vm, epoch
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    best["epoch"] = best_epoch
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return best_state, curve, best


def load_model(state: Optional[dict] = None, path: Optional[str] = None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    src = path or BASE
    model = AutoModelForSequenceClassification.from_pretrained(src, dtype=torch.float32)
    if state is not None:
        model.load_state_dict(state)
    return model.to(device()).eval(), AutoTokenizer.from_pretrained(src)


def run_recipe(sources: Optional[tuple], train: list[dict], val: list[dict], sets: dict, seeds, out: Optional[Path],
               log=print) -> dict:
    """5 seeds on the rows from `sources` (None = all); the best seed on val is saved to `out`."""
    train = [r for r in train if sources is None or r["source"] in sources]
    val = [r for r in val if sources is None or r["source"] in sources]
    per_seed, curves, chosen = {}, [], None
    for seed in seeds:
        state, curve, best = train_one(seed, train, val, log)
        curves += curve
        model, tok = load_model(state)
        tests = test_metrics(sets, lambda t: scores(model, tok, t))
        per_seed[seed] = {"val": best, "tests": tests, "headline": headline(tests)}
        if chosen is None or best["all"]["macro_f1"] > per_seed[chosen]["val"]["all"]["macro_f1"]:
            chosen = seed
            if out is not None:
                out.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(out)
                tok.save_pretrained(out)
        del model, state
    return {"sources": list(sources) if sources else "all", "train_rows": len(train), "val_rows": len(val),
            "curves": curves, "per_seed": per_seed, "chosen_seed": chosen, "chosen": per_seed[chosen],
            "seed_spread": spread(per_seed), "saved_to": str(out.relative_to(C2)) if out else None}


def run(seeds=SEEDS, external: Optional[dict] = None, data_dir: Path = DATA, smoke: bool = False,
        recipes: Optional[dict] = None, log=print) -> dict:
    train, val = load("train", data_dir), load("val", data_dir)
    if smoke:
        train, val = train[:64], val[:32]
    sets = test_sets(external)
    report = {"base_model": BASE, "config": CFG, "seeds": list(seeds), "trust": TRUST,
              "train_rows": len(train), "val_rows": len(val), "smoke": smoke}

    model, tok = load_model()
    report["original"] = {"val": val_metrics(val, scores(model, tok, [r["text"] for r in val])),
                          "tests": test_metrics(sets, lambda t: scores(model, tok, t))}
    report["original"]["headline"] = headline(report["original"]["tests"])
    log(f"original model: val macro-F1@{TRUST} {report['original']['val']['all']['macro_f1']}")
    del model

    report["recipes"] = {}
    for name, sources in (recipes or RECIPES).items():
        log(f"== recipe {name}: {', '.join(sources) if sources else 'all sources'}")
        out = None if smoke else MODEL_OUT.parent / (NAME + RECIPE_SUFFIX[name])
        report["recipes"][name] = run_recipe(sources, train, val, sets, seeds, out, log)
    if not smoke:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "model_level.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def spread(per_seed: dict) -> dict:
    """Mean and sd over seeds of each test set's AUROC and macro-F1 at the trust point."""
    out = {}
    names = next(iter(per_seed.values()))["tests"].keys()
    for name in names:
        for metric in ("auroc", "macro_f1"):
            vals = [s["tests"][name][metric] for s in per_seed.values() if s["tests"][name][metric] is not None]
            if vals:
                out[f"{name}.{metric}"] = {"mean": round(statistics.fmean(vals), 4),
                                           "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0}
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-external", action="store_true", help="skip the three external test sets")
    args = parser.parse_args(argv)
    external = None
    if not args.no_external:
        from .external import load_all

        external = load_all()
    seeds = SEEDS[:1] if args.smoke else SEEDS[:args.seeds]
    report = run(seeds, external, smoke=args.smoke)
    print(json.dumps({"original": report["original"]["headline"],
                      **{name: {"chosen_seed": r["chosen_seed"], **r["chosen"]["headline"]}
                         for name, r in report["recipes"].items()}}, indent=2))


if __name__ == "__main__":
    main()

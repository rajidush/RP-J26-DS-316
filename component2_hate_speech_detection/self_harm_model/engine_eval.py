"""The fine-tuned self-harm model inside C2's full engine, against the shipped one, at ages 9, 12 and 15.

    python -m component2_hate_speech_detection.self_harm_model.engine_eval --model models/self_harm_v2/suicidality-c2v2
    python -m component2_hate_speech_detection.self_harm_model.engine_eval --model suicidality-c2v2 --hatecheck PATH

`--model` is a saved transformers folder (torch backend, as on Colab) or the name of an exported folder in
models/onnx/ (the laptop's ONNX path). Everything else in the engine is the shipped C2: same lexicon,
gates, thresholds and other heads (shared by both engines, so they load once).

Reports, per age:
    self-harm sets (evaluation/self_harm_set.py)    caught as self_harm, benign alerted (as evaluation/self_harm.py)
    other-directed set (other_directed_set.py)      read as self_harm (should fall), still alerted, alerted as threat
    the 107 general cases                           every decision that differs between the two engines
    HateCheck (optional, slow on CPU torch)         cases labelled self_harm, alerts by gold label
Writes evaluation/results/self_harm_v2/engine_level[_<tag>].json.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from ..evaluation.dev_set import all_cases as dev_cases
from ..evaluation.heldout_set import all_cases as heldout_cases
from ..evaluation.other_directed_set import SELF_HARM_BUCKETS, dev_cases as od_dev, heldout_cases as od_heldout
from ..evaluation.self_harm import evaluate as self_harm_evaluate
from ..evaluation.self_harm_set import dev_cases as sh_dev, heldout_cases as sh_heldout
from ..src.engine.analyzer import Analyzer
from ..src.engine.heads import DEFAULT_HEADS, TextHead, ZeroShotHead
from ..src.engine.policy import default_policy
from ..src.engine.text_scorer import TextScorer
from .train import RESULTS

AGES = (9, 12, 15)


def engines(model: str, backend: str = "auto") -> tuple[Analyzer, Analyzer]:
    """(shipped, fine-tuned): two analyzers that differ only in the self-harm head."""
    policy = default_policy()
    heads = [TextHead(m, backend) for m in DEFAULT_HEADS]
    grooming = policy.decision["grooming_model"]
    groom = ZeroShotHead(grooming["model"], grooming["hypotheses"])
    shipped = TextHead(policy.decision["self_harm_model"]["model"], backend)
    # A saved transformers folder always runs in torch (on Windows its path would pass for an ONNX folder).
    tuned = TextHead(model, "torch" if Path(model).is_dir() else backend)
    out = tuple(Analyzer(policy, TextScorer(policy, heads=heads, self_harm_head=h, grooming_head=groom))
                for h in (shipped, tuned))
    for a in out:
        a.scorer.warm_up()
    for h in (shipped, tuned):
        if not h.loaded:
            raise SystemExit(f"self-harm head {h.model_id} did not load; the comparison would be meaningless")
    return out


def other_directed(cases, analyzer: Analyzer) -> dict:
    out = {}
    for age in AGES:
        by = defaultdict(lambda: {"n": 0, "as_self_harm": 0, "alerted": 0, "as_threat": 0})
        for c in cases:
            v = analyzer.analyze_text(c.text, age)
            b = by[c.bucket]
            b["n"] += 1
            b["alerted"] += v.rung != "L0"
            b["as_self_harm"] += v.rung != "L0" and v.top_category == "self_harm"
            b["as_threat"] += v.rung != "L0" and v.top_category == "threat"
        other = [b for k, b in by.items() if k not in SELF_HARM_BUCKETS]
        own = [b for k, b in by.items() if k in SELF_HARM_BUCKETS]
        out[f"age_{age}"] = {
            "other_read_as_self_harm": f"{sum(b['as_self_harm'] for b in other)}/{sum(b['n'] for b in other)}",
            "other_still_alerted": f"{sum(b['alerted'] for b in other)}/{sum(b['n'] for b in other)}",
            "other_alerted_as_threat": f"{sum(b['as_threat'] for b in other)}/{sum(b['n'] for b in other)}",
            "self_harm_caught": f"{sum(b['as_self_harm'] for b in own)}/{sum(b['n'] for b in own)}" if own else "0/0",
            "by_bucket": {k: dict(v) for k, v in sorted(by.items())},
        }
    return out


def differences(cases, a: Analyzer, b: Analyzer) -> dict:
    changed = []
    for c in cases:
        for age in AGES:
            x, y = a.analyze_text(c.text, age), b.analyze_text(c.text, age)
            if (x.rung, x.top_category) != (y.rung, y.top_category):
                changed.append({"text": c.text, "age": age, "harmful": c.hate,
                                "shipped": [x.rung, x.top_category], "tuned": [y.rung, y.top_category]})
    return {"decisions_compared": len(cases) * len(AGES), "changed": changed}


def hatecheck(path: Path, analyzer: Analyzer, age: int = 12) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by = defaultdict(lambda: {"n": 0, "alerted": 0, "as_self_harm": 0})
    for r in rows:
        v = analyzer.analyze_text(r["test_case"].strip(), age)
        for key in (r["label_gold"], f"{r['functionality']}"):
            by[key]["n"] += 1
            by[key]["alerted"] += v.rung != "L0"
            by[key]["as_self_harm"] += v.rung != "L0" and v.top_category == "self_harm"
    return {"age": age, "cases": len(rows), "labelled_self_harm": sum(by[g]["as_self_harm"] for g in ("hateful", "non-hateful")),
            "by": {k: dict(v) for k, v in sorted(by.items())}}


def run(model: str, backend: str = "auto", hatecheck_csv: Path | None = None, tag: str = "", log=print) -> dict:
    t0 = time.time()
    shipped, tuned = engines(model, backend)
    log(f"engines ready in {time.time() - t0:.0f} s: {shipped.scorer.name} | {tuned.scorer.name}")
    report = {"shipped_engine": shipped.scorer.name, "tuned_engine": tuned.scorer.name, "model": model, "ages": list(AGES)}
    for name, analyzer in (("shipped", shipped), ("tuned", tuned)):
        report[name] = {
            "self_harm_dev": self_harm_evaluate(sh_dev(), analyzer),
            "self_harm_heldout": self_harm_evaluate(sh_heldout(), analyzer),
            "other_directed_dev": other_directed(od_dev(), analyzer),
            "other_directed_heldout": other_directed(od_heldout(), analyzer),
        }
        if hatecheck_csv:
            report[name]["hatecheck"] = hatecheck(hatecheck_csv, analyzer)
        log(f"{name} done ({time.time() - t0:.0f} s)")
    report["general_107_cases"] = differences(dev_cases() + heldout_cases(), shipped, tuned)
    report["seconds"] = round(time.time() - t0, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"engine_level{'_' + tag if tag else ''}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="auto", choices=("auto", "torch", "onnx"))
    parser.add_argument("--hatecheck", type=Path, help="HateCheck test_suite_cases.csv (age 12)")
    parser.add_argument("--tag", default="", help="suffix for the results file, e.g. onnx")
    args = parser.parse_args(argv)
    report = run(args.model, args.backend, args.hatecheck, args.tag)
    print(json.dumps({k: report[k] for k in ("shipped_engine", "tuned_engine", "seconds")}, indent=2))


if __name__ == "__main__":
    main()

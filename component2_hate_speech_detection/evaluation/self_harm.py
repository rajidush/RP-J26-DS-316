"""Self-harm recall through the full engine, and what the specialist changes elsewhere.

    python -m component2_hate_speech_detection.evaluation.self_harm --split dev       # design
    python -m component2_hate_speech_detection.evaluation.self_harm --split heldout   # report once

For each age band (9, 12, 15): self-harm messages alerted *as self-harm*,
benign lines alerted at all, and how many of the 107 dev + held-out cases get
a different rung or category with the self-harm specialist than without it
(the specialist must not change anything it was not built for). Held-out
results are written to `evaluation/results/self_harm.json` with the dev ones.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ..src.engine.analyzer import Analyzer
from ..src.engine.text_scorer import TextScorer
from .dev_set import all_cases as dev_cases
from .heldout_set import all_cases as heldout_cases
from .self_harm_set import dev_cases as sh_dev, heldout_cases as sh_heldout

AGES = (9, 12, 15)
HARM_BUCKETS = ("direct", "ideation", "injury", "plan")
RESULTS = Path(__file__).resolve().parent / "results" / "self_harm.json"


def evaluate(cases, analyzer: Analyzer) -> dict:
    out = {}
    for age in AGES:
        by = defaultdict(lambda: {"n": 0, "alerted": 0, "as_self_harm": 0})
        misses = []
        for c in cases:
            v = analyzer.analyze_text(c.text, age)
            b = by[c.bucket]
            b["n"] += 1
            b["alerted"] += v.rung != "L0"
            b["as_self_harm"] += v.rung != "L0" and v.top_category == "self_harm"
            if c.hate and not (v.rung != "L0" and v.top_category == "self_harm") and age == 9:
                misses.append(c.text)
        harm = [by[k] for k in HARM_BUCKETS if k in by]
        safe = [v for k, v in by.items() if k not in HARM_BUCKETS]
        out[f"age_{age}"] = {
            "self_harm_caught": f"{sum(b['as_self_harm'] for b in harm)}/{sum(b['n'] for b in harm)}",
            "benign_false_alerts": f"{sum(b['alerted'] for b in safe)}/{sum(b['n'] for b in safe)}",
            "by_bucket": {k: f"{v['as_self_harm'] if k in HARM_BUCKETS else v['alerted']}/{v['n']}" for k, v in by.items()},
        }
        if age == 9:
            out["missed_at_age_9"] = misses
    return out


def changes_elsewhere(with_sh: Analyzer, without_sh: Analyzer) -> dict:
    changed = []
    for c in dev_cases() + heldout_cases():
        for age in AGES:
            a, b = with_sh.analyze_text(c.text, age), without_sh.analyze_text(c.text, age)
            if (a.rung, a.top_category) != (b.rung, b.top_category):
                changed.append({"text": c.text, "age": age, "without": [b.rung, b.top_category], "with": [a.rung, a.top_category]})
    return {"decisions_compared": len(dev_cases() + heldout_cases()) * len(AGES), "changed": changed}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "heldout"), default="dev")
    args = parser.parse_args(argv)
    with_sh = Analyzer()
    with_sh.scorer.warm_up()
    without_sh = Analyzer(scorer=TextScorer(use_self_harm=False))
    report = {"engine": with_sh.scorer.name,
              "with_specialist": evaluate(sh_dev() if args.split == "dev" else sh_heldout(), with_sh),
              "without_specialist": evaluate(sh_dev() if args.split == "dev" else sh_heldout(), without_sh),
              "existing_107_cases": changes_elsewhere(with_sh, without_sh)}
    print(json.dumps(report, indent=2))
    if args.split == "heldout":
        dev = {"with_specialist": evaluate(sh_dev(), with_sh), "without_specialist": evaluate(sh_dev(), without_sh)}
        RESULTS.write_text(json.dumps({"dev": dev, "heldout": report}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

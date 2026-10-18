"""Grooming recall through the full engine, and what the grooming rules change elsewhere.

    python -m component2_hate_speech_detection.evaluation.grooming --split dev                    # design
    python -m component2_hate_speech_detection.evaluation.grooming --split heldout --before DIR   # report once

For each age band (9, 12, 15): grooming messages alerted *as grooming* (and
urgent), benign lines alerted at all, and misses at age 9, for the engine
with the grooming specialist and with the rules alone (the same heads,
`use_grooming=False`). Every decision on the 107 dev + held-out cases that
the specialist changes is listed. `--before DIR` loads an older policy pack
(the four `packs/policy` files, e.g. from `git show <commit>:<path>`) and
adds its numbers too. Held-out results are written to
`evaluation/results/grooming.json` with the dev ones.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ..src.engine.analyzer import Analyzer
from ..src.engine.policy import load_policy
from ..src.engine.text_scorer import TextScorer
from .dev_set import all_cases as dev_cases
from .grooming_set import HARM_BUCKETS, dev_cases as gr_dev, heldout_cases as gr_heldout
from .heldout_set import all_cases as heldout_cases

AGES = (9, 12, 15)
RESULTS = Path(__file__).resolve().parent / "results" / "grooming.json"


def evaluate(cases, analyzer: Analyzer) -> dict:
    out = {}
    for age in AGES:
        by = defaultdict(lambda: {"n": 0, "alerted": 0, "as_grooming": 0, "urgent": 0})
        misses = []
        for c in cases:
            v = analyzer.analyze_text(c.text, age)
            caught = v.rung != "L0" and v.top_category == "grooming"
            b = by[c.bucket]
            b["n"] += 1
            b["alerted"] += v.rung != "L0"
            b["as_grooming"] += caught
            b["urgent"] += caught and v.urgent
            if c.hate and not caught and age == 9:
                misses.append(f"{c.text}  [{v.rung} {v.top_category}]")
        harm = [by[k] for k in HARM_BUCKETS if k in by]
        safe = [v for k, v in by.items() if k not in HARM_BUCKETS]
        n_harm = sum(b["n"] for b in harm)
        out[f"age_{age}"] = {
            "grooming_caught": f"{sum(b['as_grooming'] for b in harm)}/{n_harm}",
            "urgent": f"{sum(b['urgent'] for b in harm)}/{n_harm}",
            "alerted_any_category": f"{sum(b['alerted'] for b in harm)}/{n_harm}",
            "benign_false_alerts": f"{sum(b['alerted'] for b in safe)}/{sum(b['n'] for b in safe)}",
            "by_bucket": {k: f"{v['as_grooming'] if k in HARM_BUCKETS else v['alerted']}/{v['n']}" for k, v in by.items()},
        }
        if age == 9:
            out["missed_at_age_9"] = misses
    return out


def changes_elsewhere(now: Analyzer, other: Analyzer) -> dict:
    changed = []
    for c in dev_cases() + heldout_cases():
        for age in AGES:
            a, b = now.analyze_text(c.text, age), other.analyze_text(c.text, age)
            if (a.rung, a.top_category, a.urgent) != (b.rung, b.top_category, b.urgent):
                changed.append({"text": c.text, "age": age, "other": [b.rung, b.top_category, b.urgent],
                                "now": [a.rung, a.top_category, a.urgent]})
    return {"decisions_compared": len(dev_cases() + heldout_cases()) * len(AGES), "changed": changed}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "heldout"), default="dev")
    parser.add_argument("--before", type=Path, help="folder holding an older policy pack to compare against")
    args = parser.parse_args(argv)

    now = Analyzer()
    now.scorer.warm_up()
    # The same model heads throughout, so only the specialist or the policy pack differs.
    shared = dict(heads=now.scorer.heads, self_harm_head=now.scorer.self_harm_head)
    rules_only = Analyzer(now.policy, TextScorer(now.policy, use_grooming=False, **shared))
    before: Optional[Analyzer] = None
    if args.before:
        old = load_policy(args.before)
        before = Analyzer(old, TextScorer(old, **shared))

    def report(cases) -> dict:
        r = {"rules_and_model": evaluate(cases, now), "rules_only": evaluate(cases, rules_only)}
        if before is not None:
            r["before"] = evaluate(cases, before)
        return r

    out = {"engine": now.scorer.name, "policy_version": now.policy.version,
           "split": args.split, **report(gr_dev() if args.split == "dev" else gr_heldout()),
           "existing_107_cases_model_vs_rules_only": changes_elsewhere(now, rules_only)}
    if before is not None:
        out["before_policy_version"] = before.policy.version
        out["existing_107_cases_now_vs_before"] = changes_elsewhere(now, before)
    print(json.dumps(out, indent=2))
    if args.split == "heldout":
        RESULTS.write_text(json.dumps({"dev": report(gr_dev()), "heldout": out}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

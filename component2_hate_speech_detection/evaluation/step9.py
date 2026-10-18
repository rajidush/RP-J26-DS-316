"""Step 9, RQ1 and RQ2: C2 against simple filters, and each design rule switched off.

    python -m component2_hate_speech_detection.evaluation.step9

Writes `results/step9.json` (every number) and `results/step9.md` (the tables).
Measurement only: nothing here is tuned.

Data
----
Child-context sets, hand-written and small (dev parts were used for design,
held-out parts were each reported once before this):
    general     107  (dev_set 67 + heldout_set 40): abuse, hate, threats, plus
                     gaming, reporting, figurative and ordinary chat that must not alert
    self-harm    73  (self_harm_set: 40 harmful, 33 benign)
    grooming    108  (grooming_set: 60 harmful, 48 benign)
Public sets, adult social media (a check on another register, not on children;
the MVP used them to set the corroboration rule, so they are not independent),
sampled with a fixed seed (27; see SEED):
    davidson    Davidson et al. 2017 tweets: 300 hate, 300 offensive, 300 neither
    berkeley    Measuring Hate Speech (Kennedy et al. 2020): 450 hateful
                (hate_speech_score > 0.5), 450 not (score < -1)

Configurations
--------------
    RQ1  keyword layer only · Dynabench alone · toxic-bert alone · full C2
    RQ2  full C2 with one rule switched off: corroboration (max of the heads),
         framing, gaming cap, self-harm specialist, grooming specialist,
         no-real-words check
A single model alone is scored as the "simple local filter": its reading and
category go straight to the same age thresholds.

Metrics, at ages 9, 12 and 15 (an alert is any rung above L0): harmful
alerted, benign alerted, and on the general set the reporting and gaming
lines alerted. F1 is for alert vs harmful.
"""
from __future__ import annotations

import glob
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from ..src.engine.analyzer import Analyzer
from ..src.engine.decide import decide, persona_for_age
from ..src.engine.text_scorer import TextScorer
from .dev_set import all_cases as general_dev
from .grooming_set import dev_cases as grooming_dev, heldout_cases as grooming_heldout
from .heldout_set import all_cases as general_heldout
from .self_harm_set import dev_cases as self_harm_dev, heldout_cases as self_harm_heldout

AGES = (9, 12, 15)
# Seed 26 drew the sample that exposed the first no-words rule silencing slur-only
# messages; after that fix, the reported numbers use a fresh sample.
SEED = 27
RESULTS = Path(__file__).resolve().parent / "results"
HF = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


@dataclass(frozen=True)
class Item:
    dataset: str
    split: str
    text: str
    harmful: bool
    kind: str  # bucket (child sets) or label name (public sets)


def child_items() -> List[Item]:
    items = []
    for name, parts in (("general", (("dev", general_dev), ("heldout", general_heldout))),
                        ("self_harm", (("dev", self_harm_dev), ("heldout", self_harm_heldout))),
                        ("grooming", (("dev", grooming_dev), ("heldout", grooming_heldout)))):
        for split, cases in parts:
            items += [Item(name, split, c.text, c.hate, c.bucket) for c in cases()]
    return items


def _parquet(repo: str) -> str:
    found = glob.glob(str(HF / f"datasets--{repo.replace('/', '--')}" / "snapshots" / "*" / "data" / "*.parquet"))
    if not found:
        raise FileNotFoundError(f"{repo} is not in the local Hugging Face cache")
    return found[0]


def public_items() -> List[Item]:
    import pandas as pd

    rng = random.Random(SEED)
    d = pd.read_parquet(_parquet("tdavidson/hate_speech_offensive"))
    items = []
    for cls, name, harmful in ((0, "hate", True), (1, "offensive", True), (2, "neither", False)):
        rows = d[d["class"] == cls]["tweet"].tolist()
        items += [Item("davidson", "public", t, harmful, name) for t in rng.sample(rows, 300)]
    b = pd.read_parquet(_parquet("ucberkeley-dlab/measuring-hate-speech"),
                        columns=["comment_id", "text", "hate_speech_score"])
    u = b.groupby("comment_id").agg(text=("text", "first"), score=("hate_speech_score", "first"))
    for name, rows, harmful in (("hateful", u[u.score > 0.5]["text"].tolist(), True),
                                ("not_hateful", u[u.score < -1]["text"].tolist(), False)):
        items += [Item("berkeley", "public", t, harmful, name) for t in rng.sample(sorted(rows), 450)]
    return items


class Cached:
    """A head whose readings are kept, so every configuration reuses one model pass per text."""

    def __init__(self, head) -> None:
        self.head, self._memo = head, {}

    @property
    def name(self) -> str:
        return self.head.name

    @property
    def loaded(self) -> bool:
        return self.head.loaded

    def read(self, text: str):
        if text not in self._memo:
            self._memo[text] = self.head.read(text)
        return self._memo[text]


Rung = Callable[[str, int], str]


def configurations() -> Tuple[TextScorer, Dict[str, Rung]]:
    base = TextScorer()
    base.warm_up()
    heads = [Cached(h) for h in base.heads]
    shared = dict(heads=heads, self_harm_head=Cached(base.self_harm_head), grooming_head=Cached(base.grooming_head))

    def engine(**flags) -> Rung:
        analyzer = Analyzer(scorer=TextScorer(**shared, **flags))
        return lambda text, age: analyzer.analyze_text(text, age).rung

    def single(head) -> Rung:
        def rung(text: str, age: int) -> str:
            score, category, _ = head.read(text)
            return decide(category or "bullying", score or 0.0, persona_for_age(age)).rung
        return rung

    keyword = Analyzer(scorer=TextScorer(use_heads=False))
    return base, {
        "keyword layer only": lambda text, age: keyword.analyze_text(text, age).rung,
        "Dynabench alone": single(heads[0]),
        "toxic-bert alone": single(heads[1]),
        "full C2": engine(),
        "without corroboration": engine(use_corroboration=False),
        "without framing": engine(use_framing=False),
        "without gaming cap": engine(use_gaming_cap=False),
        "without self-harm specialist": engine(use_self_harm=False),
        "without grooming specialist": engine(use_grooming=False),
        "without no-real-words check": engine(use_word_check=False),
    }


def _f1(tp: int, fp: int, fn: int) -> float:
    return round(2 * tp / (2 * tp + fp + fn), 3) if tp else 0.0


def measure(items: List[Item], configs: Dict[str, Rung]) -> dict:
    out: dict = {}
    for name, rung in configs.items():
        out[name] = {}
        for age in AGES:
            c = defaultdict(lambda: [0, 0])  # key -> [alerted, n]
            for it in items:
                alerted = rung(it.text, age) != "L0"
                for key in (f"{it.dataset}|{'harmful' if it.harmful else 'benign'}",
                            f"{it.dataset}|kind:{it.kind}",
                            f"{it.dataset}|{it.split}|{'harmful' if it.harmful else 'benign'}"):
                    c[key][0] += alerted
                    c[key][1] += 1
            row = {k: f"{v[0]}/{v[1]}" for k, v in sorted(c.items())}
            for ds in sorted({it.dataset for it in items}):
                tp, n_h = c[f"{ds}|harmful"]
                fp, _ = c[f"{ds}|benign"]
                row[f"{ds}|f1"] = _f1(tp, fp, n_h - tp)
            out[name][f"age_{age}"] = row
        print(f"  {name}", flush=True)
    return out


RQ1 = ("keyword layer only", "Dynabench alone", "toxic-bert alone", "full C2")
RQ2 = ("full C2", "without corroboration", "without framing", "without gaming cap", "without self-harm specialist",
       "without grooming specialist", "without no-real-words check")
CHILD_COLUMNS = (("general|harmful", "General: harm caught"), ("general|benign", "General: false alerts"),
                 ("general|kind:reporting", "Child reporting: alerted"), ("general|kind:gaming", "Gaming: alerted"),
                 ("self_harm|harmful", "Self-harm: caught"), ("self_harm|benign", "Self-harm: false alerts"),
                 ("grooming|harmful", "Grooming: caught"), ("grooming|benign", "Grooming: false alerts"))
PUBLIC_COLUMNS = (("davidson|kind:hate", "Davidson hate: caught"), ("davidson|kind:offensive", "Davidson offensive: caught"),
                  ("davidson|kind:neither", "Davidson neither: false alerts"), ("davidson|f1", "Davidson F1"),
                  ("berkeley|kind:hateful", "Berkeley hateful: caught"),
                  ("berkeley|kind:not_hateful", "Berkeley not hateful: false alerts"), ("berkeley|f1", "Berkeley F1"))


def _table(results: dict, rows: Tuple[str, ...], columns, age: int) -> str:
    lines = ["| | " + " | ".join(title for _, title in columns) + " |",
             "|---|" + "---|" * len(columns)]
    for name in rows:
        cells = [str(results[name][f"age_{age}"].get(key, "")) for key, _ in columns]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown(child: dict, public: dict, engine: str, policy: str) -> str:
    parts = [f"# Step 9 results (RQ1, RQ2)\n\nEngine `{engine}`, policy pack {policy}. "
             "Generated by `evaluation/step9.py`; an alert is any rung above L0.\n"]
    for age in AGES:
        parts.append(f"## RQ1: C2 against simple filters, age {age} ({persona_for_age(age)})\n\n"
                     "Child-context sets\n\n" + _table(child, RQ1, CHILD_COLUMNS, age) +
                     "\n\nPublic sets (adult social media)\n\n" + _table(public, RQ1, PUBLIC_COLUMNS, age) + "\n")
    for age in AGES:
        parts.append(f"## RQ2: one rule switched off at a time, age {age}\n\n" + _table(child, RQ2, CHILD_COLUMNS, age) +
                     "\n\n" + _table(public, RQ2, PUBLIC_COLUMNS, age) + "\n")
    return "\n".join(parts)


def main() -> None:
    base, configs = configurations()
    child, public = child_items(), public_items()
    print(f"{len(child)} child-context items, {len(public)} public items", flush=True)
    print("child-context sets:", flush=True)
    child_results = measure(child, configs)
    print("public sets:", flush=True)
    public_results = measure(public, configs)
    report = {"engine": base.name, "policy_version": Analyzer(scorer=base).policy.version, "seed": SEED,
              "items": {"child": len(child), "public": len(public)},
              "child_context": child_results, "public": public_results}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "step9.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "step9.md").write_text(markdown(child_results, public_results, report["engine"], report["policy_version"]),
                                      encoding="utf-8")
    print(f"wrote {RESULTS / 'step9.json'} and step9.md")


if __name__ == "__main__":
    main()

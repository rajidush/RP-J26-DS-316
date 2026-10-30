"""
Evaluate EVALUATE's emotional_state (the default assessor: the rules arm in
src/emotion_rules.py and the shared derivation in src/assessor.py) against the
simulated dataset in out/, by running the real FSM over the saved child replies
(no model calls).

The PP1 rules were written on the DEVELOPMENT children only. The 29 Oct codebook
rules (ticket 02) were written from the codebook and the panel's examples, not
from this dataset. Report HELD-OUT numbers as the result and do not change the
rules after looking at them.

    dev     C001-C020   used to write and tune the rules
    heldout C021-C040   evaluation only

Run from component3_socratic_educator/:
    python -m evidence.evaluate_emotion_rules --split dev --show-errors 15
    python -m evidence.evaluate_emotion_rules --split heldout

Agreement is measured against the archetype each simulated child was
generated with. That is the generator's intent, not ground truth: the LLM
sometimes drifts from its brief, so this is "agreement with intended label",
not accuracy.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import generate_educator_dataset as g

OUT = Path(__file__).resolve().parents[1] / "out"
SPLITS = {"dev": range(1, 21), "heldout": range(21, 41), "all": range(1, 41)}
LABELS = ["calm", "defensive", "distressed", "unclear"]


def load(split):
    with open(OUT / "scenario_seeds.csv", newline="", encoding="utf-8") as f:
        seeds = [s for s in csv.DictReader(f) if int(s["child_id"][1:]) in SPLITS[split]]
    for s in seeds:
        s["confidence_score"] = float(s["confidence_score"])
    with open(OUT / "child_replies.jsonl", encoding="utf-8") as f:
        replies = {r["session_id"]: r["replies"] for r in map(json.loads, f)}
    return seeds, replies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--show-errors", type=int, default=0, help="print N misclassified sessions")
    args = ap.parse_args()

    seeds, replies = load(args.split)
    fsm = g.make_controller(stub_generator=True)
    templates = g.load_templates()
    rows = []
    for s in seeds:
        script, state = replies[s["session_id"]], {"i": 0, "last": ""}

        def respond(_p, script=script, state=state):
            if state["i"] < len(script):
                state["last"] = script[state["i"]]
                state["i"] += 1
            return state["last"]

        out, transcript = fsm.run(g.make_trigger(s, templates), respond)
        labelled = fsm.assessor.assess(transcript.turns).replies
        rows.append((s, g.ARCHETYPES[s["archetype"]][1], out.emotional_state.value, labelled))

    n = len(rows)
    agree = sum(i == o for _, i, o, _ in rows)
    print(f"split={args.split}  sessions={n}  children={len({s['child_id'] for s, *_ in rows})}")
    print(f"agreement with intended state: {agree}/{n} = {agree / n:.1%}\n")
    conf = Counter((i, o) for _, i, o, _ in rows)
    print("intended \\ output " + "".join(f"{l:>12}" for l in LABELS) + f"{'agree':>9}")
    for i in LABELS:
        total = sum(conf[(i, o)] for o in LABELS)
        if total:
            print(f"{i:<17}" + "".join(f"{conf[(i, o)]:>12}" for o in LABELS) + f"{conf[(i, i)] / total:>9.0%}")
    print("\noutput counts:", dict(Counter(o for *_, o, _ in rows)))

    errors = [(s, i, o, h) for s, i, o, h in rows if i != o]
    for s, i, o, labelled in errors[: args.show_errors]:
        print(f"\n{s['session_id']} intended={i} output={o}")
        for a in labelled:   # each reply's three parts, with the phrases that fired
            parts = " / ".join(v + (f" {a.evidence[k]}" if a.evidence[k] else "") for k, v in a.labels.items())
            print(f"   {a.reply!r}: {parts}")


if __name__ == "__main__":
    main()

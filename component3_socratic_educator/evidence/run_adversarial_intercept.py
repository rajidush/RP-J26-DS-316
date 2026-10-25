"""
Adversarial test against constrained INTERCEPT (plan, Wed 7 Oct; expanded for
Mon 12 Oct; ADR 0004).

Each attack template is crossed with every risk category and injected into the
INTERCEPT prompt as untrusted upstream context. This simulates a future
interface: today Component 3 reads only risk_category and confidence_score, so
no upstream text reaches the prompt.

    --set seen     data/adversarial_intercept_attacks.json (the post-check was tuned on it, 7 Oct)
    --set unseen   data/adversarial_intercept_unseen_attacks.json (held out, with a no-attack control)
    --set both     both, in one run (default)

Every delivered line is scored twice: by the post-check rules (`violations`,
0 by construction) and by the measurement-only audit (src/intercept_audit.py).
Calls use the fixed config.INTERCEPT_TEMPERATURE.

Needs LM Studio running on localhost:1234; refuses to record a run in which the
model was unreachable. Run from component3_socratic_educator/:
    python -m evidence.run_adversarial_intercept [--set both] [--tag after-fix]

Writes, in the run folder evidence/intercept/adversarial_intercept_<date>[_<tag>]/, a
.jsonl (one row per case), a .summary.json, a .log.txt (the results table it
prints) and a .review.md (every model line, with a blank column for the
owner's verdict); never overwrites a recorded run.
--tag names a second run on the same day.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from evidence.run_info import lm_studio_version
from src import config, intercept_audit
from src.adversarial import ATTACKS_FILE, UNSEEN_ATTACKS_FILE, adversarial_cases, load_attack_set, run_adversarial_batch
from src.grammar_decoder import INTERCEPT_PROMPT, INTERCEPT_SCHEMA, UNTRUSTED_CONTEXT_FENCE, GrammarConstrainedGenerator

OUT_DIR = Path(__file__).resolve().parent / "intercept"   # new runs go in their topic folder (evidence/README.md)
SETS = {"seen": ATTACKS_FILE, "unseen": UNSEEN_ATTACKS_FILE}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_sheet(rows) -> str:
    lines = ["# Model lines delivered in this run: owner review", "",
             "Every line the model wrote that reached the child (post-check passed). `audit` is the",
             "measurement-only audit's flags. Fill in `verdict` (ok / leak / other, with a word why).", "",
             "| id | set | style | category | question | audit | verdict |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["source"] == "model":
            question = intercept_audit.question_of(r["delivered"]).strip()
            lines.append(f"| {r['attack_id']} | {r['test_set']} | {r['style']} | {r['risk_category']} | "
                         f"{question} | {', '.join(r['audit_flags']) or '-'} |  |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("seen", "unseen", "both"), default="both")
    ap.add_argument("--tag", help="suffix for a second run on the same day, e.g. after-fix")
    args = ap.parse_args()

    names = list(SETS) if args.set == "both" else [args.set]
    attack_sets = {name: load_attack_set(SETS[name]) for name in names}
    cases = [case for name, attack_set in attack_sets.items()
             for case in adversarial_cases(attack_set["attacks"], test_set=name)]

    stamp = datetime.now().strftime("%Y-%m-%d")
    run_name = f"adversarial_intercept_{stamp}{'_' + args.tag if args.tag else ''}"
    out = OUT_DIR / run_name / f"{run_name}.jsonl"           # one folder per run
    outputs = [out] + [out.with_suffix(suffix) for suffix in (".summary.json", ".log.txt", ".review.md")]
    if any(p.exists() for p in outputs):     # check before spending minutes on model calls
        sys.exit(f"{out.name} (or its summary/log/review) already exists: pass --tag. Nothing was recorded.")

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    generator = GrammarConstrainedGenerator()
    report = run_adversarial_batch(generator, cases)
    n = report["total_prompts"]
    if report["unreachable"]:
        sys.exit(f"{report['unreachable']}/{n} cases got no model reply: start the LM Studio server and rerun. "
                 "Nothing was recorded.")

    out.parent.mkdir(exist_ok=True)

    def write(path, text):
        with open(path, "x", encoding="utf-8", newline="\n") as f:   # never overwrite a recorded run
            f.write(text)

    write(outputs[0], "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in report["rows"]))
    summary = {
        "run_started": started,
        "caveats": {name: s["description"] for name, s in attack_sets.items()},
        "scoring": ("violations: delivered lines breaking the grammar or post-check rules (0 by construction). "
                    "audit_*: the measurement-only audit (src/intercept_audit.py, its sha256 under settings), "
                    "a broader phrase list; a flag marks a line for review, it is not a verdict, and unflagged "
                    "lines can still be wrong."),
        **{k: v for k, v in report.items() if k != "rows"},
        "settings": {"model_id": config.LOCAL_MODEL_ID, "server": "LM Studio (llama.cpp engine)",
                     "lm_studio_version": lm_studio_version(), "hardware": "not recorded",
                     "response_format": "json_schema (strict)", "schema": INTERCEPT_SCHEMA,
                     "prompt": INTERCEPT_PROMPT, "untrusted_context_fence": UNTRUSTED_CONTEXT_FENCE,
                     "max_tokens": 100, "temperature": generator.temperature,
                     "other_sampling": "server defaults (not set)", "runs_per_case": 1,
                     "attack_files": {name: {"file": SETS[name].name, "sha256": _sha256(SETS[name]),
                                             "attacks": s["attacks"]} for name, s in attack_sets.items()},
                     "audit_rules_sha256": _sha256(Path(intercept_audit.__file__))},
        "limitations": [
            f"One sample per case at temperature {generator.temperature}: rates vary between runs.",
            "Simulated future interface: no upstream text reaches the INTERCEPT prompt today, "
            "except in the no_attack control, which is the production prompt.",
            "The audit is a phrase list: unflagged lines can still be wrong; see the .review.md sheet.",
            "Latency here is wall-clock per case on an unrecorded machine; it is not the Tue 13 Oct latency measurement.",
        ],
    }
    write(outputs[1], json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write(outputs[3], _review_sheet(report["rows"]))

    lat = report["latency_s"]
    log = [f"{n} adversarial cases ({' + '.join(names)}) at temperature {generator.temperature} -> {out.name}",
           "Simulated future interface: no upstream text reaches the INTERCEPT prompt today.",
           f"post-check violations delivered : {report['violations']}/{n} (0 by construction)",
           f"audit-flagged lines delivered   : {report['audit_flagged']}/{n} ({report['audit_rate']:.0%}) "
           f"{report['audit_flags']}",
           f"passed {report['passed']}, blocked by the post-check {report['blocked_by_post_check']}, "
           f"schema breaks {report['schema_breaks']}",
           f"seconds per case, wall-clock (min/median/max): {lat['min']} / {lat['median']} / {lat['max']}",
           "",
           "Per style (post-check violations left out: 0 by construction; audit = flagged lines):",
           f"  {'set':<7}{'style':<26}{'cases':>6}{'passed':>8}{'blocked':>9}{'schema':>8}{'audit':>7}{'rate':>7}"]
    table = [(t["test_set"], style, t) for style, t in report["by_style"].items()]
    table += [(name, "(all)", t) for name, t in report["by_set"].items()]
    log += [f"  {test_set:<7}{style:<26}{t['cases']:>6}{t['passed']:>8}{t['blocked']:>9}"
            f"{t['schema_breaks']:>8}{t['audit_flagged']:>7}{t['audit_rate']:>7.0%}" for test_set, style, t in table]
    log += ["", f"Review every model line by eye: {outputs[3].name}"]
    write(outputs[2], "\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()

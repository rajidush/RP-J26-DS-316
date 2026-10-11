"""
Adversarial test against constrained INTERCEPT (plan, Wed 7 Oct task; ADR 0004).

Each attack template in data/adversarial_intercept_attacks.json is crossed with
every risk category and injected into the INTERCEPT prompt as untrusted upstream
context. This simulates a future interface: today Component 3 reads only
risk_category and confidence_score, so no upstream text reaches the prompt.

Needs LM Studio running on localhost:1234; refuses to record a run in which the
model was unreachable. Run from component3_socratic_educator/:
    python -m evidence.run_adversarial_intercept [--tag after-fix]

Writes evidence/adversarial_intercept_<date>[_<tag>].jsonl (one row per case,
never overwritten) plus a .summary.json, and prints the test log.
--tag names a second run on the same day.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src import config
from src.adversarial import ATTACKS_FILE, adversarial_cases, load_attack_set, run_adversarial_batch
from src.grammar_decoder import INTERCEPT_PROMPT, INTERCEPT_SCHEMA, UNTRUSTED_CONTEXT_FENCE, GrammarConstrainedGenerator

OUT_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="suffix for a second run on the same day, e.g. after-fix")
    args = ap.parse_args()

    attack_set = load_attack_set()
    attacks = attack_set["attacks"]
    report = run_adversarial_batch(GrammarConstrainedGenerator(), adversarial_cases(attacks))
    n = report["total_prompts"]
    if report["unreachable"]:
        sys.exit(f"{report['unreachable']}/{n} cases got no model reply: start the LM Studio server and rerun. "
                 "Nothing was recorded.")

    stamp = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"adversarial_intercept_{stamp}{'_' + args.tag if args.tag else ''}.jsonl"
    with open(out, "x", encoding="utf-8", newline="\n") as f:   # never overwrite a recorded run
        for r in report["rows"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"caveat": attack_set["description"], **{k: v for k, v in report.items() if k != "rows"},
               "settings": {"model_id": config.LOCAL_MODEL_ID, "server": "LM Studio (llama.cpp engine)",
                            "response_format": "json_schema (strict)", "schema": INTERCEPT_SCHEMA,
                            "prompt": INTERCEPT_PROMPT, "untrusted_context_fence": UNTRUSTED_CONTEXT_FENCE,
                            "max_tokens": 100, "temperature": "server default (not set)",
                            "attacks_file": ATTACKS_FILE.name, "attacks": attacks}}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")

    print(f"{n} adversarial cases ({len(attacks)} attacks x {n // len(attacks)} categories) -> {out}")
    print("Simulated future interface: no upstream text reaches the INTERCEPT prompt today.")
    print(f"violations delivered to the child : {report['violations']}/{n} ({report['violation_rate']:.0%})")
    print(f"passed (model line kept)          : {report['passed']}/{n}")
    print(f"blocked by the post-check         : {sum(report['blocked_by_post_check'].values())}/{n} "
          f"{report['blocked_by_post_check']}")
    print(f"schema breaks (grammar layer)     : {sum(report['schema_breaks'].values())}/{n} {report['schema_breaks']}")
    lat = report["latency_s"]
    print(f"latency s, model calls (min/median/max): {lat['min']} / {lat['median']} / {lat['max']}")
    for style in dict.fromkeys(r["style"] for r in report["rows"]):
        rows = [r for r in report["rows"] if r["style"] == style]
        print(f"  {style:<22} passed {sum(r['source'] == 'model' for r in rows)}/{len(rows)}, "
              f"blocked {sum(r['source'] == 'fallback' for r in rows)}")
    print("\nKept model lines: check each by eye for a leading hint the rules cannot catch:")
    for r in report["rows"]:
        if r["source"] == "model":
            print(f"  [{r['attack_id']} {r['risk_category']}] {r['delivered']}")


if __name__ == "__main__":
    main()

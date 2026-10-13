"""
Capture INTERCEPT samples from the live model, as evidence for the plan's
5 Oct task (real SLM wired into Intercept, unconstrained: the "before") and
6 Oct task (constrained decoding, ADR 0004: the "after").

Needs LM Studio running on localhost:1234. Run from component3_socratic_educator/:
    python -m evidence.capture_intercept_samples --runs 2

Writes evidence/intercept_samples_constrained_<date>.jsonl (one row per sample)
plus a .settings.json, and prints a before/after table. The 6 Oct "before" file
is only read, never rewritten; both sides are scored with today's grammar and
post-check, so they are measured with the same yardstick.
"""
import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from src import config
from src.grammar_decoder import (_ALERT_WORDS, _CATEGORY_WORDS, _FORBIDDEN_SUBSTRINGS,
                                 INTERCEPT_PROMPT, INTERCEPT_SCHEMA, GrammarConstrainedGenerator, _has_word)
from src.schemas import RiskCategory

OUT_DIR = Path(__file__).resolve().parent
BEFORE = OUT_DIR / "intercept_samples_2026-10-06.jsonl"


def score(text: str, gen: GrammarConstrainedGenerator) -> dict:
    return {
        "matches_grammar": gen.validate("INTERCEPT", text),
        "forbidden_word": any(bad in text.lower() for bad in _FORBIDDEN_SUBSTRINGS),
        "alert_language": _has_word(_ALERT_WORDS, text),
        "category_named": _has_word(_CATEGORY_WORDS, text),
    }


def summary(rows: list[dict]) -> dict:
    n = len(rows)
    secs = [r["seconds"] for r in rows]
    return {
        "samples": n,
        "matches_grammar": sum(r["matches_grammar"] for r in rows),
        "forbidden_word": sum(r["forbidden_word"] for r in rows),
        "alert_language": sum(r["alert_language"] for r in rows),
        "category_named": sum(r["category_named"] for r in rows),
        "fallback": sum(r.get("source") == "fallback" for r in rows),
        "latency_s": {"min": min(secs), "median": statistics.median(secs), "max": max(secs)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2, help="live samples per risk category")
    args = ap.parse_args()

    gen = GrammarConstrainedGenerator()
    rows = []
    for rc in RiskCategory:
        for run in range(args.runs):
            start = time.time()
            line = gen.generate_with_source("INTERCEPT", rc)
            rows.append({"risk_category": rc.value, "run": run, "text": line.text, "source": line.source,
                         "fallback_reason": line.reason, "seconds": round(time.time() - start, 2),
                         **score(line.text, gen)})

    stamp = datetime.now().strftime("%Y-%m-%d")   # local date, as in the RP diary
    out = OUT_DIR / f"intercept_samples_constrained_{stamp}.jsonl"
    with open(out, "x", encoding="utf-8", newline="\n") as f:   # never overwrite recorded samples
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    settings = {"model_id": config.LOCAL_MODEL_ID, "server": "LM Studio (llama.cpp engine)",
                "response_format": "json_schema (strict)", "schema": INTERCEPT_SCHEMA, "prompt": INTERCEPT_PROMPT,
                "max_tokens": 100, "temperature": "server default (not set)", "runs_per_category": args.runs}
    out.with_suffix(".settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    before = [json.loads(line) for line in BEFORE.read_text(encoding="utf-8").splitlines()]
    before = [{**score(r["live_text"], gen), "seconds": r["seconds"]} for r in before]
    b, a = summary(before), summary(rows)
    print(f"{a['samples']} samples -> {out}")
    print(f"{'measure (same grammar + post-check)':<38}{'before 6 Oct':>14}{'after':>10}")
    for key in ("matches_grammar", "forbidden_word", "alert_language", "category_named", "fallback"):
        print(f"{key:<38}{b[key]:>11}/{b['samples']}{a[key]:>7}/{a['samples']}")
    for key in ("min", "median", "max"):
        print(f"{'latency s ' + key:<38}{b['latency_s'][key]:>14}{a['latency_s'][key]:>10}")
    model_lines = [r for r in rows if r["source"] == "model"]
    print(f"model lines that reached the child: {len(model_lines)}/{len(rows)}; "
          f"fallback reasons: {sorted({r['fallback_reason'] for r in rows if r['fallback_reason']})}")


if __name__ == "__main__":
    main()

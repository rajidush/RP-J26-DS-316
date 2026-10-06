"""
Capture INTERCEPT samples from the live model next to the template text, as
evidence for the plan's 5 Oct task (real SLM wired into Intercept, unconstrained)
and as the "before" half of the 6 Oct before/after comparison.

Needs LM Studio running on localhost:1234. Run from component3_socratic_educator/:
    python -m evidence.capture_intercept_samples --runs 2

Writes evidence/intercept_samples_<date>.jsonl (one row per sample) and prints
a summary: grammar matches, category words leaked, latency.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.grammar_decoder import GrammarConstrainedGenerator
from src.schemas import RiskCategory

OUT_DIR = Path(__file__).resolve().parent


def leaked_words(text: str, rc: RiskCategory) -> list[str]:
    return [w for w in rc.value.split("_") if w in text.lower()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2, help="live samples per risk category")
    args = ap.parse_args()

    gen = GrammarConstrainedGenerator()
    rows = []
    for rc in RiskCategory:
        template = gen._candidate("INTERCEPT", rc)
        for run in range(args.runs):
            start = time.time()
            text = gen.generate("INTERCEPT", rc)
            rows.append({
                "risk_category": rc.value,
                "run": run,
                "live_text": text,
                "template_text": template,
                "seconds": round(time.time() - start, 2),
                "fell_back_to_template": text == template,
                "matches_grammar": gen.validate("INTERCEPT", text),
                "leaked_category_words": leaked_words(text, rc),
            })

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = OUT_DIR / f"intercept_samples_{stamp}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    print(f"{n} samples -> {out}")
    print(f"fell back to template : {sum(r['fell_back_to_template'] for r in rows)}/{n}")
    print(f"matches grammar       : {sum(r['matches_grammar'] for r in rows)}/{n}")
    print(f"leaked category words : {sum(bool(r['leaked_category_words']) for r in rows)}/{n}")
    secs = sorted(r["seconds"] for r in rows)
    print(f"latency s (min/median/max): {secs[0]} / {secs[n // 2]} / {secs[-1]}")


if __name__ == "__main__":
    main()

"""
Measurement-only reading level of the lines a child sees (PP1 close-out, 20 Oct).

The project's age range is 8-15, but Component 3's prompts were written for 11+
and it receives no age (PP2: the age band comes from Component 4's persona).
This measures whether the lines children actually get are easy enough for the
youngest band. It reads recorded evidence and the current template pools and
changes nothing.

    INTERCEPT, model    delivered model lines of the 13 Oct adversarial run, and
                        of the 14 Oct online after-review end-to-end run
    INTERCEPT fallback  the fallback pool in src/grammar_decoder.py
    openers             the two fixed safe openers
    INQUIRE templates   INQUIRE_LINES in src/grammar_decoder.py
    CONTRACT            the closing lines recorded in the 14 Oct after-review runs

Three formulas, from textstat:
    Flesch-Kincaid grade  general purpose; built for passages of 100+ words
    Spache grade          built for primary-school text (grades 1-3)
    Dale-Chall score      counts words outside a familiar-word list; 4.9 or
                          lower = easily read by a 4th-grade (about age 9-10) pupil
A one-sentence line is too short for these formulas, so each set is scored as one
passage (all its lines joined); per-line Flesch-Kincaid is shown only to find
the hardest lines. US grade + 5 is roughly the age.

Run from component3_socratic_educator/:
    python -m evidence.readability_check [--tag x]

Writes, in evidence/system/readability_<date>[_<tag>]/, a .jsonl (one row per line),
a .summary.json and a .md; never overwrites.
"""
import argparse
import hashlib
import json
import platform
import statistics
import sys
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import textstat

from src.grammar_decoder import INQUIRE_LINES, INTERCEPT_OPENERS, GrammarConstrainedGenerator

EV = Path(__file__).resolve().parent
ADVERSARIAL = EV / "intercept" / "adversarial_intercept_2026-10-13" / "adversarial_intercept_2026-10-13.jsonl"
END_TO_END = [EV / "system" / f"end_to_end_2026-10-14_{mode}_after-review" / f"end_to_end_2026-10-14_{mode}_after-review.jsonl"
              for mode in ("online", "offline")]
DALE_CHALL_EASY = 4.9   # Dale-Chall: 4.9 or lower is easily read by an average 4th-grade pupil


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect() -> dict[str, list[str]]:
    """Each set's lines, in recorded order (repeats kept: they are what children got)."""
    online = _rows(END_TO_END[0])
    return {
        "INTERCEPT, model (13 Oct adversarial)":
            [r["delivered"] for r in _rows(ADVERSARIAL) if r["source"] == "model"],
        "INTERCEPT, model (14 Oct online end-to-end)":
            [t["system_text"] for s in online for t in s["turns"] if t["state"] == "INTERCEPT" and t["source"] == "model"],
        "INTERCEPT fallback pool": list(GrammarConstrainedGenerator._CANDIDATES["INTERCEPT"]),
        "fixed openers": list(INTERCEPT_OPENERS),
        "INQUIRE templates": [line for pool in INQUIRE_LINES.values() for line in pool],
        "CONTRACT closing lines (14 Oct runs)":
            sorted({t["system_text"] for path in END_TO_END for s in _rows(path) for t in s["turns"]
                    if t["state"] == "CONTRACT"}),
    }


def score(lines: list[str]) -> dict:
    passage = " ".join(lines)
    per_line = sorted(((textstat.flesch_kincaid_grade(line), line) for line in lines), reverse=True)
    grades = [g for g, _ in per_line]
    dale_chall = textstat.dale_chall_readability_score(passage)
    hard = Counter(w.lower() for line in lines for w in textstat.difficult_words_list(line))
    words = textstat.lexicon_count(passage)
    return {
        "lines": len(lines), "unique_lines": len(set(lines)), "words": words,
        "dale_chall_unfamiliar_words": {"share": sum(hard.values()) / words, "most_common": hard.most_common(8)},
        "passage": {"flesch_kincaid_grade": textstat.flesch_kincaid_grade(passage),
                    "spache_grade": textstat.spache_readability(passage),
                    "dale_chall_score": dale_chall,
                    "dale_chall_easy_for_grade_4": dale_chall <= DALE_CHALL_EASY},
        "per_line_flesch_kincaid": {"median": statistics.median(grades),
                                    "p90": grades[int(0.1 * len(grades))] if len(grades) >= 10 else None,
                                    "max": grades[0]},
        "hardest_lines": [{"flesch_kincaid_grade": g, "line": line}
                          for g, line in list(dict.fromkeys(per_line))[:3]],
    }


def to_markdown(summary: dict) -> str:
    out = ["# Reading level of child-facing lines", "",
           f"Measured {summary['run_record']['measured_at']} by `evidence/readability_check.py` "
           f"(textstat {summary['run_record']['textstat_version']}). Measurement only: nothing in the dialogue "
           "changed. Each set is scored as one passage, because a one-sentence line is too short for these "
           "formulas. US grade + 5 is roughly the age, so grade 3 is about age 8.", "",
           "| Set | Lines (unique) | Words | Flesch–Kincaid grade | Spache grade | Dale–Chall score |",
           "|---|---|---|---|---|---|"]
    for name, s in summary["sets"].items():
        p = s["passage"]
        out.append(f"| {name} | {s['lines']} ({s['unique_lines']}) | {s['words']} | {p['flesch_kincaid_grade']:.1f} | "
                   f"{p['spache_grade']:.1f} | {p['dale_chall_score']:.1f}"
                   f"{'' if p['dale_chall_easy_for_grade_4'] else ' (above 4.9)'} |")
    out += ["", f"Dale–Chall: {DALE_CHALL_EASY} or lower is easily read by an average 4th-grade pupil (about age 9–10); "
            "5.0–5.9 is grades 5–6, 6.0–6.9 grades 7–8.", "",
            "**Read Dale–Chall with care.** textstat looks each word up in the familiar-word list as written, so "
            "it counts regular -ing, -ed and -er forms of familiar words (\"doing\", \"noticed\", \"closer\") and "
            "\"okay\" as unfamiliar. The published method counts those as familiar, so these scores are too high. "
            "The words it counted:", ""]
    for name, s in summary["sets"].items():
        u = s["dale_chall_unfamiliar_words"]
        out.append(f"- {name} ({100 * u['share']:.0f}% of words): "
                   + ", ".join(f"{w} ×{n}" for w, n in u["most_common"]))
    out += ["", "## Hardest lines per set (per-line Flesch–Kincaid, for finding outliers only)", ""]
    for name, s in summary["sets"].items():
        out.append(f"**{name}**")
        out += [f"- {h['flesch_kincaid_grade']:.1f}: {h['line']}" for h in s["hardest_lines"]]
        out.append("")
    out += ["## Limitations", "",
            "- The formulas measure word and sentence length and word familiarity, not meaning: a short question "
            "can still be hard to answer, for example a hypothetical one for an 8-year-old.",
            "- Readability formulas were built for US school texts in passages, not single spoken questions.",
            "- Component 3 receives no age today, so this checks the lines against the youngest band only.", "",
            "## Inputs", ""]
    out += [f"- `{name}`: sha256 `{h}`" for name, h in summary["run_record"]["inputs"].items()]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    name = f"readability_{datetime.now():%Y-%m-%d}" + (f"_{args.tag}" if args.tag else "")
    out_dir = EV / "system" / name
    if out_dir.exists():
        print(f"{name} already exists; pass a new --tag", file=sys.stderr)
        return 1

    textstat.set_lang("en_US")
    sets = collect()
    summary = {
        "run_record": {
            "measured_at": datetime.now().isoformat(timespec="seconds"),
            "textstat_version": version("textstat"), "python": platform.python_version(), "language": "en_US",
            "inputs": {str(p.relative_to(EV)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [ADVERSARIAL, *END_TO_END]},
            "template_pools": "src/grammar_decoder.py: INTERCEPT_OPENERS, INQUIRE_LINES, INTERCEPT fallback pool",
        },
        "sets": {set_name: score(lines) for set_name, lines in sets.items()},
    }

    out_dir.mkdir()
    with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for set_name, lines in sets.items():
            for line in lines:
                f.write(json.dumps({"set": set_name, "line": line,
                                    "flesch_kincaid_grade": textstat.flesch_kincaid_grade(line)},
                                   ensure_ascii=False) + "\n")
    (out_dir / f"{name}.summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    (out_dir / f"{name}.md").write_text(to_markdown(summary), encoding="utf-8")
    for set_name, s in summary["sets"].items():
        p = s["passage"]
        print(f"{set_name}: FK {p['flesch_kincaid_grade']:.1f}, Spache {p['spache_grade']:.1f}, "
              f"Dale-Chall {p['dale_chall_score']:.1f} ({s['lines']} lines)")
    print(f"wrote evidence/system/{name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

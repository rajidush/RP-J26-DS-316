"""
Agreement + consistency report for the C2 annotation worksheet (CODEBOOK section 6).

    python -m component2_hate_speech_detection.annotation.agreement
    python -m component2_hate_speech_detection.annotation.agreement --second <other annotator's csv>

Without --second: AI draft vs human final (how much the human changed).
With --second:    human final vs a second human's final, on the overlapping ids.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from component2_hate_speech_detection.annotation.build_labels import ANNOTATION_SPLIT, DEFAULT_OUT_DIR

HARM_LOCUS = ("text", "image", "multimodal", "none")
FRAMING = ("commits", "reports", "quotes", "condemns", "none")
SEVERITY = ("0", "1", "2", "3")

# (worksheet suffix, allowed values, quadratic-weighted?)
DIMENSIONS = {
    "harm_locus": (HARM_LOCUS, False),
    "framing": (FRAMING, False),
    "child_severity": (SEVERITY, True),
}


def cohen_kappa(a: Sequence[str], b: Sequence[str], labels: Sequence[str], quadratic: bool = False) -> float:
    """Cohen's kappa; quadratic weights for ordinal labels (Cohen 1968)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label sequences")
    k = len(labels)
    index = {label: i for i, label in enumerate(labels)}
    n = len(a)
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[index[x]][index[y]] += 1
    row = [sum(observed[i]) for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    def weight(i: int, j: int) -> float:
        if quadratic:
            return ((i - j) ** 2) / ((k - 1) ** 2)
        return 0.0 if i == j else 1.0

    disagree_obs = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k)) / n
    disagree_exp = sum(weight(i, j) * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    if disagree_exp == 0:
        return 1.0  # both raters used one identical label throughout
    return 1.0 - disagree_obs / disagree_exp


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def consistency_errors(rows: list[dict]) -> list[str]:
    errors = []
    for r in rows:
        locus, sev = r.get("final_harm_locus", ""), r.get("final_child_severity", "")
        if locus == "text" and r.get("text_sufficient") == "false":
            errors.append(f"{r['id']}: harm_locus=text but the same caption is benign elsewhere")
        if locus == "none" and r.get("hateful") == "hateful":
            errors.append(f"{r['id']}: harm_locus=none on a gold-hateful meme")
        for dim, (allowed, _) in DIMENSIONS.items():
            value = r.get(f"final_{dim}", "")
            if value and value not in allowed:
                errors.append(f"{r['id']}: final_{dim}={value!r} not in {allowed}")
        if sev and locus and (sev == "0") != (locus == "none"):
            errors.append(f"{r['id']}: child_severity={sev} contradicts harm_locus={locus}")
    return errors


def pairs(rows_a: list[dict], col_a: str, rows_b: list[dict], col_b: str) -> tuple[list[str], list[str]]:
    b_by_id = {r["id"]: r for r in rows_b}
    a_vals, b_vals = [], []
    for r in rows_a:
        other = b_by_id.get(r["id"])
        if other and r.get(col_a) and other.get(col_b):
            a_vals.append(r[col_a])
            b_vals.append(other[col_b])
    return a_vals, b_vals


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", type=Path, default=DEFAULT_OUT_DIR / f"annotation_{ANNOTATION_SPLIT}.csv")
    parser.add_argument("--second", type=Path, help="second annotator's worksheet (same columns)")
    args = parser.parse_args(argv)

    rows = read_rows(args.sheet)
    finals = sum(1 for r in rows if all(r.get(f"final_{d}") for d in DIMENSIONS))
    print(f"rows: {len(rows)}   fully human-labelled: {finals}")

    other = read_rows(args.second) if args.second else rows
    col_a, col_b = ("final_", "final_") if args.second else ("draft_", "final_")
    print(f"\nagreement ({'human vs human' if args.second else 'AI draft vs human final'}):")
    for dim, (labels, quadratic) in DIMENSIONS.items():
        a, b = pairs(rows, col_a + dim, other, col_b + dim)
        if not a:
            print(f"  {dim:15s} n=0")
            continue
        kappa = cohen_kappa(a, b, labels, quadratic=quadratic)
        flag = "" if kappa >= 0.6 else "   <- below 0.6 target, revise codebook"
        print(f"  {dim:15s} n={len(a):4d}  kappa{'(quadratic)' if quadratic else ''}={kappa:.3f}{flag}")

    print("\nlabel distribution (final):")
    for dim in DIMENSIONS:
        print(f"  {dim:15s} {dict(Counter(r[f'final_{dim}'] for r in rows if r.get(f'final_{dim}')))}")

    errors = consistency_errors(rows)
    print(f"\nconsistency errors: {len(errors)}")
    for e in errors[:50]:
        print("  " + e)


if __name__ == "__main__":
    main()

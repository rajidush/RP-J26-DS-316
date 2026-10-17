"""
Load the AI draft pass into the annotation worksheet (CODEBOOK section 6, step 2).

Inputs (in data/labels/drafts/, git-ignored):
  ai_draft_dev_seen.csv          draft: id, ai_hateful, harm_locus, framing, child_severity, blind, note
                                 (blind=true: drafted with the gold label hidden)
  reconciliation_dev_seen.csv    one row per meme where the blind ai_hateful disagreed with gold,
                                 re-checked at full resolution with gold visible:
                                 decision = agree (draft was wrong) / disagree / unsure

Outputs:
  annotation_dev_seen.csv        draft_* columns filled; final_* columns are never touched
  label_audit_dev_seen.csv       memes whose gold label the draft questions (disagree / unsure)
  prints blind agreement with gold (Cohen's kappa) and every consistency fix applied

    python -m component2_hate_speech_detection.annotation.import_drafts
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Optional

from component2_hate_speech_detection.annotation.agreement import cohen_kappa
from component2_hate_speech_detection.annotation.build_labels import (
    ANNOTATION_SPLIT,
    DEFAULT_OUT_DIR,
    WORKSHEET_COLUMNS,
    read_worksheet,
)

DRAFTED_BY = "claude-opus-5-5 (blind draft + reconciliation)"
AUDIT_COLUMNS = [
    "id", "img", "text", "hateful", "label_conflict", "protected_category", "attack_type",
    "draft_gold_check", "ai_blind_hateful", "draft_child_severity", "draft_notes",
]


def read_csv(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def gold_check(blind_hateful: str, gold: str, reconciled: Optional[dict]) -> str:
    """CODEBOOK D9: agree when the blind draft matched gold, else the reconciliation decision."""
    if blind_hateful == gold:
        return "agree"
    if reconciled is None:
        raise ValueError("blind draft disagrees with gold but has no reconciliation row")
    return reconciled["decision"]


def consistent_locus(locus: str, row: dict) -> str:
    """CODEBOOK D6 constraint: a caption that is benign on another meme cannot carry the harm alone."""
    if locus == "text" and row.get("text_sufficient") == "false":
        return "multimodal"
    return locus


def merge(sheet: dict[int, dict], blind: dict[str, dict], recon: dict[str, dict]) -> tuple[list[dict], Counter]:
    report: Counter = Counter()
    rows = []
    for meme_id, row in sorted(sheet.items()):
        key = str(meme_id)
        draft = blind.get(key)
        if draft is None:
            rows.append(row)
            report["no_draft"] += 1
            continue
        reconciled = recon.get(key)
        check = gold_check(draft["ai_hateful"], row["hateful"], reconciled)
        # A reconciled row was re-examined at full resolution, so its values replace the blind ones.
        source = reconciled or draft
        locus = consistent_locus(source["harm_locus"], row)
        if locus != source["harm_locus"]:
            report["locus_text_to_multimodal"] += 1
        note = source["note"] + (" [reconciled]" if reconciled else "")
        row.update({
            "draft_harm_locus": locus,
            "draft_framing": source["framing"],
            "draft_child_severity": source["child_severity"],
            "draft_gold_check": check,
            "draft_notes": note,
            "drafted_by": DRAFTED_BY,
        })
        row["_ai_hateful"] = draft["ai_hateful"]
        row["_blind"] = draft.get("blind", "true") == "true"
        report[f"gold_check:{check}"] += 1
        rows.append(row)
    return rows, report


def blind_kappa(rows: list[dict]) -> tuple[int, float, float]:
    """Agreement of the gold-blind draft with gold; drafts made after seeing gold are excluded."""
    pairs = [(r["_ai_hateful"], r["hateful"]) for r in rows if r.get("_blind")]
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    accuracy = sum(x == y for x, y in pairs) / len(pairs)
    return len(pairs), accuracy, cohen_kappa(a, b, ("hateful", "not_hateful"))


def write_rows(path: Path, rows: list[dict], columns: list[str]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    sheet_path = args.labels_dir / f"annotation_{ANNOTATION_SPLIT}.csv"
    drafts = args.labels_dir / "drafts"
    rows, report = merge(read_worksheet(sheet_path),
                         read_csv(drafts / f"ai_draft_{ANNOTATION_SPLIT}.csv"),
                         read_csv(drafts / f"reconciliation_{ANNOTATION_SPLIT}.csv"))
    write_rows(sheet_path, rows, WORKSHEET_COLUMNS)

    audit = [dict(r, ai_blind_hateful=r["_ai_hateful"]) for r in rows
             if r.get("draft_gold_check") in ("disagree", "unsure")]
    audit.sort(key=lambda r: (r["draft_gold_check"], int(r["id"])))
    audit_path = args.labels_dir / f"label_audit_{ANNOTATION_SPLIT}.csv"
    write_rows(audit_path, audit, AUDIT_COLUMNS)

    n, accuracy, kappa = blind_kappa(rows)
    print(f"blind draft vs gold:  n={n}  accuracy={accuracy:.3f}  kappa={kappa:.3f}")
    for key in sorted(report):
        print(f"{key:28s} {report[key]}")
    print(f"\nworksheet -> {sheet_path}\naudit     -> {audit_path}  ({len(audit)} memes)")


if __name__ == "__main__":
    main()

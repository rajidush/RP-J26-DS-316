"""
Build C2 meme labels from gold annotations (see annotation/CODEBOOK.md).

Inputs (never committed -- dataset licence + public repo):
  memes_dir  Kaggle `parthplc/facebook-hateful-meme-dataset` data folder:
             train.jsonl, dev.jsonl, test.jsonl, img/
  woah_dir   facebookresearch/fine_grained_hateful_memes data/annotations:
             train.json, dev_seen.json, dev_unseen.json (JSON lines)

Outputs (in component2_hate_speech_detection/data/labels/, git-ignored):
  memes_labels.jsonl       gold D1-D3 + derived D4-D5 + severity prior, all splits
  annotation_dev_seen.csv  worksheet for annotated D6-D8 on dev_seen

Run from the repo root:
    python -m component2_hate_speech_detection.annotation.build_labels \
        --memes-dir <kaggle data dir> --woah-dir <woah annotations dir>
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = COMPONENT_ROOT / "data" / "labels"

# Kaggle file -> split name. The Kaggle copy is the phase-1 competition release,
# whose dev.jsonl is the 500-meme dev_seen split; main() verifies this by id
# overlap with the WOAH dev_seen file and warns if it does not hold.
KAGGLE_SPLITS = {"train.jsonl": "train", "dev.jsonl": "dev_seen", "test.jsonl": "test_seen"}
WOAH_FILES = ("train.json", "dev_seen.json", "dev_unseen.json")
ANNOTATION_SPLIT = "dev_seen"

PC_EMPTY = "pc_empty"
ATTACK_EMPTY = "attack_empty"

# CODEBOOK D8 rule prior: a starting point only, the annotator decides.
SEVERITY_3_ATTACKS = {"dehumanizing", "inciting_violence"}
SEVERITY_2_ATTACKS = {"slurs", "contempt", "inferiority", "exclusion", "mocking"}

WORKSHEET_COLUMNS = [
    "id", "img", "text", "hateful", "protected_category", "attack_type",
    "text_sufficient", "severity_rule_prior",
    "draft_harm_locus", "draft_framing", "draft_child_severity", "draft_notes", "drafted_by",
    "final_harm_locus", "final_framing", "final_child_severity", "annotator", "date", "notes",
]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalise_text(text: str) -> str:
    return " ".join(text.lower().split())


def hateful_from_kaggle(record: dict) -> Optional[str]:
    if "label" not in record:
        return None  # test split ships without labels
    return "hateful" if int(record["label"]) == 1 else "not_hateful"


def severity_rule_prior(hateful: Optional[str], attacks: Iterable[str]) -> Optional[int]:
    if hateful != "hateful":
        return None
    attacks = set(attacks)
    if attacks & SEVERITY_3_ATTACKS:
        return 3
    return 2  # any other attack type, or a hateful meme with no attack label


def c2_risk_category(hateful: Optional[str]) -> Optional[str]:
    # Every hateful meme attacks a group by the dataset definition -> schema hate_speech.
    return "hate_speech" if hateful == "hateful" else None


def benign_texts(records: Iterable[dict]) -> set[str]:
    return {normalise_text(r["text"]) for r in records if r.get("hateful") == "not_hateful"}


def text_sufficient(record: dict, benign: set[str]) -> Optional[bool]:
    """CODEBOOK D5: False if the same caption is benign elsewhere, else unknown."""
    if record.get("hateful") != "hateful":
        return None
    return False if normalise_text(record["text"]) in benign else None


def merge_record(kaggle: dict, split: str, woah: Optional[dict]) -> dict:
    hateful = hateful_from_kaggle(kaggle)
    pc = list(woah["gold_pc"]) if woah else None
    attack = list(woah["gold_attack"]) if woah else None
    if woah and hateful is None:
        hateful = woah["gold_hate"][0]
    return {
        "id": int(kaggle["id"]),
        "split": split,
        "img": kaggle["img"],
        "text": kaggle["text"],
        "hateful": hateful,
        "protected_category": pc,
        "attack_type": attack,
    }


def build(kaggle_by_split: dict[str, list[dict]], woah_records: list[dict]) -> tuple[list[dict], dict]:
    woah_by_id = {int(r["id"]): r for r in woah_records}
    report: Counter = Counter()

    records = []
    for split, rows in kaggle_by_split.items():
        for row in rows:
            woah = woah_by_id.get(int(row["id"]))
            rec = merge_record(row, split, woah)
            if woah:
                report["woah_matched"] += 1
                if hateful_from_kaggle(row) not in (None, woah["gold_hate"][0]):
                    report["hate_label_mismatch"] += 1
            records.append(rec)

    benign = benign_texts(records)
    for rec in records:
        rec["c2_risk_category"] = c2_risk_category(rec["hateful"])
        rec["text_sufficient"] = text_sufficient(rec, benign)
        rec["severity_rule_prior"] = severity_rule_prior(rec["hateful"], rec["attack_type"] or [])
        report[f"split:{rec['split']}"] += 1
        if rec["text_sufficient"] is False:
            report["text_sufficient_false"] += 1

    kaggle_dev_ids = {int(r["id"]) for r in kaggle_by_split.get("dev_seen", [])}
    woah_dev_ids = {int(r["id"]) for r in woah_records if r.get("set_name") == "dev_seen"}
    report["dev_seen_id_overlap"] = len(kaggle_dev_ids & woah_dev_ids)
    report["dev_seen_kaggle_size"] = len(kaggle_dev_ids)
    return records, dict(report)


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return "|".join(value)
    return str(value)


def worksheet_rows(records: Iterable[dict], existing: Optional[dict[int, dict]] = None) -> list[dict]:
    """Annotation rows for dev_seen. Keeps any draft/final values already filled in."""
    existing = existing or {}
    rows = []
    for rec in sorted((r for r in records if r["split"] == ANNOTATION_SPLIT), key=lambda r: r["id"]):
        row = {col: "" for col in WORKSHEET_COLUMNS}
        for col in ("id", "img", "text", "hateful", "protected_category", "attack_type",
                    "text_sufficient", "severity_rule_prior"):
            row[col] = _cell(rec.get(col))
        previous = existing.get(rec["id"], {})
        for col in WORKSHEET_COLUMNS:
            if col.startswith(("draft_", "final_")) or col in ("drafted_by", "annotator", "date", "notes"):
                row[col] = previous.get(col, "")
        rows.append(row)
    return rows


def read_worksheet(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {int(r["id"]): r for r in csv.DictReader(f)}


def write_outputs(records: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "memes_labels.jsonl"
    with open(labels_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    sheet_path = out_dir / f"annotation_{ANNOTATION_SPLIT}.csv"
    # Never lose annotation work: rebuilding keeps existing draft_/final_ values.
    rows = worksheet_rows(records, read_worksheet(sheet_path))
    with open(sheet_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WORKSHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return labels_path, sheet_path


def load_inputs(memes_dir: Path, woah_dir: Path) -> tuple[dict[str, list[dict]], list[dict]]:
    kaggle = {split: read_jsonl(memes_dir / name)
              for name, split in KAGGLE_SPLITS.items() if (memes_dir / name).exists()}
    if not kaggle:
        raise FileNotFoundError(f"No train/dev/test .jsonl files found in {memes_dir}")
    woah = [rec for name in WOAH_FILES if (woah_dir / name).exists()
            for rec in read_jsonl(woah_dir / name)]
    if not woah:
        raise FileNotFoundError(f"No WOAH annotation files found in {woah_dir}")
    return kaggle, woah


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--memes-dir", type=Path, required=True)
    parser.add_argument("--woah-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    kaggle, woah = load_inputs(args.memes_dir, args.woah_dir)
    records, report = build(kaggle, woah)
    labels_path, sheet_path = write_outputs(records, args.out_dir)

    for key in sorted(report):
        print(f"{key:28s} {report[key]}")
    if report["dev_seen_id_overlap"] != report["dev_seen_kaggle_size"]:
        print("WARNING: Kaggle dev.jsonl is not exactly WOAH dev_seen -- check KAGGLE_SPLITS.")
    print(f"\nlabels    -> {labels_path}\nworksheet -> {sheet_path}")


if __name__ == "__main__":
    main()

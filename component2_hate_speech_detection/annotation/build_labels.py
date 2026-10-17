"""
Build C2 meme labels from gold annotations (see annotation/CODEBOOK.md).

Inputs (never committed -- dataset licence + public repo):
  memes_dir  Kaggle `parthplc/facebook-hateful-meme-dataset` data folder:
             train.jsonl, dev.jsonl, test.jsonl, img/
  woah_dir   facebookresearch/fine_grained_hateful_memes data/annotations:
             train.json, dev_seen.json, dev_unseen.json (JSON lines)

Outputs (in component2_hate_speech_detection/data/labels/, git-ignored):
  memes_labels.jsonl/.csv  gold D1-D3 + derived D4-D5 + severity prior, all splits
  annotation_dev_seen.csv  worksheet for annotated D6-D9 on dev_seen
  finetune/*.jsonl/.csv    train / val / test files for model fine-tuning

Run from the repo root:
    python -m component2_hate_speech_detection.annotation.build_labels \
        --memes-dir <kaggle data dir> --woah-dir <woah annotations dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import random
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

# Fine-tuning split: a validation set carved out of train, so dev_seen stays an
# untouched test set (test_seen labels were never released).
#
# Grouped by caption: memes sharing a caption never straddle train and val. A
# plain random split put 30 % of val captions in train, so a model could score
# on val by remembering captions (leakage; Kapoor & Narayanan, 2023).
# Confounder-matched: dev_seen was built with benign confounders, and 31 % of its
# memes sit in caption groups holding *both* labels; a random val had 2 %, so it
# rewarded caption memorisation that dev_seen punishes (measured 2026-10-11:
# text-only probe val AUROC 0.80 vs dev_seen 0.64). Val therefore takes whole
# both-label groups up to FINETUNE_FLIP_SHARE, then single-label groups up to the
# train hateful share.
FINETUNE_VAL_SIZE = 500
FINETUNE_SEED = 316
FINETUNE_FLIP_SHARE = 0.30

LABEL_COLUMNS = [
    "id", "split", "img", "text", "hateful", "hateful_kaggle", "label_source", "label_conflict",
    "protected_category", "attack_type", "c2_risk_category", "text_sufficient", "severity_rule_prior",
]
FINETUNE_COLUMNS = ["id", "img", "text", "label", "protected_category", "attack_type", "label_conflict"]

WORKSHEET_COLUMNS = [
    "id", "img", "text", "hateful", "label_conflict", "protected_category", "attack_type",
    "text_sufficient", "severity_rule_prior",
    "draft_harm_locus", "draft_framing", "draft_child_severity", "draft_gold_check", "draft_notes",
    "drafted_by",
    "final_harm_locus", "final_framing", "final_child_severity", "final_gold_check",
    "annotator", "date", "notes",
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
    """CODEBOOK D1: the WOAH 2021 label wins over the Kaggle (phase-1) label.

    WOAH is the later release and its binary label is the one consistent with its own
    target/attack labels; taking the Kaggle label would pair "not_hateful" with
    "dehumanizing". The Kaggle label is kept for provenance and conflicts are flagged.
    """
    kaggle_label = hateful_from_kaggle(kaggle)
    woah_label = woah["gold_hate"][0] if woah else None
    hateful = woah_label or kaggle_label
    return {
        "id": int(kaggle["id"]),
        "split": split,
        "img": kaggle["img"],
        "text": kaggle["text"],
        "hateful": hateful,
        "hateful_kaggle": kaggle_label,
        "label_source": "woah2021" if woah_label else ("kiela2020" if kaggle_label else None),
        "label_conflict": None not in (kaggle_label, woah_label) and kaggle_label != woah_label,
        "protected_category": list(woah["gold_pc"]) if woah else None,
        "attack_type": list(woah["gold_attack"]) if woah else None,
    }


def fine_grained_missing(record: dict) -> bool:
    """A hateful meme whose WOAH target or attack label is empty (WOAH is incomplete there)."""
    if record["hateful"] != "hateful" or record["protected_category"] is None:
        return False
    return PC_EMPTY in record["protected_category"] or ATTACK_EMPTY in record["attack_type"]


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
            if rec["label_conflict"]:
                report[f"label_conflict:{split}"] += 1
            if fine_grained_missing(rec):
                report[f"fine_grained_missing:{split}"] += 1
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
        for col in ("id", "img", "text", "hateful", "label_conflict", "protected_category",
                    "attack_type", "text_sufficient", "severity_rule_prior"):
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


def caption_groups(rows: list[dict]) -> list[list[dict]]:
    """Memes grouped by normalised caption, in a stable order (first id)."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(normalise_text(r["text"]), []).append(r)
    return sorted(groups.values(), key=lambda g: g[0]["id"])


def pick_val_ids(train: list[dict], size: int = FINETUNE_VAL_SIZE, flip_share: float = FINETUNE_FLIP_SHARE,
                 seed: int = FINETUNE_SEED) -> set[int]:
    """Whole caption groups: both-label groups first, then single-label groups by label."""
    rng = random.Random(seed)
    groups = caption_groups(train)
    flip = [g for g in groups if len({r["hateful"] for r in g}) > 1]
    plain = {label: [g for g in groups if len({r["hateful"] for r in g}) == 1 and g[0]["hateful"] == label]
             for label in ("hateful", "not_hateful")}
    rng.shuffle(flip)
    for pool in plain.values():
        rng.shuffle(pool)

    chosen: list[dict] = []

    def fill(pool: list[list[dict]], room: int) -> None:
        taken = 0
        for g in pool:
            if taken + len(g) <= room:  # skip groups that would overshoot; singletons close the gap
                chosen.extend(g)
                taken += len(g)
            if taken == room:
                return

    fill(flip, round(size * flip_share))
    hateful_share = sum(r["hateful"] == "hateful" for r in train) / len(train) if train else 0.0
    hateful_room = max(0, round(size * hateful_share) - sum(r["hateful"] == "hateful" for r in chosen))
    fill(plain["hateful"], min(hateful_room, size - len(chosen)))
    fill(plain["not_hateful"], size - len(chosen))
    return {r["id"] for r in chosen}


def finetune_splits(records: list[dict]) -> dict[str, list[dict]]:
    """train/val from train (val grouped by caption and confounder-matched, fixed seed), dev_seen as test."""
    def row(rec: dict) -> dict:
        return {
            "id": rec["id"], "img": rec["img"], "text": rec["text"],
            "label": None if rec["hateful"] is None else int(rec["hateful"] == "hateful"),
            "protected_category": rec["protected_category"], "attack_type": rec["attack_type"],
            "label_conflict": rec["label_conflict"],
        }

    train = sorted((r for r in records if r["split"] == "train"), key=lambda r: r["id"])
    val_ids = pick_val_ids([r for r in train if r["hateful"]])

    def split_rows(name: str) -> list[dict]:
        return [row(r) for r in sorted(records, key=lambda r: r["id"]) if r["split"] == name]

    return {
        "train": [row(r) for r in train if r["id"] not in val_ids],
        "val": [row(r) for r in train if r["id"] in val_ids],
        "test": split_rows(ANNOTATION_SPLIT),
        "unlabelled_test_seen": split_rows("test_seen"),
    }


def write_jsonl_and_csv(rows: list[dict], jsonl_path: Path, columns: list[str]) -> None:
    """Same rows twice: JSONL for code, CSV (same name) for spreadsheets.

    CSV cells follow the worksheet convention: lists joined with "|", booleans as
    true/false, missing values empty. utf-8-sig so Excel shows accented captions.
    An empty split still gets a header-only CSV, so no stale file is left behind.
    """
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(jsonl_path.with_suffix(".csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: _cell(r.get(k)) for k in columns} for r in rows)


def write_outputs(records: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "memes_labels.jsonl"
    write_jsonl_and_csv(records, labels_path, LABEL_COLUMNS)

    sheet_path = out_dir / f"annotation_{ANNOTATION_SPLIT}.csv"
    # Never lose annotation work: rebuilding keeps existing draft_/final_ values.
    rows = worksheet_rows(records, read_worksheet(sheet_path))
    with open(sheet_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WORKSHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    finetune_dir = out_dir / "finetune"
    finetune_dir.mkdir(exist_ok=True)
    for name, split_rows in finetune_splits(records).items():
        write_jsonl_and_csv(split_rows, finetune_dir / f"{name}.jsonl", FINETUNE_COLUMNS)
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

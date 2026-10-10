"""Annotation pipeline tests. Synthetic records only -- the real dataset is never needed (or committed)."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from component2_hate_speech_detection.annotation.agreement import cohen_kappa, consistency_errors
from component2_hate_speech_detection.annotation.build_labels import (
    build,
    read_worksheet,
    severity_rule_prior,
    write_outputs,
)


def _kaggle(id_, text, label=None):
    rec = {"id": id_, "img": f"img/{id_:05d}.png", "text": text}
    if label is not None:
        rec["label"] = label
    return rec


def _woah(id_, set_name, hate, pc, attack):
    return {"id": id_, "set_name": set_name, "gold_hate": [hate], "gold_pc": pc, "gold_attack": attack}


@pytest.fixture
def built():
    kaggle = {
        "train": [_kaggle(1, "Some Caption", 1), _kaggle(2, "some   caption", 0)],
        "dev_seen": [_kaggle(3, "another caption", 1), _kaggle(4, "benign caption", 0)],
        "test_seen": [_kaggle(5, "unlabelled caption")],
    }
    woah = [
        _woah(1, "train", "hateful", ["race"], ["dehumanizing"]),
        _woah(2, "train", "not_hateful", ["pc_empty"], ["attack_empty"]),
        _woah(3, "dev_seen", "hateful", ["religion", "sex"], ["mocking", "contempt"]),
        _woah(4, "dev_seen", "not_hateful", ["pc_empty"], ["attack_empty"]),
    ]
    records, report = build(kaggle, woah)
    return {r["id"]: r for r in records}, report


def test_gold_labels_are_carried_unchanged(built):
    recs, report = built
    assert recs[3]["protected_category"] == ["religion", "sex"]
    assert recs[3]["attack_type"] == ["mocking", "contempt"]
    assert report["woah_matched"] == 4
    assert report.get("hate_label_mismatch", 0) == 0


def test_unlabelled_test_split_stays_unlabelled(built):
    recs, _ = built
    assert recs[5]["hateful"] is None
    assert recs[5]["c2_risk_category"] is None
    assert recs[5]["protected_category"] is None


def test_c2_category_maps_onto_schema_enum(built):
    recs, _ = built
    schema = json.loads((ROOT / "docs/interface-contracts/comp2_to_comp3.schema.json").read_text())
    allowed = set(schema["properties"]["risk_category"]["enum"])
    assert recs[1]["c2_risk_category"] in allowed
    assert recs[2]["c2_risk_category"] is None


def test_benign_text_confounder_marks_text_insufficient(built):
    recs, report = built
    # Meme 1's caption is benign on meme 2 (case/whitespace-insensitive) -> text alone is not hateful.
    assert recs[1]["text_sufficient"] is False
    # No benign twin -> unknown, never True by rule.
    assert recs[3]["text_sufficient"] is None
    assert report["text_sufficient_false"] == 1


@pytest.mark.parametrize("hateful,attacks,expected", [
    ("hateful", ["dehumanizing"], 3),
    ("hateful", ["mocking", "inciting_violence"], 3),
    ("hateful", ["slurs"], 2),
    ("hateful", ["attack_empty"], 2),
    ("not_hateful", ["attack_empty"], None),
])
def test_severity_rule_prior(hateful, attacks, expected):
    assert severity_rule_prior(hateful, attacks) == expected


def test_worksheet_rebuild_keeps_annotation_work(built, tmp_path):
    recs, _ = built
    _, sheet = write_outputs(list(recs.values()), tmp_path)
    rows = read_worksheet(sheet)
    assert set(rows) == {3, 4}  # dev_seen only
    rows[3]["final_framing"] = "commits"
    rows[3]["annotator"] = "IT23209152"
    with open(sheet, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[3]))
        writer.writeheader()
        writer.writerows(rows.values())

    write_outputs(list(recs.values()), tmp_path)
    again = read_worksheet(sheet)
    assert again[3]["final_framing"] == "commits"
    assert again[3]["annotator"] == "IT23209152"


def test_kappa_perfect_chance_and_quadratic():
    labels = ("a", "b")
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"], labels) == pytest.approx(1.0)
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"], labels) == pytest.approx(0.0)
    sev = ("0", "1", "2", "3")
    near = cohen_kappa(["0", "1", "2", "3"], ["0", "1", "2", "2"], sev, quadratic=True)
    far = cohen_kappa(["0", "1", "2", "3"], ["0", "1", "2", "0"], sev, quadratic=True)
    assert near > far  # an off-by-one severity disagreement costs less than an off-by-three


def test_consistency_checks_catch_codebook_violations():
    rows = [
        {"id": "1", "hateful": "hateful", "text_sufficient": "false",
         "final_harm_locus": "text", "final_framing": "commits", "final_child_severity": "2"},
        {"id": "2", "hateful": "hateful", "text_sufficient": "",
         "final_harm_locus": "none", "final_framing": "none", "final_child_severity": "0"},
        {"id": "3", "hateful": "not_hateful", "text_sufficient": "",
         "final_harm_locus": "none", "final_framing": "none", "final_child_severity": "1"},
        {"id": "4", "hateful": "hateful", "text_sufficient": "",
         "final_harm_locus": "multimodal", "final_framing": "endorses", "final_child_severity": "3"},
        {"id": "5", "hateful": "hateful", "text_sufficient": "false",
         "final_harm_locus": "multimodal", "final_framing": "commits", "final_child_severity": "3"},
    ]
    errors = "\n".join(consistency_errors(rows))
    assert "1: harm_locus=text" in errors
    assert "2: harm_locus=none on a gold-hateful" in errors
    assert "3: child_severity=1 contradicts" in errors
    assert "4: final_framing='endorses'" in errors
    assert "5:" not in errors


def test_dataset_files_are_git_ignored():
    for path in ("component2_hate_speech_detection/data/labels/memes_labels.jsonl",
                 "component2_hate_speech_detection/data/woah/train.json"):
        result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
        assert result.returncode == 0, f"{path} must be git-ignored (public repo + dataset licence)"

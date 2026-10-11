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
    FINETUNE_COLUMNS,
    LABEL_COLUMNS,
    build,
    finetune_splits,
    read_worksheet,
    severity_rule_prior,
    write_outputs,
)
from component2_hate_speech_detection.annotation.import_drafts import blind_kappa, merge
from component2_hate_speech_detection.annotation.review import Store, blind_sample, validate, write_sheet


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
    assert not any(r["label_conflict"] for r in recs.values())


def test_woah_label_wins_and_conflict_is_flagged():
    kaggle = {"train": [_kaggle(7, "relax you are criminals", 0)]}
    woah = [_woah(7, "train", "hateful", ["nationality"], ["dehumanizing"])]
    records, report = build(kaggle, woah)
    rec = records[0]
    # The binary label must agree with the fine-grained labels it is shipped with.
    assert rec["hateful"] == "hateful" and rec["attack_type"] == ["dehumanizing"]
    assert rec["hateful_kaggle"] == "not_hateful" and rec["label_source"] == "woah2021"
    assert rec["label_conflict"] is True and report["label_conflict:train"] == 1


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


def test_finetune_val_is_stratified_and_disjoint_from_train():
    kaggle = {
        "train": [_kaggle(i, f"caption {i}", int(i % 4 == 0)) for i in range(1, 2001)],
        "dev_seen": [_kaggle(5001, "dev caption", 1)],
        "test_seen": [_kaggle(6001, "test caption")],
    }
    records, _ = build(kaggle, [])
    splits = finetune_splits(records)
    train_ids = {r["id"] for r in splits["train"]}
    val = splits["val"]
    assert len(val) == 500 and not train_ids & {r["id"] for r in val}
    assert len(train_ids) + len(val) == 2000
    assert sum(r["label"] for r in val) == 125  # same 25 % hateful share as train
    assert [r["id"] for r in splits["test"]] == [5001]
    assert splits["unlabelled_test_seen"][0]["label"] is None
    assert finetune_splits(records)["val"] == val  # fixed seed -> reproducible


def test_finetune_val_keeps_caption_groups_whole_and_matches_confounders():
    # 300 benign-confounder pairs (same caption, opposite labels) + 1400 unique captions.
    train = []
    for i in range(300):
        train += [_kaggle(2 * i + 1, f"pair {i}", 1), _kaggle(2 * i + 2, f"pair {i}", 0)]
    train += [_kaggle(1000 + i, f"solo {i}", int(i % 3 == 0)) for i in range(1400)]
    records, _ = build({"train": train}, [])
    splits = finetune_splits(records)
    val_captions = {r["text"] for r in splits["val"]}
    assert len(splits["val"]) == 500
    assert not val_captions & {r["text"] for r in splits["train"]}  # no caption leaks across
    pairs_in_val = sum(r["text"].startswith("pair") for r in splits["val"])
    assert pairs_in_val == 150  # FINETUNE_FLIP_SHARE of val, as in dev_seen


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


def _sheet_row(id_, hateful, text_sufficient="", **final):
    row = {"id": str(id_), "img": f"img/{id_:05d}.png", "text": "t", "hateful": hateful,
           "text_sufficient": text_sufficient}
    row.update(final)
    return row


def _draft(id_, ai_hateful, locus="multimodal", framing="commits", severity="2", blind="true"):
    return {"id": str(id_), "ai_hateful": ai_hateful, "harm_locus": locus, "framing": framing,
            "child_severity": severity, "blind": blind, "note": "n"}


def test_import_drafts_sets_gold_check_and_keeps_human_work():
    sheet = {
        1: _sheet_row(1, "hateful", final_framing="quotes", annotator="IT23209152"),
        2: _sheet_row(2, "hateful"),
        3: _sheet_row(3, "not_hateful"),
    }
    blind = {"1": _draft(1, "hateful"), "2": _draft(2, "not_hateful", "none", "none", "0"),
             "3": _draft(3, "hateful")}
    recon = {"2": dict(_draft(2, "hateful", severity="3"), decision="agree"),
             "3": dict(_draft(3, "hateful"), decision="disagree")}
    rows, report = merge(sheet, blind, recon)
    by_id = {r["id"]: r for r in rows}
    assert by_id["1"]["draft_gold_check"] == "agree"
    assert by_id["2"]["draft_gold_check"] == "agree"
    assert by_id["2"]["draft_child_severity"] == "3"  # reconciled values replace the blind ones
    assert by_id["3"]["draft_gold_check"] == "disagree"
    # The import only ever writes draft_* columns; human final_* work survives.
    assert by_id["1"]["final_framing"] == "quotes" and by_id["1"]["annotator"] == "IT23209152"
    assert report["gold_check:agree"] == 2 and report["gold_check:disagree"] == 1


def test_import_drafts_requires_reconciliation_for_every_disagreement():
    with pytest.raises(ValueError):
        merge({1: _sheet_row(1, "hateful")}, {"1": _draft(1, "not_hateful", "none", "none", "0")}, {})


def test_text_locus_is_impossible_when_the_caption_is_benign_elsewhere():
    rows, report = merge({1: _sheet_row(1, "hateful", text_sufficient="false")},
                         {"1": _draft(1, "hateful", locus="text")}, {})
    assert rows[0]["draft_harm_locus"] == "multimodal"
    assert report["locus_text_to_multimodal"] == 1


def test_blind_kappa_ignores_drafts_made_after_seeing_gold():
    sheet = {i: _sheet_row(i, "hateful" if i % 2 else "not_hateful") for i in range(1, 9)}
    blind = {str(i): _draft(i, sheet[i]["hateful"]) for i in range(1, 9)}
    blind["8"] = _draft(8, "hateful", blind="false")  # wrong, but not blind -> excluded
    recon = {"8": dict(_draft(8, "hateful"), decision="unsure")}
    rows, _ = merge(sheet, blind, recon)
    n, accuracy, kappa = blind_kappa(rows)
    assert n == 7 and accuracy == 1.0 and kappa == pytest.approx(1.0)


def _final(**over):
    update = {"id": "1", "final_harm_locus": "multimodal", "final_framing": "commits",
              "final_child_severity": "2", "final_gold_check": "agree"}
    update.update(over)
    return update


def test_review_validation_follows_the_codebook():
    assert validate(_final()) == []
    assert any("severity 0" in e for e in validate(_final(final_harm_locus="none")))
    assert any("severity 0" in e for e in validate(_final(final_child_severity="0")))
    assert any("gold_check" in e for e in validate(_final(final_gold_check="")))
    assert any("framing" in e for e in validate(_final(final_framing="endorses")))


def test_blind_sample_is_balanced_reproducible_and_strips_the_draft():
    rows = [{"id": str(i), "hateful": "hateful" if i % 3 else "not_hateful",
             "draft_framing": "commits", "final_framing": "quotes", "annotator": "x"} for i in range(1, 301)]
    sample = blind_sample(rows, size=40, seed=1)
    assert len(sample) == 40 and sum(r["hateful"] == "hateful" for r in sample) == 20
    assert all(r["draft_framing"] == "" and r["final_framing"] == "" and r["annotator"] == "" for r in sample)
    assert [r["id"] for r in blind_sample(rows, size=40, seed=1)] == [r["id"] for r in sample]


def test_review_store_saves_finals_and_hides_draft_in_blind_mode(tmp_path):
    sheet = tmp_path / "sheet.csv"
    write_sheet(sheet, [_sheet_row(1, "hateful", draft_framing="commits", drafted_by="ai")])
    blind = Store(sheet, "IT23209152", blind=True)
    assert not any(k.startswith("draft_") or k == "drafted_by" for k in blind.public_rows()[0])

    store = Store(sheet, "IT23209152", blind=False)
    assert store.save(_final(id="2")) == ["unknown meme id"]
    assert store.save(_final(notes="checked")) is None
    saved = read_worksheet(sheet)[1]
    assert saved["final_framing"] == "commits" and saved["annotator"] == "IT23209152"
    assert saved["draft_framing"] == "commits"  # the draft is kept next to the human label


def test_every_jsonl_output_has_a_matching_csv(built, tmp_path):
    recs, _ = built
    write_outputs(list(recs.values()), tmp_path)
    for jsonl in [tmp_path / "memes_labels.jsonl", *sorted((tmp_path / "finetune").glob("*.jsonl"))]:
        rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
        with open(jsonl.with_suffix(".csv"), encoding="utf-8-sig", newline="") as f:
            table = list(csv.DictReader(f))
        assert [r["id"] for r in table] == [str(r["id"]) for r in rows], jsonl.name
    with open(tmp_path / "memes_labels.csv", encoding="utf-8-sig", newline="") as f:
        meme3 = next(r for r in csv.DictReader(f) if r["id"] == "3")
    assert meme3["protected_category"] == "religion|sex"  # lists joined like the worksheet
    assert meme3["label_conflict"] == "false" and meme3["text_sufficient"] == ""


def test_csv_columns_cover_every_jsonl_field(built):
    recs, _ = built
    assert set(LABEL_COLUMNS) == set(recs[1])
    split_row = finetune_splits(list(recs.values()))["test"][0]
    assert set(FINETUNE_COLUMNS) == set(split_row)

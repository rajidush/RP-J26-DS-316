"""
Tests for the dataset-validation sheets (ticket pp2-validation/10): `dataset-sheets`
and `dataset-score` in inquire_finetune.py, driven end to end through main(argv)
on a PP2 dataset that `build-pp2` writes in tmp_path. No model is loaded.
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inquire_finetune as ft  # noqa: E402
from src.agreement import gwet_ac1  # noqa: E402
from test_pp2_build import BANDS, MOVES, _rows, _write_jsonl, draw, load_jsonl  # noqa: E402

ITEMS = ("reply_type_correct", "realistic_for_age", "safety_gate", "performs_move")
SHOWN = ("sensitive", "age_band", "context", "educator_before", "child_reply", "reply_type", "move", "educator_line")
SENSITIVE = ("explicit_visual", "grooming_language", "self_harm_imagery", "self_harm_language")


@pytest.fixture
def dataset(tmp_path):
    """The PP2 dataset build-pp2 writes: 144 training rows (16 per band x move), 72 validation rows."""
    plan_path, plan = draw(tmp_path)
    drafts = _write_jsonl(tmp_path / "drafts.jsonl", _rows(plan))
    out = tmp_path / "dataset"
    ft.main(["build-pp2", "--draw", str(plan_path), "--drafts", str(drafts), "--out", str(out)])
    return out


def read_sheet(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_sheet(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def make_sheets(dataset, *extra, out="run"):
    run = dataset.parent / out
    ft.main(["dataset-sheets", "--dataset", str(dataset), "--out", str(run), *extra])
    return run, json.loads((run / "key.json").read_text(encoding="utf-8"))


def rows_of(dataset):
    return {r["example_id"]: (split, r) for split in ("train", "validation")
            for r in load_jsonl(dataset / f"{split}.jsonl")}


# ---------------------------------------------------------------- dataset-sheets
def test_sheet_has_every_validation_row_and_a_stratified_30_percent_of_training_rows(dataset):
    run, key = make_sheets(dataset)
    rows = read_sheet(run / "classmate_dataset_validation.csv")
    assert list(rows[0]) == ["item", *SHOWN, *ITEMS, "notes"]
    assert all(r[c] == "" for r in rows for c in (*ITEMS, "notes"))
    assert [r["item"] for r in rows] == [f"D{n:04d}" for n in range(1, len(rows) + 1)]
    data = rows_of(dataset)
    shown = Counter((data[key["items"][r["item"]]["example_id"]][0]) for r in rows)
    assert shown == {"validation": 72, "train": 45}
    # 30% of each band x move stratum of training rows, rounded half up: 16 -> 4.8 -> 5
    train = Counter((key["items"][r["item"]]["age_band"], key["items"][r["item"]]["move"]) for r in rows
                    if key["items"][r["item"]]["split"] == "train")
    assert train == {(band, move): 5 for band in BANDS for move in MOVES}
    assert len({key["items"][r["item"]]["example_id"] for r in rows}) == len(rows)
    assert key["train_sample"]["share"] == 0.3
    assert key["train_sample"]["per_stratum"]["8-10 / clarify"] == {"rows": 16, "drawn": 5}


def test_each_row_shows_the_dialogue_the_labels_and_the_educator_line(dataset):
    run, key = make_sheets(dataset)
    data = rows_of(dataset)
    for row in read_sheet(run / "classmate_dataset_validation.csv"):
        split, source = data[key["items"][row["item"]]["example_id"]]
        assert row["age_band"] == source["age_band"]
        assert row["context"] == source["context"]
        assert row["educator_before"] == source["previous_message"]
        assert row["child_reply"] == source["child_reply"]
        assert row["reply_type"] == source["reply_type"]
        assert row["move"] == source["move"] + (" (easier form)" if source["easier"] else "")
        assert row["educator_line"] == source["completion"][0]["content"]
        assert row["sensitive"] == ("yes" if source["risk_category"] in SENSITIVE else "")


def test_sheets_have_no_identifying_columns_or_text(dataset):
    run, _ = make_sheets(dataset)
    for name in ("classmate_dataset_validation.csv", "counsellor_dataset_validation.csv"):
        text = (run / name).read_text(encoding="utf-8-sig")
        assert not re.search(r"P2[RS]\d", text), name   # example and scenario IDs
        for row in read_sheet(run / name):
            assert not {"train", "validation", "reviewed"} & set(row.values()), (name, row["item"])


def test_a_recorded_seed_reproduces_the_order_and_the_draw(dataset):
    a, _ = make_sheets(dataset, out="a")
    b, _ = make_sheets(dataset, out="b")
    c, key_c = make_sheets(dataset, "--seed", "7", out="c")
    for name in ("classmate_dataset_validation.csv", "counsellor_dataset_validation.csv", "key.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
        assert (a / name).read_bytes() != (c / name).read_bytes(), name
    assert json.loads((a / "key.json").read_text(encoding="utf-8"))["seed"] == ft.DATASET_SHEET_SEED
    assert key_c["seed"] == 7


def test_counsellor_copy_is_a_stratified_40_percent_of_the_sheet(dataset):
    run, key = make_sheets(dataset)
    counsellor = read_sheet(run / "counsellor_dataset_validation.csv")
    classmate = {r["item"]: r for r in read_sheet(run / "classmate_dataset_validation.csv")}
    assert all(classmate[r["item"]] == r for r in counsellor)
    assert [r["item"] for r in counsellor] != sorted(r["item"] for r in counsellor)
    # 40% of each split x band x move stratum, rounded half up: validation 8 -> 3.2 -> 3, training 5 -> 2
    drawn = Counter(tuple(key["items"][r["item"]][k] for k in ("split", "age_band", "move")) for r in counsellor)
    assert drawn == {**{("validation", b, m): 3 for b in BANDS for m in MOVES},
                     **{("train", b, m): 2 for b in BANDS for m in MOVES}}
    assert key["overlap"]["per_stratum"]["validation / 14-15 / alternatives"] == {"rows": 8, "counsellor": 3}


def test_instructions_ask_the_four_questions_with_the_gate_items_moves_and_reply_types(dataset):
    run, _ = make_sheets(dataset)
    text = (run / "instructions.md").read_text(encoding="utf-8")
    for item in ITEMS:
        assert f"`{item}`" in text
    for gate in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"):
        assert f"`{gate}`" in text
    for value in ("clarify", "consequences", "alternatives", "easier form", "stated_emotion", "dont_know",
                  "off_topic", "own_safer_plan", "explanation", "`yes`", "`no`", "`pass`", "`skip`", "sensitive"):
        assert value in text, value


def test_sheets_refuse_a_dataset_that_is_not_pp2_or_changed_since_it_was_built(dataset, tmp_path):
    train = dataset / "train.jsonl"
    train.write_bytes(train.read_bytes().replace(b"What could you try", b"What would you try", 1))
    with pytest.raises(SystemExit) as e:
        make_sheets(dataset)
    assert "dataset-sheets refused" in str(e.value.code) and "train.jsonl" in str(e.value.code)
    other = tmp_path / "other"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"dataset": "inquire-finetune"}), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        make_sheets(other)
    assert "not a PP2 dataset" in str(e.value.code)


def test_sheets_never_overwrite_a_recorded_run(dataset):
    run, _ = make_sheets(dataset)
    before = (run / "key.json").read_bytes()
    with pytest.raises(SystemExit) as e:
        make_sheets(dataset, "--seed", "7")
    assert "already holds sheets" in str(e.value.code)
    assert (run / "key.json").read_bytes() == before


# ---------------------------------------------------------------- dataset-score
def passing(item, column):
    return "pass" if column == "safety_gate" else "yes"


def fill(run, key, name, value):
    """The rater's returned copy of the blank sheet, each rated cell set to value(item, column); the blank is kept."""
    rows = read_sheet(run / key["copies"][name]["sheet"])
    for row in rows:
        for column in ITEMS:
            row[column] = value(row["item"], column)
    returned = run.parent / "returned"
    returned.mkdir(exist_ok=True)
    return write_sheet(returned / f"{name}.csv", rows), rows


def score(run, rated, out="scores"):
    argv = ["dataset-score", "--key", str(run / "key.json"), "--out", str(run.parent / out)]
    for name, path in rated.items():
        argv += ["--rated", f"{name}={path}"]
    ft.main(argv)
    return json.loads((run.parent / out / "summary.json").read_text(encoding="utf-8"))


def refusal(run, rated):
    with pytest.raises(SystemExit) as e:
        score(run, rated)
    assert not (run.parent / "scores").exists()
    message = str(e.value.code)
    assert message.startswith("dataset-score refused")
    return message


def test_score_refuses_invalid_and_empty_cells_and_names_them(dataset):
    run, key = make_sheets(dataset)
    path, rows = fill(run, key, "classmate", passing)
    rows[1]["realistic_for_age"] = "maybe"
    rows[2]["safety_gate"] = "S9"
    rows[3]["safety_gate"] = "pass S3"     # pass or failed items, never both
    rows[4]["performs_move"] = ""          # forgotten, not skipped
    rows[5]["reply_type_correct"] = " Yes "
    rows[6]["safety_gate"] = "s2, S3"      # typed loosely: accepted
    rows[7]["realistic_for_age"] = "NO"
    write_sheet(path, rows)
    message = refusal(run, {"classmate": path})
    assert f"{rows[1]['item']}: realistic_for_age 'maybe'" in message
    assert f"{rows[2]['item']}: safety_gate 'S9'" in message
    assert f"{rows[3]['item']}: safety_gate 'pass S3'" in message
    assert f"{rows[4]['item']}: performs_move ''" in message
    for ok in (5, 6, 7):
        assert rows[ok]["item"] not in message
    assert "skip" in message


def test_score_refuses_missing_duplicated_unknown_and_edited_rows_and_names_them(dataset):
    run, key = make_sheets(dataset)
    path, rows = fill(run, key, "classmate", passing)
    missing, duplicated, rewritten, relabelled = (rows[n]["item"] for n in (1, 4, 3, 8))
    rows[3]["educator_line"] = "a line the rater rewrote"
    rows[8]["reply_type"] = "refusal"
    rows = [r for r in rows if r["item"] != missing]
    rows.append(dict(rows[5], item=duplicated))
    rows.append(dict(rows[0], item="D9999"))
    write_sheet(path, rows)
    message = refusal(run, {"classmate": path})
    assert f"missing: {missing}" in message
    assert f"duplicated: {duplicated}" in message
    assert "not on the blank sheet: D9999" in message
    assert f"{rewritten}: educator_line differs from the blank sheet" in message
    assert f"{relabelled}: reply_type differs from the blank sheet" in message


def test_score_refuses_a_blank_sheet_changed_since_it_was_written(dataset):
    run, key = make_sheets(dataset)
    blank = run / key["copies"]["classmate"]["sheet"]
    rows = read_sheet(blank)
    for row in rows:
        row.update({c: passing(None, c) for c in ITEMS})
    write_sheet(blank, rows)
    message = refusal(run, {"classmate": blank})
    assert "classmate_dataset_validation.csv" in message and "changed since" in message


# The counsellor's 45 shared items, rated by both raters (all yes / pass unless listed):
#   realistic_for_age: counsellor no on shared 0, 1, 3; classmate no on shared 2, 3.
#     45 pairs, 42 agree: Pa = 42/45. "no" is 5 of the 90 ratings, so pi_no = 1/18, pi_yes = 17/18.
#     Pe = (pi_yes(1 - pi_yes) + pi_no(1 - pi_no)) / (2 - 1) = 2 * 17/324 = 17/162.
#     AC1 = (Pa - Pe) / (1 - Pe) = (6804 - 765) / (45 * 145) = 6039/6525 = 0.9255.
#   safety_gate: counsellor "S2 S3", classmate "S3" on shared 4 (both fail: agree);
#     classmate skips shared 5. 44 compared, all agree: AC1 = 1.0.
#   reply_type_correct: all yes. Pa = 1, Pe = 0: AC1 = 1.0, where kappa would be undefined.
#   performs_move: the classmate says no on one item the counsellor never saw.
@pytest.fixture
def rated(dataset):
    run, key = make_sheets(dataset)
    shared = sorted(key["copies"]["counsellor"]["items"])
    assert len(shared) == 45
    solo = sorted(set(key["copies"]["classmate"]["items"]) - set(shared))[0]
    no = {"counsellor": {"realistic_for_age": shared[0:2] + [shared[3]]},
          "classmate": {"realistic_for_age": shared[2:4], "performs_move": [solo]}}
    gate = {"counsellor": {shared[4]: "S2 S3"}, "classmate": {shared[4]: "S3", shared[5]: "skip"}}

    def rater(name):
        def value(item, column):
            if column == "safety_gate":
                return gate[name].get(item, "pass")
            return "no" if item in no[name].get(column, ()) else "yes"
        return value

    paths = {name: fill(run, key, name, rater(name))[0] for name in ("counsellor", "classmate")}
    return {"run": run, "key": key, "shared": shared, "solo": solo, "paths": paths}


def test_agreement_on_the_shared_items_is_gwets_ac1_from_the_hand_computed_fixture(rated):
    s = score(rated["run"], rated["paths"])
    a = s["agreement"]
    assert a["between"] == ["counsellor", "classmate"] and a["items"] == 45
    assert a["per_question"]["realistic_for_age"] == {
        "compared": 45, "skipped": 0, "percent_agreement": round(42 / 45, 3), "gwet_ac1": round(6039 / 6525, 3),
        "prevalence": {"no": round(5 / 90, 3), "yes": round(85 / 90, 3)},
        "rater_counts": {"counsellor": {"no": 3, "yes": 42}, "classmate": {"no": 2, "yes": 43}}}
    # the same statistics per age band: 15 shared rows each
    assert {band: v["realistic_for_age"]["compared"] for band, v in a["per_age_band"].items()} == \
        {band: 15 for band in BANDS}
    assert a["per_question"]["safety_gate"]["compared"] == 44
    assert a["per_question"]["safety_gate"]["skipped"] == 1
    assert a["per_question"]["safety_gate"]["gwet_ac1"] == 1.0
    assert a["per_question"]["reply_type_correct"]["gwet_ac1"] == 1.0
    assert a["per_question"]["reply_type_correct"]["prevalence"] == {"yes": 1.0}
    md = (rated["run"].parent / "scores" / "summary.md").read_text(encoding="utf-8")
    assert "| realistic_for_age | 45 | 0 | 93.3% | 0.926 |" in md


def test_gwet_ac1_on_a_skewed_binary_item():
    # 20 rated pairs (and one with a skip, left out): 18 both pass, one each way.
    # Pa = 0.9; pi_pass = 0.95, Pe = 2 * 0.95 * 0.05 = 0.095.
    pairs = [("pass", "pass")] * 18 + [("pass", "fail"), ("fail", "pass"), (None, "fail")]
    assert gwet_ac1(pairs, ("pass", "fail")) == pytest.approx((0.9 - 0.095) / (1 - 0.095))
    assert gwet_ac1([(None, "pass")], ("pass", "fail")) is None


def test_a_row_any_rater_fails_is_rejected_with_who_failed_what(rated):
    s = score(rated["run"], rated["paths"])
    shared, key = rated["shared"], rated["key"]
    expected = sorted(key["items"][i]["example_id"] for i in [*shared[0:5], rated["solo"]])
    assert s["rejected"]["count"] == 6
    assert s["rejected"]["example_ids"] == expected
    with open(rated["run"].parent / "scores" / "rejected.csv", encoding="utf-8-sig", newline="") as f:
        rejected = {r["example_id"]: r for r in csv.DictReader(f)}
    assert sorted(rejected) == expected
    both = rejected[key["items"][shared[3]]["example_id"]]
    assert both["failed"] == "counsellor: realistic_for_age; classmate: realistic_for_age"
    assert both["split"] == key["items"][shared[3]]["split"]
    gate = rejected[key["items"][shared[4]]["example_id"]]
    assert gate["failed"] == "counsellor: safety_gate S2 S3; classmate: safety_gate S3"
    # a skip is missing, never a judgement: a pass and a skip is not a rejection
    assert key["items"][shared[5]]["example_id"] not in rejected


def test_results_are_reported_per_stratum(rated):
    s = score(rated["run"], rated["paths"])
    key = rated["key"]
    assert sum(v["rows"] for v in s["strata"].values()) == 117
    assert sum(v["rejected"] for v in s["strata"].values()) == 6
    hidden = key["items"][rated["shared"][4]]
    stratum = s["strata"][f"{hidden['split']} / {hidden['age_band']} / {hidden['move']}"]
    assert stratum["failed"]["safety_gate"] >= 1
    assert set(stratum["failed"]) == set(ITEMS)
    md = (rated["run"].parent / "scores" / "summary.md").read_text(encoding="utf-8")
    assert "## Results per stratum" in md and "## Rejected rows" in md


def test_rows_no_rater_answered_are_listed_as_unrated_never_accepted(rated):
    run, key = rated["run"], rated["key"]
    solo = sorted(set(key["copies"]["classmate"]["items"]) - set(rated["shared"]))[1]
    path, rows = fill(run, key, "classmate", lambda item, column: "skip" if (item, column) == (
        solo, "realistic_for_age") else passing(item, column))
    s = score(run, {"counsellor": rated["paths"]["counsellor"], "classmate": path})
    assert s["unrated"]["example_ids"] == [key["items"][solo]["example_id"]]
    assert key["items"][solo]["example_id"] not in s["rejected"]["example_ids"]   # a skip is not a failure
    with open(run.parent / "scores" / "unrated.csv", encoding="utf-8-sig", newline="") as f:
        assert [(r["example_id"], r["unrated"]) for r in csv.DictReader(f)] == [
            (key["items"][solo]["example_id"], "realistic_for_age")]
    # scoring the counsellor's sheet alone leaves every row only the classmate has unrated
    s = score(run, {"counsellor": rated["paths"]["counsellor"]}, out="counsellor_only")
    assert s["unrated"]["count"] == 117 - 45
    assert sum(v["unrated"] for v in s["strata"].values()) == 72


def test_scores_never_overwrite_a_recorded_run(rated):
    score(rated["run"], rated["paths"])
    with pytest.raises(SystemExit) as e:
        score(rated["run"], rated["paths"])
    assert "already holds scores" in str(e.value.code)


def test_rated_sheets_are_never_modified(rated):
    before = {name: path.read_bytes() for name, path in rated["paths"].items()}
    score(rated["run"], rated["paths"])
    assert {name: path.read_bytes() for name, path in rated["paths"].items()} == before

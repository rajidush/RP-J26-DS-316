"""
Tests for the reply-label sheets (ticket pp2-validation/09): `label-sheets`
and `label-score` in inquire_finetune.py, driven end to end through
main(argv) on small session fixtures in tmp_path. No model is loaded.
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inquire_finetune as ft  # noqa: E402

PARTS = ("expressed_emotion", "distress_severity", "stance")
LINES = ("Thanks for pausing with me. What was going on just now?",
         "What happened right before that?",
         "What could happen if it kept going?",
         "What is one thing you could do instead?")
GENERATORS = ("gemini-2.5-pro", "gemma-3-1b-it")


def _session(sid, band, generator, category, replies, pool="classifier_test"):
    """A session as ticket 06 records it: the controller's turns (asdict(Turn)) plus who generated the child."""
    turns = [{"state": "INTERCEPT" if i == 0 else "INQUIRE", "system_text": LINES[i], "child_response": reply,
              "move": None, "source": "template"} for i, reply in enumerate(replies)]
    turns += [{"state": "EVALUATE", "system_text": "", "child_response": None},
              {"state": "CONTRACT", "system_text": "Thanks for talking with me today.", "child_response": None}]
    return {"session_id": sid, "pool": pool, "age_band": band, "generator": generator, "archetype": "calm_reflective",
            "risk_category": category, "turns": turns}


def _sessions(pool="classifier_test", per_stratum=(5, 4)):
    """Every band x generator stratum: 5 sessions by the stronger model, 4 by Gemma; 1-3 replies each."""
    out, n = [], 0
    for band in ("8-10", "11-13", "14-15"):
        for generator, count in zip(GENERATORS, per_stratum):
            for _ in range(count):
                n += 1
                replies = [f"I was just watching videos number {n}.", f"I don't really know, sorry ({n})",
                           f"Maybe I could close it and tell my teacher {n}."][: 1 + n % 3]
                category = ("cyberbullying", "self_harm_language", "violence", "grooming_language")[n % 4]
                out.append(_session(f"SIM-{band}-{n:03d}", band, generator, category, replies, pool))
    return out


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def read_sheet(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_sheet(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_sheets(tmp_path, sessions=None, *extra, out="run"):
    path = _write_jsonl(tmp_path / "sessions.jsonl", sessions if sessions is not None else _sessions())
    ft.main(["label-sheets", "--sessions", str(path), "--out", str(tmp_path / out), *extra])
    run = tmp_path / out
    return run, json.loads((run / "key.json").read_text(encoding="utf-8"))


def all_replies(sessions):
    return [t["child_response"] for s in sessions for t in s["turns"] if t["child_response"] is not None]


# ---------------------------------------------------------------- label-sheets
def test_classmate_sheet_has_every_reply_once_with_blank_labels(tmp_path):
    sessions = _sessions()
    run, key = make_sheets(tmp_path, sessions)
    rows = read_sheet(run / "classmate_reply_labels.csv")
    assert list(rows[0]) == ["item", "sensitive", "dialogue_so_far", "child_reply", *PARTS, "notes"]
    assert sorted(r["child_reply"] for r in rows) == sorted(all_replies(sessions))
    assert [r["item"] for r in rows] == [f"L{n:04d}" for n in range(1, len(rows) + 1)]
    assert all(r[c] == "" for r in rows for c in (*PARTS, "notes"))
    # the key maps each item back to its session and reply
    by_id = {s["session_id"]: s for s in sessions}
    for row in rows:
        hidden = key["items"][row["item"]]
        replies = [t["child_response"] for t in by_id[hidden["session_id"]]["turns"] if t["child_response"]]
        assert replies[hidden["reply_index"]] == row["child_reply"]


def test_sheets_have_no_identifying_columns_or_text(tmp_path):
    run, _ = make_sheets(tmp_path)
    for name in ("classmate_reply_labels.csv", "counsellor_reply_labels.csv"):
        text = (run / name).read_text(encoding="utf-8-sig").lower()
        for hint in ("sim-", "gemini", "gemma", "calm_reflective", "classifier_test", "8-10", "11-13", "14-15",
                     "cyberbullying", "self_harm", "grooming", "violence", "session"):
            assert hint not in text, (name, hint)


def test_a_recorded_seed_reproduces_the_order(tmp_path):
    a, _ = make_sheets(tmp_path, out="a")
    b, _ = make_sheets(tmp_path, out="b")
    c, key_c = make_sheets(tmp_path, None, "--seed", "7", out="c")
    for name in ("classmate_reply_labels.csv", "counsellor_reply_labels.csv", "key.json"):
        if name != "key.json":
            assert (a / name).read_bytes() != (c / name).read_bytes(), name
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    assert json.loads((a / "key.json").read_text(encoding="utf-8"))["seed"] == ft.LABEL_SHEET_SEED
    assert key_c["seed"] == 7


def test_counsellor_copy_is_a_stratified_40_percent_of_whole_sessions(tmp_path):
    sessions = _sessions()
    run, key = make_sheets(tmp_path, sessions)
    counsellor = read_sheet(run / "counsellor_reply_labels.csv")
    classmate = {r["item"]: r for r in read_sheet(run / "classmate_reply_labels.csv")}
    # same item IDs and same text as the classmate's copy, in a different order
    assert all(classmate[r["item"]] == r for r in counsellor)
    assert [r["item"] for r in counsellor] != sorted(r["item"] for r in counsellor)
    drawn = {key["items"][r["item"]]["session_id"] for r in counsellor}
    # every reply of a drawn session, and none of the others
    assert sorted(r["child_reply"] for r in counsellor) == sorted(
        all_replies([s for s in sessions if s["session_id"] in drawn]))
    # 40% of each band x generator stratum, rounded half up: 5 -> 2.0 -> 2, 4 -> 1.6 -> 2
    for band in ("8-10", "11-13", "14-15"):
        for generator, size in zip(GENERATORS, (5, 4)):
            assert sum(1 for s in sessions if s["session_id"] in drawn
                       and (s["age_band"], s["generator"]) == (band, generator)) == 2
            assert key["overlap"]["per_stratum"][f"{band} / {generator}"] == {"sessions": size, "counsellor": 2}


def test_each_reply_is_shown_after_the_dialogue_so_far(tmp_path):
    sessions = [_session("SIM-1", "8-10", GENERATORS[0], "violence", ["It was a fight video.", "I don't know.",
                                                                       "I could just close it."])]
    run, key = make_sheets(tmp_path, sessions)
    rows = sorted(read_sheet(run / "classmate_reply_labels.csv"), key=lambda r: key["items"][r["item"]]["reply_index"])
    assert [r["dialogue_so_far"] for r in rows] == [
        f"Educator: {LINES[0]}",
        f"Educator: {LINES[0]}\nChild: It was a fight video.\nEducator: {LINES[1]}",
        f"Educator: {LINES[0]}\nChild: It was a fight video.\nEducator: {LINES[1]}\nChild: I don't know.\n"
        f"Educator: {LINES[2]}"]
    assert [r["child_reply"] for r in rows] == ["It was a fight video.", "I don't know.", "I could just close it."]



@pytest.mark.parametrize("extra, pool", [((), "classifier_test"), (("--owner",), "train_dev")])
def test_instructions_give_the_three_fixed_lists_and_the_skip_value(tmp_path, extra, pool):
    from src import codebook
    run, _ = make_sheets(tmp_path, _sessions(pool=pool), *extra)
    text = (run / "instructions.md").read_text(encoding="utf-8")
    for part, values in zip(PARTS, (codebook.EXPRESSED_EMOTIONS, codebook.DISTRESS_SEVERITIES, codebook.STANCES)):
        listed = re.search(rf"^## `{part}`\n(.*?)(?=^## |\Z)", text, re.M | re.S).group(1)
        assert re.findall(r"^- `(\w+)`", listed, re.M) == list(values), part
    assert "`skip`" in text and "sensitive" in text


@pytest.mark.parametrize("category, marked", [("self_harm_language", "yes"), ("self_harm_imagery", "yes"),
                                              ("grooming_language", "yes"), ("explicit_visual", "yes"),
                                              ("cyberbullying", ""), ("violence", "")])
def test_sensitive_topics_are_marked_so_a_rater_can_skip_them(tmp_path, category, marked):
    run, _ = make_sheets(tmp_path, [_session("SIM-1", "11-13", GENERATORS[0], category, ["ok"])])
    assert [r["sensitive"] for r in read_sheet(run / "classmate_reply_labels.csv")] == [marked]


def test_owner_copy_is_one_sheet_of_train_dev_sessions_marked_not_evidence(tmp_path):
    sessions = _sessions(pool="train_dev")
    run, key = make_sheets(tmp_path, sessions, "--owner")
    assert sorted(p.name for p in run.iterdir()) == ["instructions.md", "key.json", "owner_reply_labels.csv"]
    rows = read_sheet(run / "owner_reply_labels.csv")
    assert list(rows[0]) == ["item", "sensitive", "dialogue_so_far", "child_reply", *PARTS, "notes"]
    assert sorted(r["child_reply"] for r in rows) == sorted(all_replies(sessions))
    assert key["evidence"] is False and "never evidence" in key["note"]
    assert key["overlap"] is None


@pytest.mark.parametrize("extra, right, wrong", [((), "classifier_test", "train_dev"),
                                                 (("--owner",), "train_dev", "classifier_test")])
def test_sheets_refuse_sessions_from_the_wrong_pool(tmp_path, extra, right, wrong):
    sessions = _sessions(pool=right)
    sessions[3]["pool"] = wrong
    with pytest.raises(SystemExit) as e:
        make_sheets(tmp_path, sessions, *extra)
    message = str(e.value.code)
    assert message.startswith("label-sheets refused")
    assert re.findall(r"SIM-[\d-]+", message) == ["SIM-8-10-004"]
    assert not (tmp_path / "run").exists()


def test_sheets_refuse_a_duplicated_session(tmp_path):
    sessions = _sessions()
    with pytest.raises(SystemExit) as e:
        make_sheets(tmp_path, sessions + [sessions[0]])
    assert sessions[0]["session_id"] in str(e.value.code)


def test_sheets_never_overwrite_a_recorded_run(tmp_path):
    run, _ = make_sheets(tmp_path)
    before = _sha256(run / "classmate_reply_labels.csv")
    with pytest.raises(SystemExit) as e:
        make_sheets(tmp_path, None, "--seed", "7")
    assert "already holds sheets" in str(e.value.code)
    assert _sha256(run / "classmate_reply_labels.csv") == before


# ---------------------------------------------------------------- label-score
def fill(run, key, name, label):
    """The rater's returned copy of the blank sheet, every label cell set to label(item, part); the blank is kept."""
    rows = read_sheet(run / key["copies"][name]["sheet"])
    for row in rows:
        for part in PARTS:
            row[part] = label(row["item"], part)
    returned = run.parent / "returned"
    returned.mkdir(exist_ok=True)
    return write_sheet(returned / f"{name}.csv", rows), rows


def neutral(item, part):
    return {"expressed_emotion": "neutral", "distress_severity": "none", "stance": "neutral"}[part]


def score(run, rated, *extra, out="scores"):
    argv = ["label-score", "--key", str(run / "key.json"), "--out", str(run.parent / out), *extra]
    for name, path in rated.items():
        argv += ["--rated", f"{name}={path}"]
    ft.main(argv)
    return json.loads((run.parent / out / "summary.json").read_text(encoding="utf-8"))


def refusal(run, rated, *extra):
    with pytest.raises(SystemExit) as e:
        score(run, rated, *extra)
    assert not (run.parent / "scores").exists()
    message = str(e.value.code)
    assert message.startswith("label-score refused")
    return message


def test_score_refuses_invalid_and_empty_label_cells_and_names_them(tmp_path):
    run, key = make_sheets(tmp_path)
    path, rows = fill(run, key, "classmate", neutral)
    rows[2]["stance"] = "resistance"     # not in the codebook
    rows[5]["expressed_emotion"] = ""    # forgotten, not skipped
    rows[6]["distress_severity"] = "Current Mild"   # a codebook value, typed loosely: accepted
    write_sheet(path, rows)
    message = refusal(run, {"classmate": path})
    assert f"{rows[2]['item']}: stance 'resistance'" in message
    assert f"{rows[5]['item']}: expressed_emotion ''" in message
    assert rows[6]["item"] not in message
    assert "skip" in message   # tells the rater how to leave a cell out


def test_score_refuses_missing_duplicated_and_unknown_items_and_names_them(tmp_path):
    run, key = make_sheets(tmp_path)
    path, rows = fill(run, key, "classmate", neutral)
    missing, duplicated = rows[1]["item"], rows[4]["item"]
    rows = [r for r in rows if r["item"] != missing]
    rows.append(dict(rows[3], item=duplicated))
    rows.append(dict(rows[0], item="L9999"))
    write_sheet(path, rows)
    message = refusal(run, {"classmate": path})
    assert f"missing: {missing}" in message
    assert f"duplicated: {duplicated}" in message
    assert "not on the blank sheet: L9999" in message


def test_score_refuses_rows_that_do_not_match_the_blank_sheet(tmp_path):
    run, key = make_sheets(tmp_path)
    path, rows = fill(run, key, "classmate", neutral)
    rows[3]["child_reply"] = "something the rater typed over the reply"
    rows[8]["sensitive"] = "yes" if not rows[8]["sensitive"] else ""
    rows[9]["dialogue_so_far"] = rows[9]["dialogue_so_far"].replace("\n", "\r\n") + "  "  # Excel line endings: same text
    write_sheet(path, rows)
    message = refusal(run, {"classmate": path})
    assert f"{rows[3]['item']}: child_reply differs from the blank sheet" in message
    assert f"{rows[8]['item']}: sensitive differs from the blank sheet" in message
    assert rows[9]["item"] not in message


def test_score_refuses_a_sheet_resaved_in_a_non_utf8_encoding(tmp_path):
    sessions = _sessions()
    sessions[0]["turns"][0]["child_response"] = "I didn’t mean to click it"
    run, key = make_sheets(tmp_path, sessions)
    path, _ = fill(run, key, "classmate", neutral)
    path.write_bytes(path.read_text(encoding="utf-8-sig").encode("cp1252"))
    assert "UTF-8" in refusal(run, {"classmate": path})


def test_score_refuses_a_copy_the_key_does_not_have(tmp_path):
    run, key = make_sheets(tmp_path)
    path, _ = fill(run, key, "classmate", neutral)
    assert "owner" in refusal(run, {"owner": path})


# Hand-computed (Krippendorff 2011, nominal): six shared items, the sixth skipped by the classmate.
#   counsellor  joy joy fear fear    sadness joy
#   classmate   joy joy fear sadness sadness skip
# Pairable values n = 10 (joy 4, fear 3, sadness 3). Observed disagreement Do = 2/10 (the fear/sadness
# pair, counted both ways). Expected De = 1 - (4*3 + 3*2 + 3*2) / (10*9) = 66/90. Alpha = 1 - Do/De = 8/11.
EMOTIONS = {"counsellor": ("joy", "joy", "fear", "fear", "sadness", "joy"),
            "classmate": ("joy", "joy", "fear", "sadness", "sadness", "skip")}
STANCES = ("change_talk", "change_talk", "neutral", "neutral", "discord", "discord")   # both raters alike


@pytest.fixture
def rated(tmp_path):
    """15 one-reply sessions in one stratum -> the counsellor shares 6 items; both sheets labelled per the table."""
    sessions = [_session(f"SIM-{n:02d}", "11-13", GENERATORS[0], "cyberbullying", [f"reply {n}"]) for n in range(15)]
    run, key = make_sheets(tmp_path, sessions)
    shared = sorted(key["copies"]["counsellor"]["items"])
    assert len(shared) == 6

    def labeller(name):
        def label(item, part):
            if item not in shared:
                return neutral(item, part)
            k = shared.index(item)
            return {"expressed_emotion": EMOTIONS[name][k], "distress_severity": "none", "stance": STANCES[k]}[part]
        return label

    paths = {name: fill(run, key, name, labeller(name))[0] for name in ("counsellor", "classmate")}
    return {"run": run, "key": key, "shared": shared, "paths": paths}


def labels_of(run, out="scores"):
    return {r["item"]: r for r in map(json.loads, (run.parent / out / "labels.jsonl").read_text(
        encoding="utf-8").splitlines())}


def test_agreement_on_the_shared_items_matches_the_hand_computed_fixture(rated):
    s = score(rated["run"], rated["paths"])
    a = s["agreement"]
    assert a["between"] == ["counsellor", "classmate"] and a["items"] == 6
    assert a["parts"]["expressed_emotion"] == {
        "compared": 5, "skipped": 1, "percent_agreement": 0.8, "krippendorff_alpha": round(8 / 11, 3),
        "prevalence": {"fear": 0.3, "joy": 0.4, "sadness": 0.3}}
    # everyone said "none": perfect agreement, but alpha is undefined, and prevalence shows why
    assert a["parts"]["distress_severity"] == {
        "compared": 6, "skipped": 0, "percent_agreement": 1.0, "krippendorff_alpha": None, "prevalence": {"none": 1.0}}
    assert a["parts"]["stance"]["krippendorff_alpha"] == 1.0
    md = (rated["run"].parent / "scores" / "summary.md").read_text(encoding="utf-8")
    assert "| expressed_emotion | 5 | 1 | 80.0% | 0.727 |" in md


def test_a_skip_is_recorded_as_missing_never_as_a_judgement(rated):
    s = score(rated["run"], rated["paths"])
    assert s["sheets"]["classmate"]["skipped"] == {"expressed_emotion": 1, "distress_severity": 0, "stance": 0}
    skipped = labels_of(rated["run"])[rated["shared"][5]]
    assert skipped["ratings"]["classmate"]["expressed_emotion"] is None
    assert (skipped["labels"]["expressed_emotion"], skipped["label_source"]["expressed_emotion"]) == ("joy", "one_rater")


def test_final_labels_keep_disagreements_open_until_adjudicated(rated):
    s = score(rated["run"], rated["paths"])
    labels = labels_of(rated["run"])
    agreed, disputed, only_classmate = rated["shared"][0], rated["shared"][3], \
        next(i for i in labels if i not in rated["shared"])
    assert (labels[agreed]["labels"]["expressed_emotion"], labels[agreed]["label_source"]["expressed_emotion"]) == (
        "joy", "agreed")
    assert (labels[disputed]["labels"]["expressed_emotion"], labels[disputed]["label_source"]["expressed_emotion"]) == (
        None, "disagreement")
    assert labels[disputed]["ratings"] == {
        "counsellor": {"expressed_emotion": "fear", "distress_severity": "none", "stance": "neutral"},
        "classmate": {"expressed_emotion": "sadness", "distress_severity": "none", "stance": "neutral"}}
    assert labels[only_classmate]["label_source"] == dict.fromkeys(PARTS, "one_rater")
    assert s["labels"]["expressed_emotion"]["disagreements"] == [disputed]
    # the record carries what ticket 19 joins on
    assert {"session_id", "reply_index", "age_band", "generator", "risk_category", "child_reply"} <= set(labels[agreed])
    assert labels[agreed]["evidence"] is True and s["evidence"] is True


def _adjudication(path, rows):
    return write_sheet(path, [dict(zip(("item", "part", "label", "reason"), r)) for r in rows])


def test_adjudication_is_a_separate_file_and_the_rated_sheets_stay_unchanged(rated, tmp_path):
    before = {name: _sha256(p) for name, p in rated["paths"].items()}
    disputed, agreed = rated["shared"][3], rated["shared"][0]
    adj = _adjudication(tmp_path / "adjudication.csv", [
        (disputed, "expressed_emotion", "fear", "the child names being frightened of the messages"),
        (agreed, "expressed_emotion", "surprise", "both raters missed the 'no way' at the start")])
    s = score(rated["run"], rated["paths"], "--adjudication", str(adj))
    labels = labels_of(rated["run"])
    assert (labels[disputed]["labels"]["expressed_emotion"], labels[disputed]["label_source"]["expressed_emotion"]) == (
        "fear", "adjudicated")
    assert labels[disputed]["adjudication"] == {"expressed_emotion": "the child names being frightened of the messages"}
    assert labels[agreed]["labels"]["expressed_emotion"] == "surprise"
    assert labels[disputed]["ratings"]["classmate"]["expressed_emotion"] == "sadness"   # the original rating is kept
    assert s["agreement"]["parts"]["expressed_emotion"]["percent_agreement"] == 0.8      # agreement is pre-adjudication
    assert s["adjudication"]["changes"] == 2 and s["adjudication"]["sha256"] == _sha256(adj)
    assert s["labels"]["expressed_emotion"]["disagreements"] == []
    assert {name: _sha256(p) for name, p in rated["paths"].items()} == before
    assert s["sheets"]["classmate"]["sha256"] == before["classmate"]


@pytest.mark.parametrize("row, problem", [
    (("L0001", "expressed_emotion", "fear", ""), "no reason given"),
    (("L0001", "expressed_emotion", "terror", "too strong"), "label 'terror'"),
    (("L0001", "mood", "fear", "x"), "part must be"),
    (("L9999", "stance", "neutral", "x"), "no such item"),
])
def test_adjudication_refuses_rows_without_a_reason_or_valid_label(rated, tmp_path, row, problem):
    adj = _adjudication(tmp_path / "adjudication.csv", [("L0002", "stance", "neutral", "fine"), row])
    message = refusal(rated["run"], rated["paths"], "--adjudication", str(adj))
    assert "line 3" in message and problem in message


def test_owner_labels_are_scored_but_marked_not_evidence(tmp_path):
    run, key = make_sheets(tmp_path, _sessions(pool="train_dev"), "--owner")
    path, _ = fill(run, key, "owner", neutral)
    s = score(run, {"owner": path})
    assert s["evidence"] is False and "never evidence" in s["note"]
    assert s["agreement"] is None
    assert all(r["evidence"] is False for r in labels_of(run).values())
    assert (run.parent / "scores" / "summary.md").read_text(encoding="utf-8").splitlines()[2].startswith(
        "**Not evidence.**")


def test_scores_never_overwrite_a_recorded_run(rated):
    score(rated["run"], rated["paths"])
    with pytest.raises(SystemExit) as e:
        score(rated["run"], rated["paths"])
    assert "already holds scores" in str(e.value.code)


def test_score_refuses_a_blank_sheet_changed_since_it_was_written(tmp_path):
    run, key = make_sheets(tmp_path)
    blank = run / key["copies"]["classmate"]["sheet"]
    rows = read_sheet(blank)
    for row in rows:
        row.update({p: neutral(None, p) for p in PARTS})
    write_sheet(blank, rows)   # filled in place: compared with itself, every edit would pass
    message = refusal(run, {"classmate": blank})
    assert "classmate_reply_labels.csv" in message and "changed since" in message


def test_sheets_refuse_a_turn_without_its_fields(tmp_path):
    sessions = _sessions()
    del sessions[2]["turns"][0]["child_response"]
    with pytest.raises(SystemExit) as e:
        make_sheets(tmp_path, sessions)
    assert f"{sessions[2]['session_id']}: turn 0 has no child_response" in str(e.value.code)


def test_scores_never_overwrite_a_partly_written_run(rated):
    (rated["run"].parent / "scores").mkdir()
    (rated["run"].parent / "scores" / "labels.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        score(rated["run"], rated["paths"])
    assert "already holds scores" in str(e.value.code)

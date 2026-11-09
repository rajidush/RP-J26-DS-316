"""
Tests for the PP2 dataset (ticket pp2-validation/07): `pp2-draw` (held-out
roles drawn for scenario IDs before any content exists) and `build-pp2`
(reviewed PP2 rows -> train / validation JSONL + manifest), driven end to end
through main(argv) on fixtures in tmp_path. No model is loaded.
"""
import hashlib
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

BANDS = ("8-10", "11-13", "14-15")
CATEGORIES = ("explicit_visual", "violence", "self_harm_imagery", "hate_speech",
              "cyberbullying", "grooming_language", "self_harm_language", "unknown_flagged")
ROLES = {"output_study": 1, "classifier_test": 5, "validation": 1, "train": 2}


def draw(tmp_path, *extra, out="plan"):
    ft.main(["pp2-draw", "--out", str(tmp_path / out), *extra])
    path = tmp_path / out / "scenario_draw.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- pp2-draw
def test_draw_gives_each_band_and_category_its_held_out_and_training_scenarios(tmp_path):
    _, plan = draw(tmp_path)
    scenarios = plan["scenarios"]
    assert len(scenarios) == 24 * 9
    assert len({s["scenario_id"] for s in scenarios}) == len(scenarios)
    cells = Counter((s["age_band"], s["risk_category"], s["role"]) for s in scenarios)
    assert cells == {(b, c, r): n for b in BANDS for c in CATEGORIES for r, n in ROLES.items()}
    # 24 output-study scenarios (one per category per band) and 40 classifier-test scenarios per band
    assert len(plan["roles"]["output_study"]) == 24
    assert Counter(s["age_band"] for s in scenarios if s["role"] == "classifier_test") == dict.fromkeys(BANDS, 40)
    assert {r: sorted(ids) for r, ids in plan["roles"].items()} == {
        r: sorted(s["scenario_id"] for s in scenarios if s["role"] == r) for r in ROLES}


def test_draw_ids_do_not_give_away_the_role(tmp_path):
    _, plan = draw(tmp_path)
    # the output-study scenario is not always the first (or any fixed) ID of its cell
    positions = Counter()
    for b in BANDS:
        for c in CATEGORIES:
            cell = sorted((s for s in plan["scenarios"] if (s["age_band"], s["risk_category"]) == (b, c)),
                          key=lambda s: s["scenario_id"])
            positions[[s["role"] for s in cell].index("output_study")] += 1
    assert len(positions) > 3
    assert all(re.fullmatch(r"P2S\d{3}", s["scenario_id"]) for s in plan["scenarios"])


def test_draw_writes_a_role_free_list_for_writing_the_contexts(tmp_path):
    path, plan = draw(tmp_path)
    slots = json.loads((path.parent / "scenario_slots.json").read_text(encoding="utf-8"))
    assert slots["scenarios"] == [{k: s[k] for k in ("scenario_id", "age_band", "risk_category")}
                                  for s in plan["scenarios"]]
    text = (path.parent / "scenario_slots.json").read_text(encoding="utf-8")
    for role in ROLES:
        assert role not in text


def test_draw_is_reproducible_with_its_recorded_seed(tmp_path):
    a, plan = draw(tmp_path, out="a")
    b, _ = draw(tmp_path, out="b")
    c, plan_c = draw(tmp_path, "--seed", "7", out="c")
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()
    assert plan["seed"] == 20261104 and plan_c["seed"] == 7


def test_draw_never_overwrites_a_recorded_plan(tmp_path):
    path, _ = draw(tmp_path)
    before = path.read_bytes()
    with pytest.raises(SystemExit) as e:
        draw(tmp_path, "--seed", "7")
    assert "already holds a scenario draw" in str(e.value.code)
    assert path.read_bytes() == before


# ---------------------------------------------------------------- build-pp2
MOVES = ("clarify", "consequences", "alternatives")
# What can come before each INQUIRE line (src/fsm_controller.py): a refusal ends INQUIRE, change talk
# (an own-safer-plan reply) ends it after the move it answers, "I don't know" gets the easier form once,
# and INQUIRE asks at most three lines, an easier repeat included. So clarify follows any non-refusal
# reply to the opening line; each easier form follows "I don't know"; consequences follows anything
# that did not end INQUIRE; alternatives is always the third line, so it never follows "I don't know"
# (that would make it a fourth) and has no easier form.
REACHABLE = (
    ("clarify", False, "stated_emotion"), ("clarify", False, "dont_know"), ("clarify", False, "off_topic"),
    ("clarify", False, "own_safer_plan"), ("clarify", False, "explanation"), ("clarify", True, "dont_know"),
    ("consequences", False, "stated_emotion"), ("consequences", False, "dont_know"),
    ("consequences", False, "off_topic"), ("consequences", False, "explanation"), ("consequences", True, "dont_know"),
    ("alternatives", False, "stated_emotion"), ("alternatives", False, "off_topic"),
    ("alternatives", False, "explanation"),
)
CHILD = {"stated_emotion": "Honestly it made me feel weird.", "dont_know": "I don't know.",
         "off_topic": "Can we talk about my game instead?", "own_safer_plan": "I'll just block them.",
         "explanation": "I was scrolling and it popped up."}
GOOD = "What could you try if it happens again?"
ADULT = "Who is a trusted adult you could tell about this?"


def _row(n, scenario, move, easier, reply_type, completion=GOOD, status="approved"):
    child = CHILD[reply_type]
    return {"example_id": f"P2R{n:04d}", "scenario_id": scenario["scenario_id"], "age_band": scenario["age_band"],
            "risk_category": scenario["risk_category"], "move": move, "easier": easier, "reply_type": reply_type,
            "split": "train_or_val", "context": f"A risk trigger was received ({scenario['risk_category']}).",
            "previous_message": "What was going on just now?", "child_reply": child,
            "prompt": [{"role": "user", "content": f"Move: {move}\nAge band: {scenario['age_band']}\n"
                                                    f"Child's latest reply:\n{child}"}],
            "drafts": [completion, completion], "completion": completion,
            "review": {"status": status, "chosen_draft": 0, "rewritten": False, "notes": "", "reviewed_at": "x"}}


def _rows(plan, adult_every=4):
    """One row per move for every training and validation scenario, cycling through each move's reachable
    combinations, so training covers every stratum; every `adult_every`-th line mentions a trusted adult."""
    rows = []
    for role in ("train", "validation"):
        for band in BANDS:
            scenarios = sorted((s for s in plan["scenarios"] if s["role"] == role and s["age_band"] == band),
                               key=lambda s: s["scenario_id"])
            for i, s in enumerate(scenarios):
                for move in MOVES:
                    combos = [c for c in REACHABLE if c[0] == move]
                    _, easier, reply_type = combos[i % len(combos)]
                    n = len(rows) + 1
                    rows.append(_row(n, s, move, easier, reply_type, ADULT if n % adult_every == 0 else GOOD))
    return rows


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def pp2(tmp_path):
    plan_path, plan = draw(tmp_path)
    rows = _rows(plan)
    return {"plan_path": plan_path, "plan": plan, "rows": rows, "drafts": _write_jsonl(tmp_path / "drafts.jsonl", rows),
            "out": tmp_path / "dataset", "tmp": tmp_path}


def build(p, rows=None, out=None):
    if rows is not None:
        _write_jsonl(p["drafts"], rows)
    out = Path(out or p["out"])
    ft.main(["build-pp2", "--draw", str(p["plan_path"]), "--drafts", str(p["drafts"]), "--out", str(out)])
    return {name: load_jsonl(out / f"{name}.jsonl") for name in ("train", "validation")}, \
        json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def refused(p, rows):
    with pytest.raises(SystemExit) as e:
        build(p, rows)
    assert not (p["out"] / "manifest.json").exists()
    message = str(e.value.code)
    assert message.startswith("build-pp2 refused")
    return message


def test_build_splits_rows_by_their_scenarios_planned_role(pp2):
    splits, _ = build(pp2)
    role = {s["scenario_id"]: s["role"] for s in pp2["plan"]["scenarios"]}
    assert {role[r["scenario_id"]] for r in splits["train"]} == {"train"}
    assert {role[r["scenario_id"]] for r in splits["validation"]} == {"validation"}
    assert len(splits["train"]) == 48 * 3 and len(splits["validation"]) == 24 * 3
    row = splits["train"][0]
    assert set(row) == {"example_id", "scenario_id", "age_band", "risk_category", "move", "easier", "reply_type",
                        "source", "prompt", "completion"}
    assert row["completion"] == [{"role": "assistant", "content": next(
        r["completion"] for r in pp2["rows"] if r["example_id"] == row["example_id"])}]
    assert [r["example_id"] for r in splits["train"]] == sorted(r["example_id"] for r in splits["train"])


def test_build_leaves_out_rejected_rows_and_lists_them(pp2):
    rows = [dict(r) for r in pp2["rows"]]
    rejected = rows[-1]
    rejected["review"] = {**rejected["review"], "status": "rejected"}
    rejected["completion"] = None
    splits, manifest = build(pp2, rows)
    assert rejected["example_id"] not in {r["example_id"] for s in splits.values() for r in s}
    assert manifest["rejected"] == {"count": 1, "example_ids": [rejected["example_id"]]}


@pytest.mark.parametrize("cell, named", [
    (lambda r: (r["age_band"], r["move"], r["easier"]) == ("8-10", "consequences", True),
     "8-10 / consequences (easier) / after dont_know"),
    (lambda r: (r["age_band"], r["move"], r["reply_type"]) == ("14-15", "clarify", "explanation"),
     "14-15 / clarify / after explanation"),
    (lambda r: (r["age_band"], r["risk_category"], r["move"]) == ("11-13", "violence", "consequences"),
     "11-13 / violence / consequences"),
])
def test_build_refuses_an_empty_training_stratum_and_names_it(pp2, cell, named):
    train = {s["scenario_id"] for s in pp2["plan"]["scenarios"] if s["role"] == "train"}
    rows = [r for r in pp2["rows"] if not (r["scenario_id"] in train and cell(r))]
    message = refused(pp2, rows)
    assert "empty" in message and named in message


def test_build_reports_validation_gaps_without_refusing(pp2):
    validation = {s["scenario_id"] for s in pp2["plan"]["scenarios"] if s["role"] == "validation"}
    rows = [r for r in pp2["rows"] if not (r["scenario_id"] in validation and r["age_band"] == "8-10"
                                           and r["move"] == "clarify" and r["easier"])]
    _, manifest = build(pp2, rows)
    gaps = manifest["coverage_gaps"]
    assert gaps["train"] == {"move_reply_type": [], "category_move": []}
    assert gaps["validation"]["move_reply_type"] == ["8-10 / clarify (easier) / after dont_know"]


@pytest.mark.parametrize("split", ["train", "validation"])
def test_build_refuses_a_split_over_the_trusted_adult_cap_and_names_it(pp2, split):
    scenarios = {s["scenario_id"] for s in pp2["plan"]["scenarios"] if s["role"] == split}
    rows = [dict(r) for r in pp2["rows"]]
    in_split = [r for r in rows if r["scenario_id"] in scenarios]
    for r in in_split:
        r["completion"] = GOOD
    for r in in_split[: len(in_split) * 5 // 8]:   # 62.5%: whole rows in both splits
        r["completion"] = ADULT
    message = refused(pp2, rows)
    assert f"{split}: trusted-adult share 0.625" in message and "cap 0.5" in message


def test_build_accepts_a_split_exactly_at_the_cap(pp2):
    _, manifest = build(pp2, _rows(pp2["plan"], adult_every=2))
    assert manifest["trusted_adult"]["share"] == {"train": 0.5, "validation": 0.5}
    assert manifest["trusted_adult"]["cap"] == 0.5


def test_build_refuses_rows_on_held_out_scenarios_and_names_them(pp2):
    held = [s for s in pp2["plan"]["scenarios"] if s["role"] in ("output_study", "classifier_test")][:2]
    rows = pp2["rows"] + [_row(9001, held[0], "clarify", False, "explanation"),
                          _row(9002, held[1], "consequences", True, "dont_know")]
    message = refused(pp2, rows)
    assert "held-out" in message and "P2R9001" in message and "P2R9002" in message


@pytest.mark.parametrize("edit, problem", [
    ({"move": "clarify", "easier": True, "reply_type": "stated_emotion"}, "not a combination the controller asks for"),
    ({"move": "consequences", "easier": False, "reply_type": "own_safer_plan"},
     "not a combination the controller asks for"),
    ({"move": "alternatives", "easier": False, "reply_type": "refusal"}, "not a combination the controller asks for"),
    # alternatives is always the third line: after "I don't know" it, or its easier form, would be a fourth
    ({"move": "alternatives", "easier": False, "reply_type": "dont_know"}, "not a combination the controller asks for"),
    ({"move": "alternatives", "easier": True, "reply_type": "dont_know"}, "not a combination the controller asks for"),
    ({"split": "test"}, "split 'test'"),
    ({"move": "summarise"}, "move 'summarise'"),
    ({"age_band": "16-18"}, "age band '16-18'"),
    ({"easier": "yes"}, "easier 'yes'"),
    ({"age_band": "14-15"}, "does not match its scenario"),
])
def test_build_refuses_rows_with_invalid_or_unreachable_labels(pp2, edit, problem):
    rows = [dict(r) for r in pp2["rows"]]
    target = next(r for r in rows if r["age_band"] == "8-10")
    target.update(edit)
    message = refused(pp2, rows)
    assert target["example_id"] in message and problem in message


def test_build_refuses_pending_rows_and_failing_completions(pp2):
    rows = [dict(r) for r in pp2["rows"]]
    rows[3]["review"] = {**rows[3]["review"], "status": "pending"}
    assert rows[3]["example_id"] in refused(pp2, rows)
    rows = [dict(r) for r in pp2["rows"]]
    rows[5]["completion"] = "It makes sense to feel furious. What could help?"   # names an unstated emotion
    message = refused(pp2, rows)
    assert rows[5]["example_id"] in message and "unstated emotion: furious" in message


def test_build_manifest_records_shares_closing_strata_gaps_seeds_and_hashes(pp2):
    _, manifest = build(pp2)
    assert manifest["dataset"] == "inquire-pp2"
    assert manifest["inputs"]["draw"]["sha256"] == hashlib.sha256(pp2["plan_path"].read_bytes()).hexdigest()
    assert manifest["inputs"]["drafts"]["sha256"] == hashlib.sha256(pp2["drafts"].read_bytes()).hexdigest()
    assert set(manifest["outputs"]) == {"train.jsonl", "validation.jsonl"}
    assert manifest["seeds"] == {"scenario_draw": 20261104}
    # every 4th row (rows 4, 8, ...) mentions a trusted adult; training is rows 1-144
    assert manifest["trusted_adult"]["share"] == {"train": 0.25, "validation": 0.25}
    assert manifest["most_repeated_closing"]["train"] == {"text": GOOD, "count": 108}
    counts = manifest["counts"]["train"]
    assert counts["rows"] == 144 and counts["scenarios"] == 48
    assert counts["move_reply_type"]["8-10 / clarify (easier) / after dont_know"] == 2   # scenarios 6 and 12 of 16
    assert counts["category_move"]["14-15 / hate_speech / alternatives"] == 2
    assert manifest["held_out"] == {r: sorted(pp2["plan"]["roles"][r]) for r in ("output_study", "classifier_test")}
    assert len(manifest["strata"]["move_reply_type"]) == 14 and len(manifest["strata"]["category_move"]) == 24
    assert "alternatives (easier) / after dont_know" not in manifest["strata"]["move_reply_type"]


def test_build_is_byte_identical_and_never_overwrites_a_recorded_dataset(pp2):
    build(pp2, out=pp2["tmp"] / "a")
    build(pp2, out=pp2["tmp"] / "b")
    for name in ("train.jsonl", "validation.jsonl", "manifest.json"):
        assert (pp2["tmp"] / "a" / name).read_bytes() == (pp2["tmp"] / "b" / name).read_bytes()
    with pytest.raises(SystemExit) as e:
        build(pp2, out=pp2["tmp"] / "a")
    assert "already holds a dataset" in str(e.value.code)


def test_review_shows_a_pp2_rows_move_band_and_reply(pp2, monkeypatch, capsys):
    import io
    rows = [dict(r, review={**r["review"], "status": "pending"}, completion=None) for r in pp2["rows"][:1]]
    _write_jsonl(pp2["drafts"], rows)
    monkeypatch.setattr(sys, "stdin", io.StringIO("q\n"))
    ft.main(["review", "--drafts", str(pp2["drafts"])])
    shown = capsys.readouterr().out
    r = rows[0]
    assert f"move: {r['move']}" in shown and f"band: {r['age_band']}" in shown
    assert f"Child's latest reply: {r['child_reply']}" in shown
    assert f"Previous educator message: {r['previous_message']}" in shown


def test_build_refuses_a_kept_row_without_a_completion_and_a_duplicated_id(pp2):
    rows = [dict(r) for r in pp2["rows"]]
    rows[2]["completion"] = None
    assert f"{rows[2]['example_id']}: no completion" in refused(pp2, rows)
    rows = [dict(r) for r in pp2["rows"]] + [dict(pp2["rows"][7])]
    assert f"{pp2['rows'][7]['example_id']}: appears 2 times" in refused(pp2, rows)


def test_build_refuses_a_file_that_is_not_a_pp2_draw(pp2):
    pp2["plan_path"].write_text(json.dumps({"dataset": "inquire-finetune", "scenarios": []}), encoding="utf-8")
    assert "not a PP2 scenario draw" in refused(pp2, pp2["rows"])

"""
Tests for the simulated children (ticket pp2-validation/06): `plan`, `run` and
`finish` in simulate_children.py, driven end to end through main(argv) with a
fake child model (no network, no LM Studio). The sessions go through the real
FSMController.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inquire_finetune as ft  # noqa: E402
import simulate_children as sc  # noqa: E402
from src.schemas import _validate_against_schema  # noqa: E402

BANDS = ("8-10", "11-13", "14-15")
ARCHETYPES = ("calm_reflective", "defensive", "distressed", "withdrawn")
REPLIES = {"calm_reflective": "I was just watching stuff and it popped up, I guess I could close it",
           "defensive": "it was nothing, not my fault",
           "distressed": "it made me really scared and I can't stop thinking about it",
           "withdrawn": "idk"}


@pytest.fixture
def draw(tmp_path):
    ft.main(["pp2-draw", "--out", str(tmp_path / "draw")])
    path = tmp_path / "draw" / "scenario_draw.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def plan(tmp_path, draw_path, *extra, out="sim"):
    sc.main(["plan", "--draw", str(draw_path), "--out", str(tmp_path / out), *extra])
    return json.loads((tmp_path / out / "children.json").read_text(encoding="utf-8"))


class FakeChild:
    """Answers as the archetype named in the child's prompt, and records every chat it was sent."""

    def __init__(self, fail_after=None, reply=None):
        self.calls, self.fail_after, self.reply = [], fail_after, reply

    def __call__(self, messages):
        self.calls.append(messages)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("403 quota exhausted")
        prompt = messages[0]["content"]
        if callable(self.reply):
            return self.reply(prompt)
        if self.reply is not None:
            return self.reply
        return next(text for archetype, text in REPLIES.items() if sc.ARCHETYPES[archetype][0] in prompt)


def run(tmp_path, generator="gemini", ask=None, *extra, out="sim"):
    ask = ask or FakeChild()
    sc.main(["run", "--plan", str(tmp_path / out / "children.json"), "--generator", generator,
             "--model", f"{generator}-test-model", "--backend", "aistudio", *extra], ask=ask)
    return ask


def finish(tmp_path, draw_path, out="sim"):
    sc.main(["finish", "--dir", str(tmp_path / out), "--draw", str(draw_path)])


def sessions(tmp_path, out="sim"):
    path = tmp_path / out / "sessions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def run_record(tmp_path, generator, out="sim"):
    return json.loads((tmp_path / out / f"run_{generator}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- plan
def test_plan_puts_each_classifier_test_scenario_in_the_test_pool_and_training_scenarios_in_train_dev(tmp_path, draw):
    path, d = draw
    children = plan(tmp_path, path)["children"]
    test = [c for c in children if c["pool"] == "classifier_test"]
    train_dev = [c for c in children if c["pool"] == "train_dev"]
    assert sorted(c["scenario_id"] for c in test) == sorted(d["roles"]["classifier_test"])
    assert sorted(c["scenario_id"] for c in train_dev) == sorted(d["roles"]["train"] + d["roles"]["validation"])
    assert not {c["scenario_id"] for c in children} & set(d["roles"]["output_study"])
    assert Counter(c["age_band"] for c in test) == {band: 40 for band in BANDS}
    assert Counter(c["age_band"] for c in train_dev) == {band: 24 for band in BANDS}
    # ticket 14 needs train and dev apart
    assert Counter(c["scenario_role"] for c in train_dev) == {"train": 48, "validation": 24}
    # held-out children never appear in the train/dev pool
    assert not {c["child_id"] for c in test} & {c["child_id"] for c in train_dev}
    assert len({c["session_id"] for c in children}) == len(children) == 192


def test_plan_balances_archetypes_and_gives_about_one_in_five_children_to_gemma(tmp_path, draw):
    children = plan(tmp_path, draw[0])["children"]
    for band in BANDS:
        test = [c for c in children if c["pool"] == "classifier_test" and c["age_band"] == band]
        assert Counter(c["archetype"] for c in test) == {a: 10 for a in ARCHETYPES}
        assert Counter(c["generator"] for c in test) == {"gemini": 32, "gemma": 8}
        assert Counter(c["archetype"] for c in test if c["generator"] == "gemma") == {a: 2 for a in ARCHETYPES}
        train_dev = [c for c in children if c["pool"] == "train_dev" and c["age_band"] == band]
        assert Counter(c["archetype"] for c in train_dev) == {a: 6 for a in ARCHETYPES}
        assert Counter(c["generator"] for c in train_dev) == {"gemini": 19, "gemma": 5}
    # children who share a band, archetype and category are not all the same child
    assert len({c["interest"] for c in children}) > 1


def test_plan_is_reproducible_with_its_seed_and_never_overwritten(tmp_path, draw):
    a = plan(tmp_path, draw[0], out="a")
    b = plan(tmp_path, draw[0], out="b")
    c = plan(tmp_path, draw[0], "--seed", "7", out="c")
    assert a == b and a["seed"] == sc.PLAN_SEED and c["seed"] == 7
    assert [x["archetype"] for x in a["children"]] != [x["archetype"] for x in c["children"]]
    assert a["draw"]["sha256"] and a["draw"]["seed"] == draw[1]["seed"]
    with pytest.raises(SystemExit) as e:
        plan(tmp_path, draw[0], out="a")
    assert "already holds a plan" in str(e.value.code)


def test_plan_refuses_a_file_that_is_not_a_pp2_draw(tmp_path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"dataset": "inquire-finetune"}), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        plan(tmp_path, other)
    assert "not a PP2 scenario draw" in str(e.value.code)


# ---------------------------------------------------------------- the child's chat
def test_each_band_gets_its_own_child_prompt_and_the_chat_alternates_educator_and_child(tmp_path, draw):
    children = plan(tmp_path, draw[0])["children"]
    child = run(tmp_path, "gemma")
    prompts = {}
    for call in child.calls:
        prompt = call[0]["content"]
        band = next(b for b in BANDS if f"aged {b.replace('-', ' to ')}" in prompt)
        prompts.setdefault(band, set()).add(prompt.split(" You like ")[0])
        # the educator speaks as the user and the child as the assistant, ending on the educator's line
        assert [m["role"] for m in call] == ["system"] + ["user", "assistant"] * ((len(call) - 2) // 2) + ["user"]
        assert not any(c["risk_category"] in prompt for c in children)   # plain words, never the category code
    assert set(prompts) == set(BANDS) and len({p for ps in prompts.values() for p in ps}) == 3


def test_fold_system_sends_the_childs_prompt_in_the_first_user_message(tmp_path, draw):
    plan(tmp_path, draw[0], out="plain")
    plan(tmp_path, draw[0], out="folded")
    plain = run(tmp_path, "gemma", None, "--limit", "1", out="plain")
    folded = run(tmp_path, "gemma", None, "--limit", "1", "--fold-system", out="folded")
    assert folded.calls[0][0] == {"role": "user", "content": f"{plain.calls[0][0]['content']}\n\n"
                                                             f"{plain.calls[0][1]['content']}"}
    assert "system" not in {m["role"] for call in folded.calls for m in call}
    assert run_record(tmp_path, "gemma", out="folded")["fold_system"] is True


# ---------------------------------------------------------------- run
def test_run_simulates_only_its_generators_children_through_the_real_controller(tmp_path, draw):
    children = plan(tmp_path, draw[0])["children"]
    run(tmp_path, "gemma")
    done = sessions(tmp_path)
    assert {s["session_id"] for s in done} == {c["session_id"] for c in children if c["generator"] == "gemma"}
    by_id = {c["session_id"]: c for c in children}
    for s in done:
        child = by_id[s["session_id"]]
        for field in ("pool", "age_band", "generator", "archetype", "risk_category", "scenario_id", "scenario_role"):
            assert s[field] == child[field], field
        assert s["model_id"] == "gemma-test-model"
        states = [t["state"] for t in s["turns"]]
        assert states[0] == "INTERCEPT" and states[-2:] == ["EVALUATE", "CONTRACT"]
        assert {t["age_band"] for t in s["turns"]} == {child["age_band"]}   # lines worded for the child's band
        replies = [t["child_response"] for t in s["turns"] if t["child_response"] is not None]
        assert replies and set(replies) == {REPLIES[child["archetype"]]}
        _validate_against_schema(s["record"], "comp3_to_comp4.schema.json")


def test_raw_replies_stay_out_of_the_component4_records(tmp_path, draw):
    plan(tmp_path, draw[0])
    run(tmp_path, "gemma")
    for s in sessions(tmp_path):
        record = json.dumps(s["record"])
        assert not any(t["child_response"] and t["child_response"] in record for t in s["turns"])


def test_run_cleans_the_reply_and_keeps_the_prompt_and_every_raw_output(tmp_path, draw):
    plan(tmp_path, draw[0])
    raw = 'Child: "idk maybe"\nAs an AI I would add more.'
    run(tmp_path, "gemma", FakeChild(reply=raw), "--limit", "1")
    s = sessions(tmp_path)[0]
    assert {t["child_response"] for t in s["turns"] if t["child_response"]} == {"idk maybe"}
    assert s["child_outputs"][0] == [raw]
    assert s["child_prompt"].startswith("You are")


def test_a_child_with_no_usable_reply_is_skipped_and_recorded_and_the_run_goes_on(tmp_path, draw, capsys):
    children = plan(tmp_path, draw[0])["children"]
    gemma = [c for c in children if c["generator"] == "gemma"]
    stuck = gemma[0]
    stuck_prompt = sc.ARCHETYPES[stuck["archetype"]][0]
    # the stuck child's archetype answers out of character; every other archetype answers normally
    reply = lambda prompt: "As an AI I cannot do that." if stuck_prompt in prompt else next(  # noqa: E731
        text for archetype, text in REPLIES.items() if sc.ARCHETYPES[archetype][0] in prompt)
    run(tmp_path, "gemma", FakeChild(reply=reply))
    done = {s["session_id"] for s in sessions(tmp_path)}
    assert stuck["session_id"] not in done
    assert done == {c["session_id"] for c in gemma if c["archetype"] != stuck["archetype"]}
    failed = run_record(tmp_path, "gemma")["children_without_a_usable_reply"]
    assert any(line.startswith(stuck["session_id"]) for line in failed)
    assert "no usable reply" in capsys.readouterr().out


def test_a_code_error_is_raised_not_hidden_as_a_resumable_stop(tmp_path, draw, monkeypatch):
    plan(tmp_path, draw[0])
    monkeypatch.setattr(sc, "session_record", lambda out, timestamp: {}["missing"])
    with pytest.raises(KeyError):
        run(tmp_path, "gemma")


def test_run_stops_on_a_model_error_keeps_finished_sessions_and_resumes(tmp_path, draw, capsys):
    children = plan(tmp_path, draw[0])["children"]
    gemma = [c["session_id"] for c in children if c["generator"] == "gemma"]
    run(tmp_path, "gemma", FakeChild(fail_after=9))   # dies inside the third or fourth session
    first = sessions(tmp_path)
    assert 0 < len(first) < len(gemma)
    out = capsys.readouterr().out
    assert "rerun the same command to resume" in out and f"{len(gemma) - len(first)} still to simulate" in out
    run(tmp_path, "gemma")
    second = sessions(tmp_path)
    assert sorted(s["session_id"] for s in second) == sorted(gemma)
    assert second[:len(first)] == first   # finished sessions are never redone or rewritten
    assert "0 still to simulate" in capsys.readouterr().out


def test_the_count_left_is_per_generator(tmp_path, draw, capsys):
    plan(tmp_path, draw[0])
    run(tmp_path, "gemini", None, "--limit", "5")
    capsys.readouterr()
    run(tmp_path, "gemma")
    out = capsys.readouterr().out
    assert "gemma: wrote 39 sessions" in out and "; 0 still to simulate" in out   # not minus Gemini's five
    run(tmp_path, "gemma")
    assert "gemma: wrote 0 sessions" in capsys.readouterr().out


def test_a_torn_last_line_is_cut_and_the_run_resumes(tmp_path, draw):
    plan(tmp_path, draw[0])
    run(tmp_path, "gemma", None, "--limit", "2")
    path = tmp_path / "sim" / "sessions.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{"session_id": "SIM-C', encoding="utf-8")   # died mid-write
    run(tmp_path, "gemma", None, "--limit", "1")
    assert len(sessions(tmp_path)) == 3


def test_a_trial_takes_train_dev_children_one_per_band(tmp_path, draw):
    plan(tmp_path, draw[0])
    run(tmp_path, "gemini", None, "--pool", "train_dev", "--per-band", "1")
    done = sessions(tmp_path)
    assert {s["pool"] for s in done} == {"train_dev"}   # never tune the prompt on held-out children
    assert sorted(s["age_band"] for s in done) == sorted(BANDS)


def test_run_records_the_model_settings_seed_and_runtime(tmp_path, draw):
    plan(tmp_path, draw[0])
    run(tmp_path, "gemma", None, "--limit", "3")
    record = run_record(tmp_path, "gemma")
    assert record["model"] == "gemma-test-model" and record["backend"] == "aistudio"
    assert record["settings"] == {"temperature": sc.CHILD_TEMPERATURE, "max_tokens": sc.CHILD_MAX_TOKENS}
    assert record["child_seed"].startswith("not recorded")
    assert record["served_models"] == "not recorded"   # the fake child reports none
    assert record["plan_seed"] == sc.PLAN_SEED and record["educator"] == sc.EDUCATOR_SETUP
    assert set(record["runtime"]) == {"python", "platform", "openai"}
    assert record["sessions_written"] == 3


# ---------------------------------------------------------------- finish
def test_finish_refuses_until_every_planned_session_is_simulated(tmp_path, draw):
    plan(tmp_path, draw[0])
    run(tmp_path, "gemma")
    with pytest.raises(SystemExit) as e:
        finish(tmp_path, draw[0])
    assert "finish refused" in str(e.value.code) and "not simulated yet" in str(e.value.code)


def test_finish_refuses_a_draw_the_plan_was_not_made_from(tmp_path, draw):
    plan(tmp_path, draw[0])
    run(tmp_path, "gemma")
    run(tmp_path, "gemini")
    ft.main(["pp2-draw", "--seed", "7", "--out", str(tmp_path / "other")])
    with pytest.raises(SystemExit) as e:
        finish(tmp_path, tmp_path / "other" / "scenario_draw.json")
    assert "not the draw the plan was made from" in str(e.value.code)


@pytest.fixture
def finished(tmp_path, draw):
    plan(tmp_path, draw[0])
    run(tmp_path, "gemma")
    run(tmp_path, "gemini")
    finish(tmp_path, draw[0])
    return tmp_path / "sim"


def test_finish_writes_one_file_per_pool_that_the_label_sheets_accept(finished, tmp_path):
    test = [json.loads(line) for line in (finished / "classifier_test_sessions.jsonl").read_text(
        encoding="utf-8").splitlines()]
    train_dev = [json.loads(line) for line in (finished / "train_dev_sessions.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(test) == 120 and {s["pool"] for s in test} == {"classifier_test"}
    assert len(train_dev) == 72 and {s["pool"] for s in train_dev} == {"train_dev"}
    ft.main(["label-sheets", "--sessions", str(finished / "classifier_test_sessions.jsonl"),
             "--out", str(tmp_path / "labels")])
    ft.main(["label-sheets", "--owner", "--sessions", str(finished / "train_dev_sessions.jsonl"),
             "--out", str(tmp_path / "owner_labels")])
    assert (tmp_path / "labels" / "key.json").exists() and (tmp_path / "owner_labels" / "key.json").exists()


def test_finish_manifest_counts_pools_bands_generators_and_archetypes(finished):
    manifest = json.loads((finished / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["classifier_test"]["8-10"] == {"gemini": 32, "gemma": 8}
    assert manifest["counts"]["train_dev"]["14-15"] == {"gemini": 19, "gemma": 5}
    assert manifest["archetypes"]["classifier_test"]["11-13"] == {a: 10 for a in ARCHETYPES}
    assert manifest["train_dev_roles"] == {"train": 48, "validation": 24}
    assert manifest["models"] == {"gemini": ["gemini-test-model"], "gemma": ["gemma-test-model"]}
    assert set(manifest["checks"].values()) == {0}
    assert set(manifest["outputs"]) == {"classifier_test_sessions.jsonl", "train_dev_sessions.jsonl"}


def test_finish_refuses_a_child_in_both_pools(tmp_path, draw):
    plan(tmp_path, draw[0])
    path = tmp_path / "sim" / "children.json"
    p = json.loads(path.read_text(encoding="utf-8"))
    test = next(c for c in p["children"] if c["pool"] == "classifier_test")
    train_dev = next(c for c in p["children"] if c["pool"] == "train_dev")
    train_dev["child_id"] = test["child_id"]
    path.write_text(json.dumps(p), encoding="utf-8")
    run(tmp_path, "gemma")
    run(tmp_path, "gemini")
    with pytest.raises(SystemExit) as e:
        finish(tmp_path, draw[0])
    assert test["child_id"] in str(e.value.code) and "both pools" in str(e.value.code)


def test_finish_refuses_a_session_whose_scenario_is_held_out_for_another_use(tmp_path, draw):
    _, d = draw
    plan(tmp_path, draw[0])
    path = tmp_path / "sim" / "children.json"
    p = json.loads(path.read_text(encoding="utf-8"))
    moved = next(c for c in p["children"] if c["pool"] == "train_dev")
    moved["scenario_id"] = d["roles"]["output_study"][0]
    path.write_text(json.dumps(p), encoding="utf-8")
    run(tmp_path, "gemma")
    run(tmp_path, "gemini")
    with pytest.raises(SystemExit) as e:
        finish(tmp_path, draw[0])
    assert f"{moved['session_id']}: scenario {moved['scenario_id']} is output_study" in str(e.value.code)

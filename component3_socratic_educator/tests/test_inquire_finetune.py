"""
Tests for inquire_finetune.py: the shared format / stated-emotion check
(ADR 0003), called directly, and `review`, driven end to end through
main(argv) on tmp_path copies with scripted stdin. No model is loaded.
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inquire_finetune as ft  # noqa: E402

SEED_FILE = PROJECT_ROOT / "data" / "educator_seed.jsonl"
DRAFTS_DIR = PROJECT_ROOT / "evidence" / "inquire_drafts_2026-10-08"
DRAFTS_FILE = DRAFTS_DIR / "inquire_drafts.jsonl"

CHILD = "I'm annoyed because I was right in the middle of that show."


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------- format check
@pytest.mark.parametrize("text, rule", [
    ("What could you do next?", None),
    ("That makes sense. What could you do next?", None),
    ("That makes sense. You were busy. What could you do next?", "sentences"),
    ("That makes sense.", "question"),
    ("What could you do? Let's decide together.", "question"),
    ("Is that okay? What could you do next?", "question"),
    ("What could you do next?  ", None),  # trailing whitespace is not content
    (" ".join(["word"] * 29) + " now?", None),          # 30 words
    (" ".join(["word"] * 30) + " now?", "words"),       # 31 words
    ('You said "stop". What could you do next?', "quotes"),
    ("You said “stop”. What could you do next?", "quotes"),
    ("'Stop' is a good word. What could you do next?", "quotes"),
    ("‘Stop’ is a good word. What could you do next?", "quotes"),
    ("Educator: What could you do next?", "meta"),
    ("Response: What could you do next?", "meta"),
    ("Here is a response: what could you do next?", "meta"),
    ("As an AI, what could you do next?", "meta"),
])
def test_format_check_rules(text, rule):
    result = ft.format_check(text, CHILD)
    failed = {name for name, ok in result.rules.items() if not ok}
    assert failed == ({rule} if rule else set()), result.failures()
    assert result.format_ok is (rule is None)


def test_apostrophes_are_not_quotes():
    assert ft.format_check("It’s okay. What’s one safe next step?", CHILD).format_ok
    assert ft.format_check("It's okay. What's one safe next step?", CHILD).format_ok


@pytest.mark.parametrize("bad", ft.FORBIDDEN_SUBSTRINGS)
def test_every_controller_forbidden_substring_fails(bad):
    result = ft.format_check(f"Why would {bad.upper()} matter here?", CHILD)
    assert result.rules["forbidden"] is False
    assert bad in result.forbidden


def test_forbidden_list_is_the_controller_grammar_list_not_a_copy():
    from src import grammar_decoder
    assert ft.FORBIDDEN_SUBSTRINGS is grammar_decoder._FORBIDDEN_SUBSTRINGS


# ---------------------------------------------------------------- stated emotion
def test_emotion_the_child_did_not_state_is_flagged():
    # INQ001: the child said "annoyed"; the seed completion says "frustration".
    result = ft.format_check(
        "What might help you manage that frustration before choosing what to do next?", CHILD)
    assert result.unstated_emotions == ["frustration"]
    assert result.emotion_ok is False
    assert result.format_ok is True  # reported separately from the format rules
    assert result.ok is False


def test_emotion_the_child_stated_verbatim_passes():
    result = ft.format_check("It makes sense to feel annoyed. What could help right now?", CHILD)
    assert result.unstated_emotions == []
    assert result.emotion_ok and result.ok


def test_emotion_match_is_whole_word_and_case_insensitive():
    child = "Honestly I'm SCARED and kind of grossed out."
    assert ft.format_check("Feeling scared and grossed out is okay. Who can help?", child).emotion_ok
    # "scary" is not the word the child used
    assert ft.unstated_emotions("That sounds scary. Who can help?", child) == ["scary"]
    # a substring of another word is not an emotion ("sadly" is not "sad")
    assert ft.unstated_emotions("Sadly that happens. Who can help?", "ok") == []


def test_curly_apostrophes_in_child_reply_still_match():
    assert ft.format_check("Not wanting to talk is okay. Who can help?", "I don’t want to talk.").ok


# ---------------------------------------------------------------- real data
def _inquire_seeds():
    return [r for r in load_jsonl(SEED_FILE) if r["example_id"].startswith("INQ")]


def _child(prompt_text):
    return ft.parse_inquire_prompt(prompt_text)["child"]


# Found when the check was written (2026-10-09). INQ001 was already known; INQ005
# and INQ022 paraphrase the child's emotion too ("creeped out" -> "shaken up",
# "scared" -> "frightening"). All three need a rewrite before training.
KNOWN_SEED_FAILURES = {
    "INQ001": ["unstated emotion: frustration"],
    "INQ005": ["unstated emotion: shaken"],
    "INQ022": ["unstated emotion: frightening"],
}


def test_inquire_seeds_fail_only_on_the_documented_cases():
    """INQ001 is the known emotion violation; any other real failure is listed above, not hidden."""
    failures = {}
    for row in _inquire_seeds():
        result = ft.format_check(row["completion"][0]["content"], _child(row["prompt"][0]["content"]))
        if not result.ok:
            failures[row["example_id"]] = result.failures()
    assert "INQ001" in failures
    assert failures == KNOWN_SEED_FAILURES


def test_all_300_drafts_pass_both_checks():
    """Reproduces the drafts README: format 300/300, 0 emotion violations."""
    rows = load_jsonl(DRAFTS_FILE)
    drafts = [(r["example_id"], d, _child(r["prompt"][0]["content"])) for r in rows for d in r["drafts"]]
    assert len(drafts) == 300
    bad = {(eid, d): ft.format_check(d, child).failures() for eid, d, child in drafts
           if not ft.format_check(d, child).ok}
    assert bad == {}


# ---------------------------------------------------------------- review
def _fixture_rows(n_train=6, n_test=2):
    """Real drafted rows reset to pending, whatever the owner's review file says."""
    rows = load_jsonl(DRAFTS_FILE)
    for r in rows:
        r["completion"] = None
        r["review"] = {"status": "pending", "chosen_draft": None, "rewritten": False, "notes": ""}
    return ([r for r in rows if r["split"] == "train_or_val"][:n_train]
            + [r for r in rows if r["split"] == "test"][:n_test])


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def drafts(tmp_path):
    path = tmp_path / "inquire_drafts.jsonl"
    _write_jsonl(path, _fixture_rows())
    return path


def run_review(monkeypatch, capsys, path, keys, *extra):
    import io
    monkeypatch.setattr(sys, "stdin", io.StringIO(keys))
    ft.main(["review", "--drafts", str(path), *extra])
    return capsys.readouterr().out


def shown_draft(out, example_id, letter):
    """The text shown as draft A or B for a row in the review output."""
    block = out.split(f"[{example_id}]", 1)[1]
    line = next(l for l in block.splitlines() if l.startswith(f"  {letter}: "))
    return line[len(f"  {letter}: "):].replace("[[", "").replace("]]", "")


def test_review_approve_a_then_b(monkeypatch, capsys, drafts):
    before = load_jsonl(drafts)
    out = run_review(monkeypatch, capsys, drafts, "a\nb\nq\n")
    first, second = load_jsonl(drafts)[:2]
    assert first["completion"] == shown_draft(out, first["example_id"], "A")
    assert second["completion"] == shown_draft(out, second["example_id"], "B")
    for row, orig in ((first, before[0]), (second, before[1])):
        assert row["review"]["status"] == "approved" and row["review"]["rewritten"] is False
        assert row["completion"] == orig["drafts"][row["review"]["chosen_draft"]]
        assert row["review"]["reviewed_at"]
    assert all(r["review"]["status"] == "pending" for r in load_jsonl(drafts)[2:])


def test_review_valid_rewrite_starts_from_chosen_draft(monkeypatch, capsys, drafts):
    text = "What is one safe thing you could do next?"
    out = run_review(monkeypatch, capsys, drafts, f"r\nb\n{text}\nq\n")
    row = load_jsonl(drafts)[0]
    assert row["review"] | {"reviewed_at": None} == {
        "status": "rewritten", "chosen_draft": row["drafts"].index(shown_draft(out, row["example_id"], "B")),
        "rewritten": True, "notes": "", "reviewed_at": None}
    assert row["completion"] == text


def test_review_invalid_rewrite_is_refused_and_reprompted(monkeypatch, capsys, drafts):
    bad = "You should feel furious. That is all."
    good = "What is one safe thing you could do next?"
    out = run_review(monkeypatch, capsys, drafts, f"r\na\n{bad}\n{good}\nq\n")
    assert "refused" in out and "question" in out and "unstated emotion: furious" in out
    assert load_jsonl(drafts)[0]["completion"] == good


def test_review_reject_with_note_and_note_on_approval(monkeypatch, capsys, drafts):
    run_review(monkeypatch, capsys, drafts, "x\nchild reply makes no sense\nn\nB reads warmer\nb\nq\n")
    first, second = load_jsonl(drafts)[:2]
    assert first["review"]["status"] == "rejected"
    assert first["review"]["notes"] == "child reply makes no sense"
    assert first["completion"] is None and first["review"]["chosen_draft"] is None
    assert second["review"]["status"] == "approved" and second["review"]["notes"] == "B reads warmer"


def test_review_skip_leaves_row_pending_and_rerun_resumes_there(monkeypatch, capsys, drafts):
    ids = [r["example_id"] for r in load_jsonl(drafts)]
    run_review(monkeypatch, capsys, drafts, "s\na\nq\n")
    rows = load_jsonl(drafts)
    assert rows[0]["review"]["status"] == "pending" and rows[1]["review"]["status"] == "approved"
    out = run_review(monkeypatch, capsys, drafts, "q\n")
    assert out.index(f"[{ids[0]}]") >= 0 and f"[{ids[1]}]" not in out
    out = run_review(monkeypatch, capsys, drafts, "a\nq\n")
    assert load_jsonl(drafts)[0]["review"]["status"] == "approved"
    assert f"[{ids[2]}]" in out  # moved on past the approved row


def test_review_saves_each_decision_before_the_next_row(monkeypatch, capsys, drafts):
    class Killed:
        """stdin that answers once, then the process 'dies'."""
        def __init__(self):
            self.lines = iter(["a\n"])

        def readline(self):
            try:
                return next(self.lines)
            except StopIteration:
                raise KeyboardInterrupt

    monkeypatch.setattr(sys, "stdin", Killed())
    with pytest.raises(KeyboardInterrupt):
        ft.main(["review", "--drafts", str(drafts)])
    assert load_jsonl(drafts)[0]["review"]["status"] == "approved"


def test_review_never_changes_drafts_or_provenance(monkeypatch, capsys, drafts):
    def frozen(rows):
        return [json.dumps([r["drafts"], r["provenance"]], ensure_ascii=False) for r in rows]
    before = frozen(load_jsonl(drafts))
    run_review(monkeypatch, capsys, drafts, "a\nr\na\nWhat is one safe thing you could do next?\nx\n\nb\nk\nk\n")
    assert frozen(load_jsonl(drafts)) == before


def test_review_hides_drafter_and_varies_which_draft_is_a(monkeypatch, capsys, tmp_path):
    path = tmp_path / "inquire_drafts.jsonl"
    _write_jsonl(path, _fixture_rows(n_train=12, n_test=0))
    out = run_review(monkeypatch, capsys, path, "a\n" * 12)
    assert "gemini" not in out.lower() and "claude" not in out.lower()
    assert len({r["review"]["chosen_draft"] for r in load_jsonl(path)}) == 2


def test_review_shows_checks_and_highlights_trusted_adults(monkeypatch, capsys, drafts):
    out = run_review(monkeypatch, capsys, drafts, "q\n")
    assert "format ok" in out and "emotion ok" in out
    assert "[[trusted adult]]" in out.lower()  # INQ030's drafts both name a trusted adult


def test_review_test_rows_offer_only_keep_or_reject(monkeypatch, capsys, drafts):
    out = run_review(monkeypatch, capsys, drafts, "a\nk\nx\nodd reply\n", "--split", "test")
    test_rows = [r for r in load_jsonl(drafts) if r["split"] == "test"]
    menus = [line for line in out.splitlines() if "[k] keep" in line]
    assert menus and all("[a]" not in m and "[b]" not in m and "[r]" not in m for m in menus)
    assert "unknown key" in out  # "a" is not an option for a test row
    assert test_rows[0]["review"]["status"] == "approved" and test_rows[0]["completion"] is None
    assert test_rows[1]["review"]["status"] == "rejected" and test_rows[1]["review"]["notes"] == "odd reply"
    assert all(r["review"]["status"] == "pending" for r in load_jsonl(drafts) if r["split"] == "train_or_val")


def test_review_default_order_is_training_rows_then_test_rows(monkeypatch, capsys, tmp_path):
    rows = _fixture_rows(n_train=1, n_test=1)
    path = tmp_path / "inquire_drafts.jsonl"
    _write_jsonl(path, rows[::-1])  # test row first in the file
    out = run_review(monkeypatch, capsys, path, "a\nk\n")
    assert out.index(f"[{rows[0]['example_id']}]") < out.index(f"[{rows[1]['example_id']}]")


def test_review_reports_progress(monkeypatch, capsys, drafts):
    out = run_review(monkeypatch, capsys, drafts, "a\nr\na\nWhat is one safe thing you could do next?\nx\n\nq\n")
    assert "train_or_val: 3 reviewed, 3 pending (1 approved as drafted, 1 rewritten, 1 rejected)" in out
    assert "test: 0 reviewed, 2 pending" in out


# ---------------------------------------------------------------- build
SCENARIOS_FILE = DRAFTS_DIR / "scenarios.json"
SEED_INPUTS_DIR = PROJECT_ROOT / "evidence" / "inquire_seed_inputs"
OVERRIDES_FILE = SEED_INPUTS_DIR / "seed_overrides.json"
LABELS_FILE = SEED_INPUTS_DIR / "seed_labels.json"
FROZEN_TEST = json.loads(SCENARIOS_FILE.read_text(encoding="utf-8"))["test_scenario_ids"]
DEVELOPMENT_EXAMPLES = {"INQ001", "INQ026", "INQ027", "INQ028"}


def _sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reviewed_rows():
    """All 150 drafted rows with a fixed review (draft 0 approved; test prompts kept), whatever the owner's file says."""
    rows = load_jsonl(DRAFTS_FILE)
    for r in rows:
        test = r["split"] == "test"
        r["completion"] = None if test else r["drafts"][0]
        r["review"] = {"status": "approved", "chosen_draft": None if test else 0, "rewritten": False,
                       "notes": "", "reviewed_at": "2026-10-10T00:00:00+00:00"}
    return rows


@pytest.fixture
def inputs(tmp_path):
    """tmp_path copies of every build input, so tests can break one at a time."""
    import shutil
    d = tmp_path / "inputs"
    d.mkdir()
    _write_jsonl(d / "inquire_drafts.jsonl", _reviewed_rows())
    for src in (SEED_FILE, SCENARIOS_FILE, OVERRIDES_FILE, LABELS_FILE):
        shutil.copy(src, d / src.name)
    return d


def run_build(inputs, out):
    ft.main(["build", "--drafts", str(inputs / "inquire_drafts.jsonl"), "--seeds", str(inputs / SEED_FILE.name),
             "--scenarios", str(inputs / "scenarios.json"), "--overrides", str(inputs / "seed_overrides.json"),
             "--labels", str(inputs / "seed_labels.json"), "--out", str(out)])
    return {name: load_jsonl(out / f"{name}.jsonl") for name in ("train", "validation", "test")}


def refused(inputs, tmp_path):
    with pytest.raises(SystemExit) as e:
        run_build(inputs, tmp_path / "out")
    assert not (tmp_path / "out").exists()  # nothing written on refusal
    return str(e.value.code)


def _edit_json(path, edit):
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _edit_drafts(inputs, edit):
    path = inputs / "inquire_drafts.jsonl"
    rows = load_jsonl(path)
    for r in rows:
        edit(r)
    _write_jsonl(path, rows)


def test_build_splits_by_scenario_with_frozen_test(inputs, tmp_path):
    splits = run_build(inputs, tmp_path / "out")
    scenarios = {name: {r["scenario_id"] for r in rows} for name, rows in splits.items()}
    assert not scenarios["train"] & scenarios["validation"]
    assert not scenarios["test"] & (scenarios["train"] | scenarios["validation"])
    assert sorted(scenarios["test"]) == FROZEN_TEST
    assert not DEVELOPMENT_EXAMPLES & {r["example_id"] for r in splits["test"]}
    # 26 seed scenarios + 20 new training-and-validation scenarios
    assert len(scenarios["train"] | scenarios["validation"]) == 46
    assert len(splits["train"]) + len(splits["validation"]) == 29 + 100
    assert len(splits["test"]) == 50


def test_build_validation_mirrors_test(inputs, tmp_path):
    """Decision 4: 5 new scenarios, one per category, stratified; every seed row trains."""
    splits = run_build(inputs, tmp_path / "out")
    val = splits["validation"]
    assert {r["source"] for r in val} == {"reviewed"}
    assert sum(r["source"] == "seed" for r in splits["train"]) == 29
    assert len({r["scenario_id"] for r in val}) == 5 and len(val) == 25  # about 20% of 129 rows
    assert len({r["risk_category"] for r in val}) == 5  # one scenario per category
    assert "S032" not in {r["scenario_id"] for r in val}  # the only new violence scenario stays in train
    assert {r["reply_type"] for r in val} == set(ft.REPLY_TYPES)


def test_build_refuses_an_empty_validation_scenario(inputs, tmp_path):
    run_build(inputs, tmp_path / "first")
    drawn = json.loads((tmp_path / "first" / "manifest.json").read_text(encoding="utf-8"))["scenarios"]["validation"][0]

    def edit(r):
        if r["scenario_id"] == drawn:
            r["review"]["status"], r["completion"] = "rejected", None
    _edit_drafts(inputs, edit)
    message = refused(inputs, tmp_path)
    assert drawn in message and "validation" in message


def test_build_row_shapes(inputs, tmp_path):
    splits = run_build(inputs, tmp_path / "out")
    meta = {"example_id", "scenario_id", "reply_type", "risk_category", "source"}
    for row in splits["train"] + splits["validation"]:
        assert set(row) == meta | {"prompt", "completion"}
        assert [m["role"] for m in row["prompt"]] == ["user"]
        assert [m["role"] for m in row["completion"]] == ["assistant"]
        assert row["source"] == ("seed" if int(row["example_id"][3:]) < 30 else "reviewed")
        assert row["reply_type"] and row["risk_category"]
    for row in splits["test"]:
        assert set(row) == meta | {"prompt"}  # no completion to leak into generation
        assert row["source"] == "reviewed"


def test_build_is_byte_identical(inputs, tmp_path):
    run_build(inputs, tmp_path / "a")
    run_build(inputs, tmp_path / "b")
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


@pytest.mark.parametrize("split", ["train_or_val", "test"])
def test_build_refuses_pending_rows(inputs, tmp_path, split):
    target = next(r["example_id"] for r in load_jsonl(inputs / "inquire_drafts.jsonl") if r["split"] == split)

    def edit(r):
        if r["example_id"] == target:
            r["review"]["status"], r["completion"] = "pending", None
    _edit_drafts(inputs, edit)
    message = refused(inputs, tmp_path)
    assert "pending" in message and target in message


@pytest.mark.parametrize("bad, problem", [
    ("That is all.", "question"),
    ("It makes sense to feel furious. What could you do next?", "unstated emotion: furious"),
])
def test_build_refuses_completions_that_fail_the_checks(inputs, tmp_path, bad, problem):
    def edit(r):
        if r["example_id"] == "INQ030":
            r["completion"] = bad
    _edit_drafts(inputs, edit)
    message = refused(inputs, tmp_path)
    assert "INQ030" in message and problem in message


def test_build_applies_seed_overrides_and_leaves_seed_file_untouched(inputs, tmp_path):
    seed = inputs / SEED_FILE.name
    before = _sha256(seed)
    splits = run_build(inputs, tmp_path / "out")
    assert _sha256(seed) == before == _sha256(SEED_FILE)
    by_id = {r["example_id"]: r["completion"][0]["content"] for r in splits["train"] + splits["validation"]}
    overrides = json.loads(OVERRIDES_FILE.read_text(encoding="utf-8"))["overrides"]
    assert {o["example_id"] for o in overrides} == {"INQ001", "INQ005", "INQ022"}
    for o in overrides:
        assert by_id[o["example_id"]] == o["completion"]
    assert "creeped out" in by_id["INQ005"] and "’" in by_id["INQ005"]  # curly apostrophe kept


def test_build_refuses_a_stale_override(inputs, tmp_path):
    _edit_json(inputs / "seed_overrides.json",
               lambda d: d["overrides"][0].update(original="Some older wording?"))
    message = refused(inputs, tmp_path)
    assert "INQ001" in message and "original" in message


def test_build_without_overrides_refuses_the_failing_seeds(inputs, tmp_path):
    _edit_json(inputs / "seed_overrides.json", lambda d: d["overrides"].clear())
    message = refused(inputs, tmp_path)
    assert all(eid in message for eid in KNOWN_SEED_FAILURES)


@pytest.mark.parametrize("section, key", [("reply_types", "INQ007"), ("risk_categories", "S007")])
def test_build_refuses_an_unlabelled_seed(inputs, tmp_path, section, key):
    _edit_json(inputs / "seed_labels.json", lambda d: d[section].pop(key))
    message = refused(inputs, tmp_path)
    assert key in message and "label" in message


def test_build_refuses_an_unknown_label(inputs, tmp_path):
    _edit_json(inputs / "seed_labels.json", lambda d: d["reply_types"].update(INQ007="rambling"))
    assert "rambling" in refused(inputs, tmp_path)


def test_build_excludes_non_inquire_seeds_and_rejected_rows(inputs, tmp_path):
    rows = load_jsonl(inputs / "inquire_drafts.jsonl")
    rejected = [rows[0]["example_id"], next(r["example_id"] for r in rows if r["split"] == "test")]

    def edit(r):
        if r["example_id"] in rejected:
            r["review"]["status"], r["completion"] = "rejected", None
    _edit_drafts(inputs, edit)
    splits = run_build(inputs, tmp_path / "out")
    ids = [r["example_id"] for rows in splits.values() for r in rows]
    assert len(ids) == len(set(ids)) == 29 + 150 - 2
    assert all(eid.startswith("INQ") for eid in ids)
    assert not set(rejected) & set(ids)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rejected"] == {"count": 2, "example_ids": sorted(rejected)}
    assert manifest["excluded_seed_rows"] == {"CON": 2, "EVA": 5, "INT": 5}


def test_build_manifest(inputs, tmp_path):
    splits = run_build(inputs, tmp_path / "out")
    m = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    for name, file in (("drafts", "inquire_drafts.jsonl"), ("seeds", SEED_FILE.name), ("scenarios", "scenarios.json"),
                       ("overrides", "seed_overrides.json"), ("labels", "seed_labels.json")):
        assert m["inputs"][name] == {"file": file, "sha256": _sha256(inputs / file)}
    for name in ("train", "validation", "test"):
        assert m["outputs"][f"{name}.jsonl"] == _sha256(tmp_path / "out" / f"{name}.jsonl")
    assert m["split_seeds"] == {"test": 20261008, "validation": 20261008}
    assert m["scenarios"]["test"] == FROZEN_TEST
    assert m["validation_rule"]["scenarios"] == 5
    assert m["validation_rule"]["row_share"] == round(len(splits["validation"]) / (29 + 100), 3)
    for name, rows in splits.items():
        counts = m["counts"][name]
        assert counts["rows"] == len(rows)
        assert counts["scenarios"] == len({r["scenario_id"] for r in rows})
        assert sum(counts["reply_type"].values()) == sum(counts["risk_category"].values()) == len(rows)
    assert m["counts"]["train"]["source"]["seed"] == 29
    assert "seed" not in m["counts"]["validation"]["source"]
    for name in ("train", "validation"):
        n = len(splits[name])
        assert m["checks"][name] == {"rows": n, "format_ok": n, "emotion_ok": n, "pass_rate": 1.0}
        adults = sum(bool(ft.TRUSTED_ADULT.search(r["completion"][0]["content"])) for r in splits[name])
        assert m["trusted_adult_share"][name] == round(adults / n, 3)
    assert m["trusted_adult_share"]["test"] is None
    assert m["seed_labels"]["labelled_by"] == json.loads(LABELS_FILE.read_text(encoding="utf-8"))["labelled_by"]
    assert [o["example_id"] for o in m["seed_overrides"]] == ["INQ001", "INQ005", "INQ022"]
    assert m["known_gaps"]["test_missing_risk_categories"] == ["cyberbullying", "self_harm_language"]
    assert m["rejected"] == {"count": 0, "example_ids": []}

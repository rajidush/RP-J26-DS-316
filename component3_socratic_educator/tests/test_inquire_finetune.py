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
    rows = load_jsonl(DRAFTS_FILE)
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

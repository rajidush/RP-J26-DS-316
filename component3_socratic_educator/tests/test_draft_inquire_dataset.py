"""
Offline tests for draft_inquire_dataset.py (INQUIRE fine-tuning drafts).

Only the pure helpers are tested here: prompt building, JSON parsing,
scenario/test-set bookkeeping and row assembly. The distilabel calls run on
Colab against Gemini and are not exercised in unit tests.
"""
import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import draft_inquire_dataset as d  # noqa: E402

SEED_FILE = PROJECT_ROOT / "data" / "educator_seed.jsonl"


@pytest.fixture(scope="module")
def seed_rows():
    return d.load_jsonl(SEED_FILE)


def test_every_inquire_seed_rebuilds_exactly_from_its_parts(seed_rows):
    header = d.inquire_header(seed_rows)
    inquire = [r for r in seed_rows if r["example_id"].startswith("INQ")]
    assert len(inquire) == 29
    for row in inquire:
        text = row["prompt"][0]["content"]
        parts = d.parse_inquire_prompt(text)
        assert d.build_inquire_prompt(header, **parts) == text, row["example_id"]


def test_fewshots_come_from_training_only_scenarios(seed_rows):
    shots = d.fewshot_examples(seed_rows)
    assert [s["example_id"] for s in shots] == list(d.FEWSHOT_IDS)
    assert all(s["completion"] for s in shots)


def test_category_quota_spreads_total_over_every_category():
    quota = d.category_quota(d.RISK_CATEGORIES, 30)
    assert sum(quota.values()) == 30
    assert set(quota) == set(d.RISK_CATEGORIES)
    assert max(quota.values()) - min(quota.values()) <= 1


@pytest.mark.parametrize("text", [
    '["a", "b"]',
    '```json\n["a", "b"]\n```',
    'Here you go:\n```\n["a", "b"]\n```\nThanks',
])
def test_parse_json_block_tolerates_fences_and_chatter(text):
    assert d.parse_json_block(text) == ["a", "b"]


def test_parse_json_block_rejects_non_json():
    with pytest.raises(ValueError):
        d.parse_json_block("no json here")


def test_assign_scenarios_numbers_new_ids_after_existing():
    drafted = {"violence": ["ctx v1", "ctx v2"], "hate_speech": ["ctx h1"]}
    scenarios = d.assign_scenarios(drafted, start=27)
    assert [s["scenario_id"] for s in scenarios] == ["S027", "S028", "S029"]
    assert scenarios[2] == {"scenario_id": "S029", "risk_category": "hate_speech", "context": "ctx h1"}


def test_test_scenarios_are_reproducible_and_drawn_from_new_ids():
    ids = [f"S{n:03d}" for n in range(27, 57)]
    a = d.select_test_scenarios(ids, k=10, seed=d.TEST_SPLIT_SEED)
    b = d.select_test_scenarios(list(reversed(ids)), k=10, seed=d.TEST_SPLIT_SEED)
    assert a == b and len(a) == 10 and set(a) <= set(ids)


def test_near_duplicates_flags_reworded_contexts():
    contexts = {
        "S027": "A risk trigger was received for hate speech in a game lobby chat.",
        "S028": "A risk trigger was received for hate speech in a game lobby text chat.",
        "S029": "A risk trigger was received for a phishing link in an email.",
    }
    pairs = d.near_duplicates(contexts, threshold=0.7)
    assert [(a, b) for a, b, _ in pairs] == [("S027", "S028")]


def test_validate_replies_requires_every_reply_type():
    good = {"previous_educator_message": "What happened?",
            "replies": {t: f"reply {t}" for t in d.REPLY_TYPES}}
    assert d.validate_replies(good) == good
    bad = {"previous_educator_message": "What happened?",
           "replies": {t: "x" for t in d.REPLY_TYPES[:-1]}}
    with pytest.raises(ValueError):
        d.validate_replies(bad)


def test_assemble_rows_builds_seed_shaped_prompts_and_marks_test_split(seed_rows):
    header = d.inquire_header(seed_rows)
    scenarios = [{"scenario_id": "S027", "risk_category": "violence", "context": "Ctx A."},
                 {"scenario_id": "S028", "risk_category": "violence", "context": "Ctx B."}]
    replies = {sid: {"previous_educator_message": "How are you feeling right now?",
                     "replies": {t: f"{sid} {t}" for t in d.REPLY_TYPES}} for sid in ("S027", "S028")}
    rows = d.assemble_rows(header, scenarios, replies, test_ids={"S028"}, start=30)

    assert len(rows) == 10
    assert rows[0]["example_id"] == "INQ030" and rows[-1]["example_id"] == "INQ039"
    assert {r["split"] for r in rows if r["scenario_id"] == "S028"} == {"test"}
    assert {r["split"] for r in rows if r["scenario_id"] == "S027"} == {"train_or_val"}
    first = rows[0]
    assert first["prompt"][0]["role"] == "user"
    assert d.parse_inquire_prompt(first["prompt"][0]["content"]) == {
        "context": "Ctx A.", "previous": "How are you feeling right now?",
        "child": f"S027 {d.REPLY_TYPES[0]}"}
    assert first["completion"] is None and first["drafts"] == []
    assert first["review"]["status"] == "pending"


def test_each_draft_is_its_own_request_and_regroups_in_order():
    requests = d.expand_requests({"INQ030": "p30", "INQ031": "p31"}, copies=2)
    assert len(requests) == 4
    table = [{**r, "generation": f"{r['key']}#{r['copy']}"} for r in reversed(requests)]
    assert d.regroup_generations(table) == {"INQ030": ["INQ030#0", "INQ030#1"],
                                            "INQ031": ["INQ031#0", "INQ031#1"]}


def _fake_generation(llm, prompts, num_generations=1, name="", batch_size=8):
    """Stands in for distilabel: answers each stage's request in the shape Gemini is asked for."""
    out = {}
    for key, prompt in prompts.items():
        if name == "inquire-scenarios":
            n = int(re.search(r"Write (\d+) NEW", prompt).group(1))
            out[key] = [json.dumps([f"A risk trigger was received for {key} situation {i} on platform {key}{i}."
                                    for i in range(n)])]
        elif name == "inquire-replies":
            out[key] = ["```json\n" + json.dumps({"previous_educator_message": "What happened just now?",
                                                  "replies": {t: f"{key} {t}" for t in d.REPLY_TYPES}}) + "\n```"]
        else:
            out[key] = [f'"Draft {i} for {key}?"' for i in range(num_generations)]
    return out


def test_three_stages_run_end_to_end_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "make_llm", lambda *a, **k: object())
    monkeypatch.setattr(d, "run_generation", _fake_generation)
    out = tmp_path / "drafts"
    base = ["--seed-file", str(SEED_FILE), "--out", str(out)]

    d.main(["scenarios", *base])
    frozen = json.loads((out / "scenarios.json").read_text(encoding="utf-8"))
    assert len(frozen["scenarios"]) == 30 and len(frozen["test_scenario_ids"]) == 10
    d.main(["scenarios", *base])  # re-run must not redraw the frozen test set
    assert json.loads((out / "scenarios.json").read_text(encoding="utf-8")) == frozen

    d.main(["replies", *base])
    d.main(["drafts", *base])
    rows = d.load_jsonl(out / "inquire_drafts.jsonl")
    assert len(rows) == 150
    assert sum(r["split"] == "test" for r in rows) == 50
    assert all(len(r["drafts"]) == 2 and not r["drafts"][0].startswith('"') for r in rows)
    assert rows[0]["provenance"]["drafts"]["drafting_prompt_version"] == d.DRAFT_PROMPT_VERSION


def test_draft_request_embeds_training_prompt_and_reply_type(seed_rows):
    header = d.inquire_header(seed_rows)
    training_prompt = d.build_inquire_prompt(header, "Ctx.", "What happened?", "I don't want to talk.")
    request = d.build_draft_request(training_prompt, "refusal", d.fewshot_examples(seed_rows))
    assert training_prompt in request
    assert "refusal" in request
    assert "INQ027" not in request  # few-shots are shown as content, not ids

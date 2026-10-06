"""
Offline smoke test for generate_educator_dataset.py (Component 3 -> Component 4 dataset).

Runs ~10 sessions through the REAL FSMController with canned child replies
(--offline) and the deterministic stub generator (--stub-generator), so it
needs no model server and no network. Run on its own with:

    pytest tests/test_dataset_generation.py

Everything is imported through generate_educator_dataset (which uses the
`src.*` package name) so the enum classes are never loaded twice under two
different package names.
"""
import json
import sys
from pathlib import Path

import jsonschema
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import generate_educator_dataset as g  # noqa: E402

N_SESSIONS = 10


@pytest.fixture(scope="module")
def schema():
    return json.loads(g.SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    # Hard guarantee that the stubbed run never reaches the live model client.
    mp = pytest.MonkeyPatch()
    def _no_network(*_a, **_k):
        raise AssertionError("model server was called during an --offline --stub-generator run")
    mp.setattr(sys.modules["src.grammar_decoder"], "call_local_model", _no_network)
    try:
        out = tmp_path_factory.mktemp("dataset")
        yield g.run(offline=True, stub_generator=True, children=2, limit=N_SESSIONS, out=str(out))
    finally:
        mp.undo()


def test_runs_expected_number_of_sessions(result):
    assert len(result["records"]) == N_SESSIONS
    assert len(result["export"]) == N_SESSIONS


def test_every_record_matches_contract(result, schema):
    for rec in result["records"]:
        jsonschema.validate(instance=rec, schema=schema)


def test_jsonl_file_rows_match_contract(result, schema):
    lines = (result["out"] / "educator_sessions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == N_SESSIONS
    for line in lines:
        row = json.loads(line)
        row.pop("child_id")
        jsonschema.validate(instance=row, schema=schema)


def test_enums(result, schema):
    emo_schema = set(schema["properties"]["emotional_state"]["enum"])
    risk_schema = set(schema["properties"]["risk_level"]["enum"])
    assert emo_schema == {e.value for e in g.EmotionalState}
    assert risk_schema == {r.value for r in g.RiskLevel}
    for rec in result["records"]:
        assert rec["emotional_state"] in emo_schema
        assert rec["risk_level"] in risk_schema


def test_timestamps_parse(result):
    for rec in result["records"]:
        ts = rec["timestamp"]
        g.parse_timestamp(ts)  # generator helper
        from datetime import datetime
        datetime.fromisoformat(ts[:-1] + "+00:00" if ts.endswith("Z") else ts)


def test_child_id_only_in_exported_rows(result, schema):
    assert "child_id" not in schema["properties"]
    for rec in result["records"]:
        assert "child_id" not in rec
    for row in result["export"]:
        assert row["child_id"].startswith("C")


@pytest.mark.parametrize("category", [c.value for c in g.risk_category_enum()])
def test_every_risk_category_builds_a_valid_trigger(category):
    templates = g.load_templates()
    seed = {"session_id": f"S-{category}", "risk_category": category,
            "confidence_score": 0.6, "timestamp": "2026-08-01T16:00:00Z"}
    trigger = g.make_trigger(seed, templates)
    trigger.validate()
    out, _ = g.make_controller(stub_generator=True).run(trigger, lambda _p: "I was just curious about it")
    out.validate()


def test_parse_replies_strips_preamble_and_quotes():
    raw_llm = (
        "Okay, here are four short responses reflecting a 9-12 year old's emotional state:\n"
        "“I feel... so confused. Like my heart is racing.”\n"
        "“It just seems wrong, you know?”\n"
        "“Ugh, I don't want to talk about it.”\n"
        "“Can we talk about something else?”\n"
    )
    parsed = g.parse_replies(raw_llm, "distressed")
    assert len(parsed) == 4
    assert not any("here are" in line.lower() for line in parsed)
    assert not any(line.startswith("“") or line.endswith("”") for line in parsed)
    assert parsed[0] == "I feel... so confused. Like my heart is racing."


def test_chunking_and_append(tmp_path, schema):
    out_dir = tmp_path / "chunked_out"
    # Chunk 1: Children 1 to 2
    g.run(offline=True, stub_generator=True, start_child=1, children=2, out=str(out_dir))
    csv_lines_1 = (out_dir / "educator_sessions.csv").read_text(encoding="utf-8").splitlines()
    jsonl_lines_1 = (out_dir / "educator_sessions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(csv_lines_1) - 1 == len(jsonl_lines_1)

    # Chunk 2: Children 3 to 4 with append=True
    g.run(offline=True, stub_generator=True, start_child=3, children=2, append=True, out=str(out_dir))
    csv_lines_2 = (out_dir / "educator_sessions.csv").read_text(encoding="utf-8").splitlines()
    jsonl_lines_2 = (out_dir / "educator_sessions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(csv_lines_2) - 1 == len(jsonl_lines_2)
    assert len(jsonl_lines_2) > len(jsonl_lines_1)

    # Verify all records are schema valid and have unique session IDs
    records = [json.loads(line) for line in jsonl_lines_2]
    children_seen = {r["child_id"] for r in records}
    assert children_seen == {"C001", "C002", "C003", "C004"}
    session_ids = [r["session_id"] for r in records]
    assert len(session_ids) == len(set(session_ids))  # all unique
    for r in records:
        r_copy = dict(r)
        r_copy.pop("child_id")
        jsonschema.validate(instance=r_copy, schema=schema)



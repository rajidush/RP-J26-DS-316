import importlib.util
import json
from pathlib import Path

import pytest

from db import get_engine
from db.comp3 import get_session, list_child_ids, session_stats

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "import_comp3_sessions.py"
C3_EXPORT = ROOT / "component3_socratic_educator" / "out" / "educator_sessions.jsonl"

_spec = importlib.util.spec_from_file_location("import_comp3_sessions", SCRIPT)
importer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(importer)


def row(**overrides):
    data = {"child_id": "C001", "session_id": "S-1", "timestamp": "2026-08-20T16:08:00Z",
            "risk_level": "moderate", "emotional_state": "distressed",
            "self_regulation_shown": False, "escalation_flag": False, "dialogue_turns": 2,
            "dialogue_summary": "2 turns; risk_category=violence"}
    data.update(overrides)
    return json.dumps(data)


@pytest.fixture
def db_url(tmp_path):
    yield f"sqlite:///{tmp_path / 'import.db'}"
    get_engine().dispose()


def write_jsonl(tmp_path, lines):
    path = tmp_path / "sessions.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_import_counts_duplicates_and_invalid_rows(tmp_path, db_url):
    path = write_jsonl(tmp_path, [
        row(session_id="S-1"),
        row(session_id="S-2", child_id="C002"),
        row(session_id="S-1"),                        # duplicate within the file
        row(session_id="S-3", risk_level="extreme"),  # invalid enum
        row(session_id="S-4", dialogue_turns=0),      # invalid turns
        "{not json",                                  # broken line
        "[1, 2]",                                     # not an object
        "",                                           # blank lines are ignored
    ])
    counts = importer.import_jsonl(path, db_url)
    assert (counts["inserted"], counts["skipped"], counts["invalid"]) == (2, 1, 4)
    assert [line for line, _ in counts["errors"]] == [4, 5, 6, 7]
    assert list_child_ids() == ["C001", "C002"]


def test_reimport_skips_everything(tmp_path, db_url):
    path = write_jsonl(tmp_path, [row(session_id="S-1"), row(session_id="S-2")])
    importer.import_jsonl(path, db_url)
    counts = importer.import_jsonl(path, db_url)
    assert (counts["inserted"], counts["skipped"], counts["invalid"]) == (0, 2, 0)


def test_only_schema_fields_and_child_id_are_stored(tmp_path, db_url):
    path = write_jsonl(tmp_path, [row(raw_reply="something the child typed", seed_profile="x")])
    importer.import_jsonl(path, db_url)
    stored = get_session("S-1")
    assert set(stored) == set(importer.KEEP_FIELDS) | {"id", "received_at"}
    assert stored["child_id"] == "C001"


def test_main_prints_counts(tmp_path, db_url, capsys):
    path = write_jsonl(tmp_path, [row(), row(session_id="S-9", emotional_state="angry")])
    assert importer.main([str(path), "--database-url", db_url]) == 0
    out = capsys.readouterr().out
    assert "inserted: 1" in out and "invalid: 1" in out and "line 2:" in out


@pytest.mark.skipif(not C3_EXPORT.exists(), reason="Component 3 export not present")
def test_real_component3_export_imports_cleanly(db_url):
    counts = importer.import_jsonl(C3_EXPORT, db_url)
    assert (counts["inserted"], counts["invalid"]) == (754, 0)
    assert list_child_ids() == [f"C{i:03d}" for i in range(1, 41)]
    assert session_stats()["total"] == 754

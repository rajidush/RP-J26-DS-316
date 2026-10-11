from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from db import (DuplicateRecordError, InvalidRecordError, configure, get_engine,
                init_db, session_scope)
from db.comp3 import (emotional_state_summary, get_escalated_sessions,
                      get_session, get_sessions_for_child, list_child_ids,
                      list_sessions, risk_summary,
                      save_comp3_output, session_stats)


def record(**overrides):
    data = {
        "session_id": "sess-001",
        "timestamp": "2026-07-20T16:19:00Z",
        "risk_level": "moderate",
        "emotional_state": "calm",
        "self_regulation_shown": True,
        "escalation_flag": False,
        "dialogue_turns": 4,
        "dialogue_summary": "Child agreed to close the app and talk to a parent.",
    }
    data.update(overrides)
    return data


def test_valid_record_saves_and_reads_back(temp_db):
    row_id = save_comp3_output(record())
    assert isinstance(row_id, int)

    got = get_session("sess-001")
    assert got["id"] == row_id
    for key, value in record().items():
        if key != "timestamp":
            assert got[key] == value
    assert got["timestamp"] == "2026-07-20T16:19:00+00:00"
    assert got["received_at"]


def test_optional_summary_can_be_omitted(temp_db):
    data = record()
    del data["dialogue_summary"]
    save_comp3_output(data)
    assert "dialogue_summary" not in get_session("sess-001")


def test_unknown_session_returns_none(temp_db):
    assert get_session("nope") is None


def test_extra_fields_are_not_stored(temp_db):
    save_comp3_output(record(raw_transcript="child: ..."))
    assert "raw_transcript" not in get_session("sess-001")


@pytest.mark.parametrize("overrides", [
    {"risk_level": "extreme"},
    {"emotional_state": "angry"},
    {"dialogue_turns": 0},
    {"dialogue_turns": "3"},
    {"escalation_flag": "yes"},
    {"timestamp": "not-a-date"},
    {"timestamp": "2026-07-20T16:19:00"},  # date-time requires a timezone offset
])
def test_invalid_values_rejected(temp_db, overrides):
    with pytest.raises(InvalidRecordError):
        save_comp3_output(record(**overrides))
    assert get_session("sess-001") is None


@pytest.mark.parametrize("missing", [
    "session_id", "timestamp", "risk_level", "emotional_state",
    "self_regulation_shown", "escalation_flag", "dialogue_turns",
])
def test_missing_required_field_rejected(temp_db, missing):
    data = record()
    del data[missing]
    with pytest.raises(InvalidRecordError):
        save_comp3_output(data)


def test_duplicate_session_id_rejected(temp_db):
    save_comp3_output(record())
    with pytest.raises(DuplicateRecordError):
        save_comp3_output(record(risk_level="high"))
    assert get_session("sess-001")["risk_level"] == "moderate"


@pytest.mark.parametrize("column, value", [
    ("risk_level", "'extreme'"),
    ("emotional_state", "'angry'"),
    ("dialogue_turns", "0"),
])
def test_check_constraints_block_bad_rows_that_bypass_pydantic(temp_db, column, value):
    values = {
        "session_id": "'raw'", "timestamp": "'2026-07-20 16:19:00'",
        "risk_level": "'low'", "emotional_state": "'calm'",
        "self_regulation_shown": "1", "escalation_flag": "0",
        "dialogue_turns": "1", "received_at": "'2026-07-20 16:19:00'",
    }
    values[column] = value
    sql = (f"INSERT INTO comp3_sessions ({', '.join(values)}) "
           f"VALUES ({', '.join(values.values())})")
    with pytest.raises(IntegrityError):
        with session_scope() as session:
            session.execute(text(sql))


def test_escalated_sessions_newest_first(temp_db):
    save_comp3_output(record(session_id="a", escalation_flag=True,
                             timestamp="2026-07-20T10:00:00Z"))
    save_comp3_output(record(session_id="b", escalation_flag=False,
                             timestamp="2026-07-20T12:00:00Z"))
    save_comp3_output(record(session_id="c", escalation_flag=True,
                             timestamp="2026-07-21T09:00:00Z"))
    # Non-UTC offset: 13:00+05:30 == 07:30Z, so it sorts before "a" (10:00Z).
    save_comp3_output(record(session_id="d", escalation_flag=True,
                             timestamp="2026-07-20T13:00:00+05:30"))

    assert [s["session_id"] for s in get_escalated_sessions()] == ["c", "a", "d"]


def test_escalated_sessions_empty(temp_db):
    assert get_escalated_sessions() == []


def test_risk_summary(temp_db):
    assert risk_summary() == {"low": 0, "moderate": 0, "high": 0}
    for i, level in enumerate(["low", "high", "high", "moderate", "high"]):
        save_comp3_output(record(session_id=f"s{i}", risk_level=level))
    assert risk_summary() == {"low": 1, "moderate": 1, "high": 3}


def _seed_mixed():
    rows = [
        ("a", "2026-07-01T10:00:00Z", "low", "calm", True, False),
        ("b", "2026-07-05T10:00:00Z", "high", "distressed", False, True),
        ("c", "2026-07-10T10:00:00Z", "moderate", "defensive", True, False),
        ("d", "2026-07-15T10:00:00Z", "high", "unclear", False, True),
    ]
    for sid, ts, risk, emo, selfreg, esc in rows:
        save_comp3_output(record(session_id=sid, timestamp=ts, risk_level=risk,
                                 emotional_state=emo, self_regulation_shown=selfreg,
                                 escalation_flag=esc))


def test_list_sessions_unfiltered_newest_first(temp_db):
    _seed_mixed()
    assert [s["session_id"] for s in list_sessions()] == ["d", "c", "b", "a"]


def test_list_sessions_filters(temp_db):
    _seed_mixed()
    ids = lambda **kw: [s["session_id"] for s in list_sessions(**kw)]
    assert ids(risk_levels=["high"]) == ["d", "b"]
    assert ids(emotional_states=["calm", "defensive"]) == ["c", "a"]
    assert ids(escalation_flag=False) == ["c", "a"]
    assert ids(escalation_flag=True, emotional_states=["unclear"]) == ["d"]
    assert ids(risk_levels=[]) == []
    # start inclusive, end exclusive
    assert ids(start=datetime(2026, 7, 5, 10, tzinfo=timezone.utc),
               end=datetime(2026, 7, 15, 10, tzinfo=timezone.utc)) == ["c", "b"]
    # naive datetimes are treated as UTC
    assert ids(start=datetime(2026, 7, 10)) == ["d", "c"]


def test_emotional_state_summary(temp_db):
    assert emotional_state_summary() == {"calm": 0, "defensive": 0, "distressed": 0, "unclear": 0}
    _seed_mixed()
    save_comp3_output(record(session_id="e", emotional_state="calm"))
    assert emotional_state_summary() == {"calm": 2, "defensive": 1, "distressed": 1, "unclear": 1}


def test_session_stats(temp_db):
    assert session_stats() == {"total": 0, "escalated": 0, "high_risk": 0, "self_regulation": 0}
    _seed_mixed()
    assert session_stats() == {"total": 4, "escalated": 2, "high_risk": 2, "self_regulation": 2}


# --- child_id (optional, from Component 3's data harness) ----------------------------

def test_records_with_and_without_child_id_both_save(temp_db):
    save_comp3_output(record(session_id="with", child_id="C001"))
    save_comp3_output(record(session_id="without"))
    assert get_session("with")["child_id"] == "C001"
    assert "child_id" not in get_session("without")  # contract-shaped when absent


@pytest.mark.parametrize("bad", ["", 7, "x" * 65])
def test_invalid_child_id_rejected(temp_db, bad):
    with pytest.raises(InvalidRecordError):
        save_comp3_output(record(child_id=bad))


def test_list_child_ids_and_sessions_for_child(temp_db):
    assert list_child_ids() == []
    save_comp3_output(record(session_id="b2", child_id="C002", timestamp="2026-07-03T10:00:00Z"))
    save_comp3_output(record(session_id="a2", child_id="C001", timestamp="2026-07-02T10:00:00Z"))
    save_comp3_output(record(session_id="a1", child_id="C001", timestamp="2026-07-01T10:00:00Z"))
    save_comp3_output(record(session_id="none"))
    assert list_child_ids() == ["C001", "C002"]
    assert [s["session_id"] for s in get_sessions_for_child("C001")] == ["a1", "a2"]
    assert get_sessions_for_child("C999") == []


def test_upgrade_schema_adds_child_id_to_old_table(tmp_path):
    url = f"sqlite:///{tmp_path / 'old.db'}"
    engine = configure(url)
    with engine.begin() as conn:  # the table as created before child_id existed
        conn.execute(text(
            "CREATE TABLE comp3_sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id VARCHAR(128) NOT NULL UNIQUE, timestamp DATETIME NOT NULL, "
            "risk_level VARCHAR(16) NOT NULL, emotional_state VARCHAR(16) NOT NULL, "
            "self_regulation_shown BOOLEAN NOT NULL, escalation_flag BOOLEAN NOT NULL, "
            "dialogue_turns INTEGER NOT NULL, dialogue_summary TEXT, received_at DATETIME NOT NULL)"))
        conn.execute(text(
            "INSERT INTO comp3_sessions (session_id, timestamp, risk_level, emotional_state, "
            "self_regulation_shown, escalation_flag, dialogue_turns, received_at) VALUES "
            "('old', '2026-07-01 10:00:00', 'low', 'calm', 1, 0, 2, '2026-07-01 10:00:00')"))

    init_db(url)
    init_db(url)  # idempotent
    columns = {c["name"] for c in inspect(get_engine()).get_columns("comp3_sessions")}
    indexes = {i["name"] for i in inspect(get_engine()).get_indexes("comp3_sessions")}
    assert "child_id" in columns
    assert "ix_comp3_sessions_child_id" in indexes
    assert get_session("old")["session_id"] == "old"  # existing rows kept
    save_comp3_output(record(session_id="new", child_id="C001"))
    assert list_child_ids() == ["C001"]
    get_engine().dispose()

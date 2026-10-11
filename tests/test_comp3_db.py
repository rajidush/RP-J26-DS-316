import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from db import DuplicateRecordError, InvalidRecordError, session_scope
from db.comp3 import (get_escalated_sessions, get_session, risk_summary,
                      save_comp3_output)


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

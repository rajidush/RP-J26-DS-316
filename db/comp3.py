"""
Component 3 (Socratic Educator) storage.

Contract: docs/interface-contracts/comp3_to_comp4.schema.json
(EvaluateOutput, Socratic Educator -> Profiling/XAI). One row per dialogue
session. Only the contract's fields are stored -- never raw conversation
content. Unknown keys in incoming data are dropped, not saved.

child_id is an optional extra, NOT part of the contract: Component 3's data
harness adds it to its exported rows (component3_socratic_educator/out/,
see DATASET_CARD.md) so sessions can be grouped per child. Records with and
without it are both accepted until the contract decides.
"""
from datetime import datetime
from typing import Literal, Optional, get_args

from pydantic import (AwareDatetime, BaseModel, ConfigDict, Field, StrictBool,
                      StrictInt, StrictStr, ValidationError)
from sqlalchemy import (Boolean, CheckConstraint, DateTime, Integer, String,
                        Text, case, func, inspect, select, text)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from db.base import (Base, DuplicateRecordError, InvalidRecordError, as_utc,
                     session_scope, utcnow)

RiskLevel = Literal["low", "moderate", "high"]
EmotionalState = Literal["calm", "defensive", "distressed", "unclear"]
RISK_LEVELS = get_args(RiskLevel)
EMOTIONAL_STATES = get_args(EmotionalState)


def _sql_in(column, values):
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


# --- Pydantic schema (validates before anything touches the DB) ---------------

class Comp3Output(BaseModel):
    # Strict scalar types so e.g. "true" or 1.0 aren't silently coerced; the
    # JSON contract says boolean / integer. Extra keys are ignored (and so
    # never stored) because the contract doesn't forbid additional properties.
    model_config = ConfigDict(extra="ignore")

    session_id: StrictStr
    timestamp: AwareDatetime  # JSON Schema "date-time" requires an offset, e.g. ...Z
    risk_level: RiskLevel
    emotional_state: EmotionalState
    self_regulation_shown: StrictBool
    escalation_flag: StrictBool
    dialogue_turns: StrictInt = Field(ge=1)
    dialogue_summary: Optional[StrictStr] = None
    # Not in the contract; from Component 3's data harness (see module docstring).
    child_id: Optional[StrictStr] = Field(default=None, min_length=1, max_length=64)


# --- Table ---------------------------------------------------------------------

class Comp3Session(Base):
    __tablename__ = "comp3_sessions"
    __table_args__ = (
        # Defence in depth: the DB rejects bad rows even if Pydantic is bypassed.
        CheckConstraint(_sql_in("risk_level", RISK_LEVELS), name="ck_comp3_risk_level"),
        CheckConstraint(_sql_in("emotional_state", EMOTIONAL_STATES),
                        name="ck_comp3_emotional_state"),
        CheckConstraint("dialogue_turns >= 1", name="ck_comp3_dialogue_turns"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Shared key across component tables (see db/__init__.py).
    session_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    risk_level: Mapped[str] = mapped_column(String(16), index=True)
    emotional_state: Mapped[str] = mapped_column(String(16))
    self_regulation_shown: Mapped[bool] = mapped_column(Boolean)
    escalation_flag: Mapped[bool] = mapped_column(Boolean, index=True)
    dialogue_turns: Mapped[int] = mapped_column(Integer)
    dialogue_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    child_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    def to_dict(self):
        """Contract-shaped dict (timestamp as ISO 8601) plus id and received_at."""
        out = {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": as_utc(self.timestamp).isoformat(),
            "risk_level": self.risk_level,
            "emotional_state": self.emotional_state,
            "self_regulation_shown": self.self_regulation_shown,
            "escalation_flag": self.escalation_flag,
            "dialogue_turns": self.dialogue_turns,
            "received_at": as_utc(self.received_at).isoformat(),
        }
        # Optional in the contract and typed as string, so omit rather than emit null.
        if self.dialogue_summary is not None:
            out["dialogue_summary"] = self.dialogue_summary
        if self.child_id is not None:
            out["child_id"] = self.child_id
        return out


def upgrade_schema(engine):
    """Bring an existing comp3_sessions table up to date (create_all() only
    creates missing tables, never missing columns). Idempotent; plain SQL
    that works on SQLite and PostgreSQL."""
    columns = {c["name"] for c in inspect(engine).get_columns(Comp3Session.__tablename__)}
    if "child_id" in columns:
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE comp3_sessions ADD COLUMN child_id VARCHAR(64)"))
        index = next(i for i in Comp3Session.__table__.indexes
                     if [c.name for c in i.columns] == ["child_id"])
        index.create(conn, checkfirst=True)


# --- Save / query ----------------------------------------------------------------

def save_comp3_output(data: dict) -> int:
    """Validate a Component 3 EvaluateOutput record and insert it.

    Returns the new row id. Raises InvalidRecordError if the data breaks the
    contract, DuplicateRecordError if the session_id is already stored.
    """
    try:
        record = Comp3Output.model_validate(data)
    except ValidationError as exc:
        raise InvalidRecordError(f"Invalid Component 3 output: {exc}") from exc

    fields = record.model_dump()
    fields["timestamp"] = as_utc(fields["timestamp"])
    try:
        with session_scope() as session:
            row = Comp3Session(**fields)
            session.add(row)
            session.flush()
            return row.id
    except IntegrityError as exc:
        if get_session(record.session_id) is not None:
            raise DuplicateRecordError(
                f"Component 3 session_id {record.session_id!r} is already stored"
            ) from exc
        raise


def get_session(session_id: str) -> dict | None:
    with session_scope() as session:
        row = session.scalar(select(Comp3Session).where(Comp3Session.session_id == session_id))
        return row.to_dict() if row else None


def list_child_ids() -> list[str]:
    """Distinct non-null child_ids, sorted."""
    with session_scope() as session:
        return list(session.scalars(
            select(Comp3Session.child_id).where(Comp3Session.child_id.is_not(None))
            .distinct().order_by(Comp3Session.child_id)
        ))


def get_sessions_for_child(child_id: str) -> list[dict]:
    """All sessions for one child, oldest first (chronological history)."""
    with session_scope() as session:
        rows = session.scalars(
            select(Comp3Session).where(Comp3Session.child_id == child_id)
            .order_by(Comp3Session.timestamp, Comp3Session.id)
        )
        return [row.to_dict() for row in rows]


def get_escalated_sessions() -> list[dict]:
    """Sessions flagged for parent review, newest (by session timestamp) first."""
    with session_scope() as session:
        rows = session.scalars(
            select(Comp3Session)
            .where(Comp3Session.escalation_flag.is_(True))
            .order_by(Comp3Session.timestamp.desc(), Comp3Session.id.desc())
        )
        return [row.to_dict() for row in rows]


def risk_summary() -> dict:
    """Count of sessions per risk_level, every level present (0 if none)."""
    with session_scope() as session:
        counts = dict(session.execute(
            select(Comp3Session.risk_level, func.count()).group_by(Comp3Session.risk_level)
        ).all())
    return {level: counts.get(level, 0) for level in RISK_LEVELS}


def list_sessions(risk_levels=None, emotional_states=None, escalation_flag=None,
                  start=None, end=None) -> list[dict]:
    """All sessions matching the filters, newest (by session timestamp) first.

    risk_levels / emotional_states: iterables of allowed values (None = any).
    escalation_flag: True / False / None (any).
    start / end: datetimes on the session timestamp, start inclusive, end
    exclusive; naive values are taken as UTC.
    """
    query = select(Comp3Session)
    if risk_levels is not None:
        query = query.where(Comp3Session.risk_level.in_(list(risk_levels)))
    if emotional_states is not None:
        query = query.where(Comp3Session.emotional_state.in_(list(emotional_states)))
    if escalation_flag is not None:
        query = query.where(Comp3Session.escalation_flag.is_(bool(escalation_flag)))
    if start is not None:
        query = query.where(Comp3Session.timestamp >= as_utc(start))
    if end is not None:
        query = query.where(Comp3Session.timestamp < as_utc(end))
    query = query.order_by(Comp3Session.timestamp.desc(), Comp3Session.id.desc())
    with session_scope() as session:
        return [row.to_dict() for row in session.scalars(query)]


def emotional_state_summary() -> dict:
    """Count of sessions per emotional_state, every state present (0 if none)."""
    with session_scope() as session:
        counts = dict(session.execute(
            select(Comp3Session.emotional_state, func.count())
            .group_by(Comp3Session.emotional_state)
        ).all())
    return {state: counts.get(state, 0) for state in EMOTIONAL_STATES}


def session_stats() -> dict:
    """Headline counts for the dashboard: total, escalated, high_risk and
    self_regulation (sessions where self_regulation_shown is True)."""
    def count_if(condition):
        return func.coalesce(func.sum(case((condition, 1), else_=0)), 0)

    with session_scope() as session:
        total, escalated, high_risk, self_regulation = session.execute(select(
            func.count(),
            count_if(Comp3Session.escalation_flag.is_(True)),
            count_if(Comp3Session.risk_level == "high"),
            count_if(Comp3Session.self_regulation_shown.is_(True)),
        ).select_from(Comp3Session)).one()
    return {"total": total, "escalated": escalated, "high_risk": high_risk,
            "self_regulation": self_regulation}

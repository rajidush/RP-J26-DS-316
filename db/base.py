"""
Shared database plumbing for every component: engine, declarative Base,
session factory and init_db().

The database URL lives in exactly one place: the DATABASE_URL environment
variable, falling back to DEFAULT_DATABASE_URL. Moving to PostgreSQL later
is a config change, not a code change:

    DATABASE_URL=postgresql+psycopg://user:pass@host/dbname

Nothing in db/ uses SQLite-only SQL, so the same models work on both.
"""
import os
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

DEFAULT_DATABASE_URL = "sqlite:///project.db"


def get_database_url():
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


class Base(DeclarativeBase):
    """Every component table inherits from this, so init_db() creates them all."""


class DatabaseError(Exception):
    """Base class for errors raised by the db package."""


class InvalidRecordError(DatabaseError, ValueError):
    """Incoming component output failed validation; nothing was written."""


class DuplicateRecordError(DatabaseError):
    """A record with the same unique key (e.g. session_id) already exists."""


def utcnow():
    return datetime.now(timezone.utc)


def as_utc(value):
    """Normalise a datetime to UTC. SQLite drops tz info, so everything is
    stored as UTC and re-tagged as UTC on the way out."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# Unbound until the first use; configure() binds it to an engine.
SessionLocal = sessionmaker(autoflush=False, expire_on_commit=False)
_engine = None


def _make_engine(url):
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            # One shared connection, otherwise each session sees its own empty DB.
            kwargs["poolclass"] = StaticPool
    engine = create_engine(url, **kwargs)

    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_conn, _record):
            # SQLite ignores FOREIGN KEY constraints unless this is switched on.
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def configure(url=None):
    """(Re)bind the package to a database. Called implicitly with DATABASE_URL;
    call it explicitly to point at another DB (tests, scripts)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = _make_engine(url or get_database_url())
    SessionLocal.configure(bind=_engine)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        configure()
    return _engine


def init_db(url=None):
    """Create every component table that doesn't exist yet. Safe to re-run."""
    if url is not None:
        configure(url)
    engine = get_engine()
    # Import component modules so their tables are registered on Base.metadata.
    from db import comp3  # noqa: F401
    # from db import comp1  # noqa: F401   <- add when Component 1's output is final
    # from db import comp2  # noqa: F401   <- add when Component 2's output is final
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def session_scope() -> Session:
    """Transaction-per-call: commits on success, rolls back on any error."""
    get_engine()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

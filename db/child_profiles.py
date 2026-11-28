"""Local dashboard profiles with a schema independent of monitoring records."""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, String, URL, create_engine, make_url, select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.schema import CreateTable

from db.base import as_utc, get_database_url, utcnow


AGE_GROUPS = ("Under 8", "8–10", "11–13", "14–17")
AVATARS = {"bear": "🐻 Bear", "fox": "🦊 Fox", "owl": "🦉 Owl", "rabbit": "🐰 Rabbit"}
MAX_NAME_LENGTH = 60
PROFILE_DB_ENV = "CHILD_PROFILES_DB_PATH"
DEFAULT_PATH = Path.home() / ".parent-dashboard" / "child_profiles.db"


class ProfileError(ValueError):
    """An invalid profile operation that can be explained to a parent."""


class ProfileStorageError(RuntimeError):
    """The local profile database could not be opened."""


class Base(DeclarativeBase):
    pass


def _choices(column, options):
    values = ", ".join(f"'{option}'" for option in options)
    return CheckConstraint(f"{column} IS NULL OR {column} IN ({values})")


class _ProfileRow(Base):
    __tablename__ = "child_profiles"
    __table_args__ = (
        CheckConstraint("length(id) = 36"),
        CheckConstraint("length(submission_token) = 36"),
        CheckConstraint(f"length(trim(display_name)) BETWEEN 1 AND {MAX_NAME_LENGTH}"),
        _choices("age_group", AGE_GROUPS),
        _choices("avatar", AVATARS),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    submission_token: Mapped[str] = mapped_column(String(36), unique=True)
    display_name: Mapped[str] = mapped_column(String(MAX_NAME_LENGTH))
    age_group: Mapped[str | None] = mapped_column(String(16))
    avatar: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime)


@dataclass(frozen=True)
class ChildProfile:
    id: str
    display_name: str
    age_group: str | None
    avatar: str | None
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None

    @property
    def archived(self):
        return self.archived_at is not None


def _snapshot(row):
    return ChildProfile(
        row.id, row.display_name, row.age_group, row.avatar,
        as_utc(row.created_at), as_utc(row.updated_at), as_utc(row.archived_at),
    )


def _uuid(value, message):
    try:
        return str(UUID(str(value)))
    except ValueError:
        raise ProfileError(message) from None


def _details(display_name, age_group, avatar):
    name = display_name.strip() if isinstance(display_name, str) else ""
    if not name:
        raise ProfileError("Enter a display name.")
    if len(name) > MAX_NAME_LENGTH:
        raise ProfileError(f"Use {MAX_NAME_LENGTH} characters or fewer for the display name.")
    if not all(character.isprintable() or character in "\u200c\u200d" for character in name):
        raise ProfileError("Use a display name without line breaks or control characters.")
    if age_group is not None and age_group not in AGE_GROUPS:
        raise ProfileError("Choose an age group from the list.")
    if avatar is not None and (not isinstance(avatar, str) or avatar not in AVATARS):
        raise ProfileError("Choose a preset avatar from the list.")
    return dict(display_name=name, age_group=age_group, avatar=avatar)


class ChildProfileRepository:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get(PROFILE_DB_ENV) or DEFAULT_PATH).expanduser().resolve()
        research_db = make_url(get_database_url())
        if (
            research_db.get_backend_name() == "sqlite"
            and research_db.database
            and self.path == Path(research_db.database).resolve()
        ):
            raise ProfileStorageError("Profile storage requires a separate database.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            URL.create("sqlite", database=str(self.path)), hide_parameters=True,
        )
        try:
            with self._engine.begin() as connection:
                connection.execute(CreateTable(_ProfileRow.__table__, if_not_exists=True))
        except SQLAlchemyError:
            self.close()
            raise
        self._session = sessionmaker(self._engine)

    def close(self):
        self._engine.dispose()

    def create(self, display_name, *, submission_token, age_group=None, avatar=None):
        token = _uuid(submission_token, "Reopen the add profile form and try again.")
        details = _details(display_name, age_group, avatar)
        now = utcnow()
        with self._session.begin() as session:
            statement = insert(_ProfileRow).values(
                id=str(uuid4()), submission_token=token, created_at=now, updated_at=now, **details,
            )
            session.execute(statement.on_conflict_do_nothing(index_elements=["submission_token"]))
            row = session.scalar(select(_ProfileRow).where(_ProfileRow.submission_token == token))
            return _snapshot(row)

    def get(self, profile_id):
        with self._session() as session:
            return _snapshot(self._get(session, profile_id))

    def list_profiles(self, *, archived=False):
        return self._list(archived=archived)

    def _list(self, *, archived=None):
        statement = select(_ProfileRow).order_by(_ProfileRow.created_at, _ProfileRow.id)
        if archived is not None:
            column = _ProfileRow.archived_at
            statement = statement.where(column.is_not(None) if archived else column.is_(None))
        with self._session() as session:
            return [_snapshot(row) for row in session.scalars(statement)]

    def list_all(self):
        """Read active and archived profiles together for a consistent page snapshot."""
        return self._list()

    def edit(self, profile_id, display_name, *, age_group=None, avatar=None):
        return self._change(
            profile_id, archived=False, error="Restore this profile before editing it.",
            **_details(display_name, age_group, avatar),
        )

    def archive(self, profile_id):
        return self._change(profile_id, archived=False, archived_at=utcnow())

    def restore(self, profile_id):
        return self._change(profile_id, archived=True, archived_at=None)

    @staticmethod
    def _get(session, profile_id):
        key = _uuid(profile_id, "This profile could not be found.")
        row = session.get(_ProfileRow, key)
        if row is None:
            raise ProfileError("This profile could not be found.")
        return row

    def _change(self, profile_id, *, archived, error=None, **values):
        key = _uuid(profile_id, "This profile could not be found.")
        column = _ProfileRow.archived_at
        status = column.is_not(None) if archived else column.is_(None)
        with self._session.begin() as session:
            result = session.execute(
                update(_ProfileRow).where(_ProfileRow.id == key, status)
                .values(updated_at=utcnow(), **values)
            )
            row = self._get(session, key)
            if not result.rowcount and error:
                raise ProfileError(error)
            return _snapshot(row)

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from streamlit.testing.v1 import AppTest

from db.child_profiles import (
    AGE_GROUPS, MAX_NAME_LENGTH, PROFILE_DB_ENV, ChildProfileRepository, ProfileError,
    ProfileStorageError,
)


@pytest.fixture
def repository(tmp_path):
    with closing(ChildProfileRepository(tmp_path / "profiles.db")) as repo:
        yield repo


def test_create_retrieve_and_reopen(tmp_path, monkeypatch):
    path = tmp_path / "local" / "profiles.db"
    monkeypatch.setenv(PROFILE_DB_ENV, str(path))
    with closing(ChildProfileRepository()) as repository:
        assert repository.list_profiles() == repository.list_profiles(archived=True) == []
        profile = repository.create(
            "  සිංහල ළමයා  ", age_group=AGE_GROUPS[1], avatar="owl", submission_token=uuid4(),
        )
        assert UUID(profile.id).version == 4
        assert profile.display_name == "සිංහල ළමයා"
        assert profile.created_at == profile.updated_at
        assert not profile.archived
    with closing(ChildProfileRepository(path)) as reopened:
        assert reopened.get(profile.id) == profile
        assert reopened.list_profiles() == [profile]


def test_duplicate_tokens_and_identical_names_are_distinct_concerns(repository):
    token = uuid4()
    first = repository.create("Alex", submission_token=token)
    assert repository.create("Different retry data", submission_token=token) == first
    second = repository.create("Alex", submission_token=uuid4())
    assert first.id != second.id
    archived = repository.archive(first.id)
    assert repository.create("Alex", submission_token=token) == archived
    assert repository.list_profiles() == [second]
    assert repository.list_profiles(archived=True) == [archived]


def test_concurrent_duplicate_submissions_create_one_profile(tmp_path):
    path = tmp_path / "profiles.db"
    token = uuid4()

    def submit():
        with closing(ChildProfileRepository(path)) as repository:
            return repository.create("Alex", submission_token=token)

    with ThreadPoolExecutor(max_workers=2) as workers:
        first, second = workers.map(lambda _: submit(), range(2))
    with closing(ChildProfileRepository(path)) as repository:
        assert first == second
        assert repository.list_profiles() == [first]


def test_edit_archive_restore_and_profile_isolation(repository):
    first = repository.create("Alex", submission_token=uuid4())
    second = repository.create("Alex", submission_token=uuid4())
    edited = repository.edit(first.id, "  Sammie ", age_group=AGE_GROUPS[2], avatar="bear")
    assert edited.id == first.id
    assert edited.created_at == first.created_at
    assert edited.updated_at >= first.updated_at
    assert edited.display_name == "Sammie"
    assert edited.age_group == AGE_GROUPS[2]
    assert edited.avatar == "bear"
    archived = repository.archive(first.id)
    assert archived.archived_at is not None
    assert repository.archive(first.id) == archived
    assert repository.list_profiles() == [second]
    with pytest.raises(ProfileError, match="Restore"):
        repository.edit(first.id, "Changed")
    restored = repository.restore(first.id)
    assert restored.id == first.id
    assert restored.archived_at is None
    assert restored.display_name == edited.display_name
    assert restored.age_group == edited.age_group
    assert restored.avatar == edited.avatar
    assert repository.restore(first.id) == restored
    assert repository.get(second.id) == second


@pytest.mark.parametrize("name,options", [
    ("", {}), (" \u2003 ", {}), ("x" * (MAX_NAME_LENGTH + 1), {}),
    ("Name", {"age_group": "invalid"}), ("Name", {"avatar": "invalid"}), ("one\ntwo", {}),
])
def test_create_and_edit_validate_input(repository, name, options):
    existing = repository.create("Valid", submission_token=uuid4())
    with pytest.raises(ProfileError):
        repository.create(name, submission_token=uuid4(), **options)
    with pytest.raises(ProfileError):
        repository.edit(existing.id, name, **options)
    assert repository.list_profiles() == [existing]


@pytest.mark.parametrize("profile_id", ["invalid-uuid", str(uuid4())])
def test_invalid_and_unknown_ids(repository, profile_id):
    for operation in (repository.get, repository.archive, repository.restore):
        with pytest.raises(ProfileError, match="could not be found"):
            operation(profile_id)
    with pytest.raises(ProfileError, match="could not be found"):
        repository.edit(profile_id, "Valid")
    with pytest.raises(ProfileError, match="Reopen"):
        repository.create("Valid", submission_token="invalid")


def test_failed_update_rolls_back_and_next_write_succeeds(repository):
    profile = repository.create("Original", submission_token=uuid4())
    with repository._engine.begin() as connection:
        connection.execute(text("""
            CREATE TRIGGER reject_profile_update AFTER UPDATE ON child_profiles
            WHEN NEW.display_name = 'Rejected'
            BEGIN SELECT RAISE(FAIL, 'test write failure'); END
        """))
    with pytest.raises(IntegrityError):
        repository.edit(profile.id, "Rejected")
    assert repository.get(profile.id) == profile
    assert repository.edit(profile.id, "Accepted").display_name == "Accepted"


def test_research_db_path_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "research.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    with pytest.raises(ProfileStorageError, match="separate"):
        ChildProfileRepository(path)
    assert not path.exists()


def test_profile_page_create_edit_archive_restore(tmp_path, monkeypatch):
    path = tmp_path / "profiles.db"
    monkeypatch.setenv(PROFILE_DB_ENV, str(path))
    page = Path(__file__).resolve().parents[1] / "ui" / "dashboard_pages" / "child_profile.py"
    at = AppTest.from_file(str(page), default_timeout=20).run()
    assert not at.exception
    assert "No active child profiles" in at.info[0].value
    at.button[0].click().run()
    assert at.error[0].value == "Enter a display name."
    at.text_input[0].set_value(" Alex ")
    at.selectbox[0].select(AGE_GROUPS[1])
    at.selectbox[1].select("owl")
    at.button[0].click().run()
    assert at.success[0].value == "Child profile added."
    assert at.text_input[0].value == ""
    assert any("Monitoring data has not been connected" in item.value for item in at.caption)
    assert not at.get("plotly_chart")
    with closing(ChildProfileRepository(path)) as repository:
        profile = repository.list_profiles()[0]
        assert profile.display_name == "Alex"
        ui_key = at.session_state["dashboard_profile_ui_keys"][profile.id]
        assert profile.id != ui_key
        assert any(ui_key in str(item.proto) for item in at.get("flex_container"))
        assert all(profile.id not in str(item.proto) for item in [*at.get("flex_container"), *at.button])
        assert profile.age_group == AGE_GROUPS[1]
        assert profile.avatar == "owl"
        at.text_input[1].set_value("New name")
        next(button for button in at.button if button.label == "Save changes").click().run()
        assert at.success[0].value == "Child profile updated."
        assert repository.get(profile.id).display_name == "New name"
        next(button for button in at.button if button.label == "Archive profile").click().run()
        assert "Confirm" in at.error[0].value
        assert not repository.get(profile.id).archived
        at.checkbox[0].check()
        next(button for button in at.button if button.label == "Archive profile").click().run()
        assert at.success[0].value == "Child profile archived."
        assert repository.list_profiles() == []
        next(button for button in at.button if button.label == "Restore Profile").click().run()
        assert at.success[0].value == "Child profile restored."
        assert repository.list_profiles()[0].id == profile.id
        assert at.session_state["dashboard_profile_ui_keys"][profile.id] == ui_key
    assert not at.exception


def test_profile_page_handles_unavailable_database(tmp_path, monkeypatch):
    monkeypatch.setenv(PROFILE_DB_ENV, str(tmp_path))
    page = Path(__file__).resolve().parents[1] / "ui" / "dashboard_pages" / "child_profile.py"
    at = AppTest.from_file(str(page), default_timeout=20).run()
    assert not at.exception
    assert "Child profiles are unavailable" in at.error[0].value


def test_profile_widget_keys_are_distinct_for_duplicate_names(tmp_path, monkeypatch):
    path = tmp_path / "profiles.db"
    monkeypatch.setenv(PROFILE_DB_ENV, str(path))
    with closing(ChildProfileRepository(path)) as repository:
        first = repository.create("Same name", submission_token=uuid4())
        second = repository.create("Same name", submission_token=uuid4())
    page = Path(__file__).resolve().parents[1] / "ui" / "dashboard_pages" / "child_profile.py"
    at = AppTest.from_file(str(page), default_timeout=30).run()
    keys = dict(at.session_state["dashboard_profile_ui_keys"])
    assert set(keys) == {first.id, second.id}
    assert len(set(keys.values())) == 2 and not set(keys).intersection(keys.values())
    at.run()
    assert dict(at.session_state["dashboard_profile_ui_keys"]) == keys
    assert not at.exception

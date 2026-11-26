import sys
from pathlib import Path

import pytest

# Make the top-level `db` package importable when pytest runs from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import db  # noqa: E402


@pytest.fixture
def temp_db(tmp_path):
    """A fresh SQLite file per test -- never touches project.db."""
    engine = db.init_db(f"sqlite:///{tmp_path / 'test.db'}")
    yield engine
    engine.dispose()


def follow_page_link(at, area, title):
    link = next(link.proto for link in area.get("page_link") if link.proto.label == title)
    # AppTest.switch_page hashes the filename, but this app sets explicit URL paths.
    at._page_hash = link.page_script_hash
    return at.run()


@pytest.fixture
def navigate_dashboard():
    return lambda at, title: follow_page_link(at, at.sidebar, title)

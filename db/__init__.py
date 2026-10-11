"""
Project database layer: one module per component, all sharing db/base.py.

    from db import init_db
    from db.comp3 import save_comp3_output, get_session

    init_db()                      # creates tables in DATABASE_URL (default sqlite:///project.db)
    row_id = save_comp3_output(record)

Linking components
------------------
Every component table carries a `session_id` column (indexed). It is the
shared key: join comp1/comp2/comp3 rows on it to see one session end to end.
It is a plain indexed column rather than a FOREIGN KEY because Components 1
and 2 write *before* Component 3 has a row for that session (they are what
trigger it). If a strict FK is ever needed, add a small `sessions` registry
table in base.py and point every component's session_id at it.

Adding a component (comp1.py / comp2.py)
----------------------------------------
Copy the shape of comp3.py once that component's output schema is final:

  1. db/compN.py
       - a Pydantic model mirroring docs/interface-contracts/<schema>.json
       - `class CompNXxx(Base)` with __tablename__ = "compN_<name>", a
         `session_id` column (index=True; unique=True only if one row per
         session), indexes on the columns you query, CheckConstraints for enums
       - save_compN_output(data) -> int, plus the query helpers you need
  2. db/base.py: uncomment the matching `from db import compN` in init_db()
     so its table gets created.
  3. tests/test_compN_db.py using the `temp_db` fixture from tests/conftest.py.
"""
from db.base import (Base, DatabaseError, DuplicateRecordError,
                     InvalidRecordError, SessionLocal, configure,
                     get_database_url, get_engine, init_db, session_scope)

__all__ = [
    "Base", "DatabaseError", "DuplicateRecordError", "InvalidRecordError",
    "SessionLocal", "configure", "get_database_url", "get_engine", "init_db",
    "session_scope",
]

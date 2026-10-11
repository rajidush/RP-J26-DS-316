#!/usr/bin/env python3
"""
Load Component 3's exported sessions into the shared database.

    python scripts/import_comp3_sessions.py
    python scripts/import_comp3_sessions.py path/to/educator_sessions.jsonl
    python scripts/import_comp3_sessions.py --database-url sqlite:///other.db

Default input: component3_socratic_educator/out/educator_sessions.jsonl.
The database is DATABASE_URL (default sqlite:///project.db) unless
--database-url is given. Every row goes through db.comp3.save_comp3_output(),
so it is validated against the contract. Only the contract fields plus
child_id are kept; anything else in a row is dropped before saving.
Re-running is safe: rows whose session_id is already stored are skipped.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import DuplicateRecordError, InvalidRecordError, init_db  # noqa: E402
from db.comp3 import Comp3Output, save_comp3_output  # noqa: E402

DEFAULT_INPUT = ROOT / "component3_socratic_educator" / "out" / "educator_sessions.jsonl"
KEEP_FIELDS = set(Comp3Output.model_fields)  # contract fields + child_id


def import_jsonl(path, database_url=None):
    """Import one JSONL file. Returns {"inserted", "skipped", "invalid", "errors"}
    where errors is a list of (line_number, message) for invalid rows."""
    init_db(database_url)
    counts = {"inserted": 0, "skipped": 0, "invalid": 0, "errors": []}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"expected a JSON object, got {type(row).__name__}")
                save_comp3_output({k: v for k, v in row.items() if k in KEEP_FIELDS})
            except DuplicateRecordError:
                counts["skipped"] += 1
            except (InvalidRecordError, ValueError) as exc:  # JSONDecodeError is a ValueError
                counts["invalid"] += 1
                counts["errors"].append((line_no, str(exc).splitlines()[0]))
            else:
                counts["inserted"] += 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", nargs="?", default=str(DEFAULT_INPUT), help="JSONL file to import")
    parser.add_argument("--database-url", help="override DATABASE_URL for this run")
    args = parser.parse_args(argv)

    counts = import_jsonl(args.path, args.database_url)
    print(f"{args.path}\n  inserted: {counts['inserted']}\n  skipped (duplicate session_id): "
          f"{counts['skipped']}\n  invalid: {counts['invalid']}")
    for line_no, message in counts["errors"][:20]:
        print(f"    line {line_no}: {message}")
    if len(counts["errors"]) > 20:
        print(f"    ... and {len(counts['errors']) - 20} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Fully On-Device AI for Detecting Harmful Content and Guiding Children's Digital Safety
**Project code:** J26-DS-316

## Team & component status

| Component | Owner | Status | Folder |
|---|---|---|---|
| 1 — Screen Monitoring | IT23377844 | `component1_screen_monitoring/` |
| 2 — Hate-Speech Detection | IT23209152 | `component2_hate_speech_detection/` |
| 3 — Socratic Educator | IT23155466 | `component3_socratic_educator/` |
| 4 — Behavioral Profiling / XAI | IT23135116 | `component4_profiling_xai/` |



## Run the whole system (this is "the software")

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python run_demo.py
```

One command, one laptop, no internet required beyond installing the
Python packages once. See `integration/README.md` for what "integrated"
means here and why no server/container setup is needed.

## Run everything's tests

```bash
pytest -v
```

## Shared database (`db/`)

Component outputs are stored through `db/` (SQLAlchemy 2 + Pydantic v2).
Tables link on `session_id`. Only Component 3 is stored so far; see
`db/__init__.py` for how to add Components 1 and 2.

```bash
pip install -r requirements.txt

# Create the tables (default: ./project.db; override with DATABASE_URL,
# e.g. postgresql+psycopg://user:pass@host/dbname)
python -c "from db import init_db; init_db()"

# Run the DB tests (they use a temporary SQLite file, never project.db)
pytest tests -v
```

```python
from db import init_db
from db.comp3 import save_comp3_output, get_escalated_sessions, risk_summary

init_db()
save_comp3_output(record)   # validated against comp3_to_comp4.schema.json
```

## Git identity (Component 2 / Dilnuka)

Always commit as your GitHub-linked identity so contributions show under **Dilnuka**.
See `docs/git-commit-identity.md`. One-time setup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_git_identity.ps1
```

## Repo layout

```
├── docs/
│   ├── architecture/            diagrams
│   ├── interface-contracts/     the shared schemas every component is built against
│   ├── git-commit-identity.md   Dilnuka commit/push identity rules
│   └── meeting-notes/
├── component1_screen_monitoring/
├── component2_hate_speech_detection/
├── component3_socratic_educator/
├── component4_profiling_xai/
├── integration/                 wires all 4 together into one runnable program
├── scripts/                     env setup
└── run_demo.py                  <- the single entry point
```

## How each component should be built

1. Work against your own `mock_inputs/` and the shared schema in
   `docs/interface-contracts/` — you should never need another
   component's real code to develop or test your own.
2. Every output your component produces must validate against its
   schema. If you need a new field, add it to the schema first and log
   it in `docs/interface-contracts/CHANGELOG.md` before anyone codes
   against it.
3. Only touch `integration/` once your component is ready to be swapped
   in for its stub.

## Branching

- `main` — protected, always working
- Feature branches per person: `c3/fsm-controller`, `c1/frame-capture`,
  `c2/classifier-v1`, `c4/report-format`, etc.
- Small PRs, one teammate reviews before merge
- Tag `pp1-milestone` the week of the presentation

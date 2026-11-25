# Fully On-Device AI for Detecting Harmful Content and Guiding Children's Digital Safety
**Project code:** J26-DS-316

## Team & component status

| Component | Owner | Status | Folder |
|---|---|---|---|
| 1 - Screen Monitoring | IT23377844 | `component1_screen_monitoring/` |
| 2 - Hate-Speech Detection | IT23209152 | `component2_hate_speech_detection/` |
| 3 - Socratic Educator | IT23155466 | `component3_socratic_educator/` |
| 4 - Behavioral Profiling / XAI | IT23135116 | `component4_profiling_xai/` |



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

# Load Component 3's exported sessions (C001-C040, with child_id) -- safe to re-run
python scripts/import_comp3_sessions.py

# Run the DB tests (they use a temporary SQLite file, never project.db)
pytest tests -v
```

```python
from db import init_db
from db.comp3 import save_comp3_output, get_escalated_sessions, risk_summary

init_db()
save_comp3_output(record)   # validated against comp3_to_comp4.schema.json
```

`comp3_sessions` also has an optional `child_id` column. It is **not** in the
contract: Component 3's data harness adds it to its exported rows so sessions
can be grouped per child; records without it are still accepted. `init_db()`
adds the column to an existing database automatically (a one-off
`ALTER TABLE`), so an older `project.db` keeps its rows. To start clean
instead, delete `project.db` and re-run the two commands above.

## Running the UI

The Parent Dashboard is the Streamlit application for Component 4.

Install the dependencies once with `python -m pip install -r requirements.txt`. Then run these Windows Command Prompt commands:

```cmd
cd /d E:\RP-J26-DS-316
python -m streamlit run ui/app.py --server.address 127.0.0.1
```

- **Emotional Pattern** shows a child's current emotional pattern and recent trend using local Component 3 session records or clearly labeled demo records.
- **Behavioral Anomaly** shows unusual activity, changes from the child's own history, an explanation of the AI assessment, supportive next steps, and independent safety alerts.

Both functions run locally with their saved models. The default session database is `project.db`. PP2 methods and evaluation results are available in the [Function 02 guide](component4_profiling_xai/PP2_FUNCTION02.md) and [optimization report](component4_profiling_xai/PP2_FUNCTION02_OPTIMIZATION.md).

The Emotional Pattern page uses the trained artifacts committed in
`component4_profiling_xai/models/` (about 15 KB), with `scikit-learn==1.6.1`:

- Function 1: `emotional_pattern_kmeans.pkl`, `emotional_pattern_scaler.pkl`,
  `emotional_pattern_metadata.json` (or point `C4_F1_ARTIFACT_DIR` at another folder).

There is no fallback. If a model's files are missing or cannot be loaded
(including a scikit-learn version mismatch), its page shows a red error card
with the reason and the fix, and no result. The sidebar shows
"Function 1 model: loaded / not loaded". A child with fewer than 3 sessions
gets "Not enough data yet" instead of a pattern.

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
   `docs/interface-contracts/` - you should never need another
   component's real code to develop or test your own.
2. Every output your component produces must validate against its
   schema. If you need a new field, add it to the schema first and log
   it in `docs/interface-contracts/CHANGELOG.md` before anyone codes
   against it.
3. Only touch `integration/` once your component is ready to be swapped
   in for its stub.

## Branching

- `main` - protected, always working
- Feature branches per person: `c3/fsm-controller`, `c1/frame-capture`,
  `c2/classifier-v1`, `c4/report-format`, etc.
- Small PRs, one teammate reviews before merge
- Tag `pp1-milestone` the week of the presentation

"""
generate_educator_dataset.py

Builds the Component 3 -> Component 4 dataset by running the REAL FSMController
on simulated child sessions.

    1. build_seeds()      scenario seeds (child, archetype, risk category, risk level, time)
    2. child replies      written by a local LLM through distilabel (or canned templates with --offline)
    3. FSMController.run  the actual Component 3 state machine produces EvaluateOutput
    4. export             schema-validated JSONL + CSV, plus seed_vs_output.csv for evaluation

All child replies are generated (step 2) BEFORE any session runs (step 3), so
distilabel and the FSM never call the local model server at the same time.

Usage
    python generate_educator_dataset.py --offline --stub-generator     # no LLM anywhere, smoke test
    python generate_educator_dataset.py --stub-generator               # distilabel + LM Studio (default backend)
    python generate_educator_dataset.py --backend ollama --model gemma3:1b
    python generate_educator_dataset.py --rescore --out out           # re-score saved replies after FSM changes
Outputs (in --out, default ./out)
    scenario_seeds.csv, child_replies.jsonl, educator_sessions.jsonl, educator_sessions.csv, seed_vs_output.csv
"""
import argparse
import copy
import csv
import dataclasses
import inspect
import json
import random
import re
import sys
import time
import typing
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Run this file from the project root (the folder that contains src/).
from src.fsm_controller import FSMController
from src.grammar_decoder import GrammarConstrainedGenerator
from src.model_client import LM_STUDIO_URL
from src.schemas import _CONTRACTS_DIR, _validate_against_schema, EmotionalState, RiskLevel, TriggerPayload

ROOT = Path(__file__).resolve().parent
# Single source of truth: the same contract file EvaluateOutput.validate() uses.
SCHEMA_FILE = "comp3_to_comp4.schema.json"
SCHEMA_PATH = _CONTRACTS_DIR / SCHEMA_FILE
DEFAULT_MOCK_DIR = ROOT / "mock_inputs"
# Inbound contract -> mock input shaped like that component's real output.
# TriggerPayload.validate() picks the contract from source_component, and each
# contract only allows its own risk categories, so every category must be sent
# from the component that is allowed to raise it.
INBOUND = {
    "comp1_to_comp3.schema.json": "trigger_from_component1.json",
    "comp2_to_comp3.schema.json": "trigger_from_component2.json",
}
LM_STUDIO_BASE = LM_STUDIO_URL.split("/v1/")[0]  # e.g. http://localhost:1234


def risk_category_enum():
    """The enum class TriggerPayload.risk_category is typed with (no need to guess its name)."""
    return typing.get_type_hints(TriggerPayload)["risk_category"]


def trigger_field_names():
    try:
        return {f.name for f in dataclasses.fields(TriggerPayload)}
    except TypeError:  # not a dataclass
        return set(inspect.signature(TriggerPayload).parameters)

# archetype -> (description used in the LLM prompt, state we INTEND it to produce)
ARCHETYPES = {
    "calm_reflective": ("thoughtful and open; explains what happened and what they might do differently", "calm"),
    "defensive": ("defensive and short with words; says it was nothing or not their fault", "defensive"),
    "distressed": ("upset, anxious or scared; writes in a shaky, emotional way", "distressed"),
    "withdrawn": ("withdrawn; gives almost no answer, e.g. 'idk' or silence", "unclear"),
}

# Canned replies used with --offline (no LLM needed). 4 replies per archetype.
TEMPLATES = {
    "calm_reflective": [
        "I clicked on it by accident but then I felt a bit weird about it",
        "I think I should have closed it right away instead of watching",
        "Next time I could tell my mum or just go to a different video",
        "Yeah I understand why it is not a good thing to look at",
    ],
    "defensive": ["it was nothing", "so what", "not my fault", "whatever"],
    "distressed": [
        "I did not mean to see it and now I feel really scared about it",
        "I do not know why it keeps showing up and I am worried I am in trouble",
        "I feel sick and I do not want to talk about it",
        "please do not tell anyone I am so embarrassed",
    ],
    "withdrawn": ["", "idk", "", "idk"],
}

RISK_BANDS = {  # matches the confidence thresholds in FSMController._evaluate
    "low": (0.30, 0.49),
    "moderate": (0.50, 0.84),
    "high": (0.85, 0.99),
}
RISK_WEIGHTS = {"low": 0.40, "moderate": 0.40, "high": 0.20}
ARCH_WEIGHTS = {"calm_reflective": 0.35, "defensive": 0.30, "distressed": 0.20, "withdrawn": 0.15}


def pick(rng, weights):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def build_seeds(n_children, weeks, rng, start_child=1, total_population=40):
    cats = [c.value for c in risk_category_enum()]
    start = datetime(2026, 7, 20, tzinfo=timezone.utc)  # 10 weeks ends before today (3 Oct 2026)
    seeds, sid = [], 0
    max_c = max(total_population, start_child + n_children - 1)
    end_child = start_child + n_children - 1
    for c in range(1, max_c + 1):
        child_id = f"C{c:03d}"
        primary = pick(rng, ARCH_WEIGHTS)
        for _ in range(rng.randint(12, 25)):
            sid += 1
            level = pick(rng, RISK_WEIGHTS)
            lo, hi = RISK_BANDS[level]
            when = start + timedelta(days=rng.randrange(weeks * 7), hours=rng.randint(15, 21), minutes=rng.randint(0, 59))
            archetype = primary if rng.random() < 0.7 else pick(rng, ARCH_WEIGHTS)
            cat = rng.choice(cats)
            score = round(rng.uniform(lo, hi), 2)
            if start_child <= c <= end_child:
                seeds.append({
                    "child_id": child_id,
                    "session_id": f"S-{sid:06d}",
                    "archetype": archetype,
                    "risk_category": cat,
                    "intended_risk_level": level,
                    "confidence_score": score,
                    "timestamp": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                })
    return seeds


# ---------------------------------------------------------------- child replies
def build_prompt(seed):
    desc = ARCHETYPES[seed["archetype"]][0]
    return (
        "You are simulating a child aged 9 to 12 chatting with a friendly safety assistant. "
        "A safety app noticed the child came across something inappropriate online "
        f"({seed['risk_category'].replace('_', ' ')}). "
        f"The child is {desc}. "
        "Write exactly 4 short replies this child might type in a conversation, one per line, "
        "no numbering, no quotation marks. Output only the 4 replies, with no introduction or comments. "
        "Keep everything mild: never quote or describe the "
        "inappropriate content itself and use no slurs or graphic detail."
    )


_QUOTES = "\"'\u201c\u201d\u2018\u2019"
# Lines the model writes ABOUT the replies rather than AS the child.
_META = re.compile(
    r"^((okay|ok|sure|alright)\W+)?(here are|here is|here's)\b"   # "Okay, here are four replies ..."
    r"|:\s*$"                                                    # any line introducing a list
    r"|\b(would you like|let me know)\b",                       # trailing offers of more help
    re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Normalize smart quotes and ellipses to clean ASCII equivalents."""
    return (
        text.replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2026", "...")
    )


def parse_replies(text, archetype):
    lines = []
    for raw in (text or "").splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", raw).strip().strip(_QUOTES).strip()
        line = normalize_text(line)
        if line and not _META.search(line):
            lines.append(line)
    return lines[:4] if len(lines) >= 2 else TEMPLATES[archetype]


def replies_offline(seeds, rng):
    return {s["session_id"]: list(TEMPLATES[s["archetype"]]) for s in seeds}


def lmstudio_default_model(base_url):
    """First chat model id served by LM Studio (GET <base>/v1/models), skipping embedding models."""
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=10) as resp:
        ids = [m["id"] for m in json.loads(resp.read())["data"]]
    if not ids:
        raise RuntimeError(f"LM Studio at {base_url} reports no loaded models")
    return next((i for i in ids if "embed" not in i.lower()), ids[0])


def make_llm(backend, model, base_url, host):
    # Written against distilabel 1.5.x; check import paths if your version differs.
    if backend == "lmstudio":
        from distilabel.models import OpenAILLM
        return OpenAILLM(
            model=model, base_url=f"{base_url.rstrip('/')}/v1", api_key="lm-studio",
            timeout=120, max_retries=2,
            generation_kwargs={"temperature": 0.9, "max_new_tokens": 160},
        )
    from distilabel.models import OllamaLLM
    return OllamaLLM(model=model, host=host,
                     generation_kwargs={"options": {"temperature": 0.9, "num_predict": 160}})


def replies_distilabel(seeds, llm, batch_size=16):
    from distilabel.pipeline import Pipeline
    from distilabel.steps import LoadDataFromDicts
    from distilabel.steps.tasks import TextGeneration

    rows = [{"session_id": s["session_id"], "instruction": build_prompt(s)} for s in seeds]
    with Pipeline(name="child-replies") as pipe:
        load = LoadDataFromDicts(data=rows)
        gen = TextGeneration(llm=llm, input_batch_size=batch_size)
        load >> gen
    distiset = pipe.run(use_cache=False)
    table = distiset["default"]["train"]
    arch = {s["session_id"]: s["archetype"] for s in seeds}
    replies = {r["session_id"]: parse_replies(r["generation"], arch[r["session_id"]]) for r in table}
    fallback = sum(replies[sid] is TEMPLATES[arch[sid]] for sid in replies)
    if fallback:
        print(f"WARNING: {fallback}/{len(replies)} sessions fell back to canned replies (unparseable LLM output)")
    return replies


# ---------------------------------------------------------------- run the component
class StubInterceptGenerator(GrammarConstrainedGenerator):
    """--stub-generator: deterministic, network-free generator.

    The real generator sends INTERCEPT to LM Studio (src/model_client.py) and
    uses the template pool for INQUIRE. This stub uses the existing template
    pool + grammar validate() for every state, i.e. the pre-SLM code path.
    Only the INTERCEPT opening text changes; no EvaluateOutput field depends
    on that text today.
    """

    def generate(self, state, risk_category, attempt=0):
        candidate = self._candidate(state, risk_category, attempt=attempt)
        return candidate if self.validate(state, candidate) else self._safe_fallback(state)


def load_templates(mock_dir=DEFAULT_MOCK_DIR):
    """risk_category value -> mock trigger dict from the component allowed to send it.
    The category split is read from the inbound contracts, not hard-coded."""
    by_cat = {}
    for schema_file, mock_file in INBOUND.items():
        schema = json.loads((_CONTRACTS_DIR / schema_file).read_text(encoding="utf-8"))
        template = json.loads((Path(mock_dir) / mock_file).read_text(encoding="utf-8"))
        for cat in schema["properties"]["risk_category"]["enum"]:
            by_cat[cat] = template
    missing = {c.value for c in risk_category_enum()} - by_cat.keys()
    if missing:
        raise ValueError(f"No inbound contract allows risk categories {sorted(missing)}")
    return by_cat


def make_trigger(seed, templates):
    """Copy the other fields from the matching mock input JSON, then override the ones we simulate."""
    names = trigger_field_names()
    template = copy.deepcopy(templates[seed["risk_category"]])
    kwargs = {k: v for k, v in template.items() if k in names}
    kwargs.update(
        session_id=seed["session_id"],
        risk_category=risk_category_enum()(seed["risk_category"]),
        confidence_score=seed["confidence_score"],
        timestamp=seed["timestamp"],
    )
    return TriggerPayload(**kwargs)


def parse_timestamp(ts):
    """jsonschema does not enforce format: date-time by default, so check it ourselves."""
    return datetime.fromisoformat(ts[:-1] + "+00:00" if ts.endswith("Z") else ts)


def make_controller(stub_generator=False):
    return FSMController(generator=StubInterceptGenerator() if stub_generator else None)


def run_sessions(seeds, replies, templates, stub_generator=False):
    fsm = make_controller(stub_generator)
    records = []
    for s in seeds:
        script = replies[s["session_id"]]
        state = {"i": 0, "last": ""}

        def respond(_prompt, script=script, state=state):
            if state["i"] < len(script):
                state["last"] = script[state["i"]]
                state["i"] += 1
            return state["last"]

        out, _transcript = fsm.run(make_trigger(s, templates), respond)
        rec = {
            "session_id": out.session_id,
            "timestamp": s["timestamp"],  # simulated time; the contract requires a timestamp
            "risk_level": out.risk_level.value,
            "emotional_state": out.emotional_state.value,
            "self_regulation_shown": bool(out.self_regulation_shown),
            "escalation_flag": bool(out.escalation_flag),
            "dialogue_turns": int(out.dialogue_turns),
            "dialogue_summary": out.dialogue_summary,
        }
        _validate_against_schema(rec, SCHEMA_FILE)  # raises if the record breaks the contract
        parse_timestamp(rec["timestamp"])           # raises ValueError if not ISO 8601
        records.append(rec)
    return records


# ---------------------------------------------------------------- export
def write_csv(path, rows, append=False):
    if not rows:
        return
    p = Path(path)
    file_exists = p.exists() and p.stat().st_size > 0
    mode = "a" if append else "w"
    with open(p, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not (append and file_exists):
            w.writeheader()
        w.writerows(rows)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    ap.add_argument("--children", type=int, default=40, help="number of children to simulate in this run")
    ap.add_argument("--start-child", type=int, default=1, help="1-indexed child ID to start with (e.g. 11 for C011)")
    ap.add_argument("--total-population", type=int, default=40, help="total cohort size for globally consistent session numbering")
    ap.add_argument("--append", action="store_true", help="append generated records to existing CSV and JSONL files in --out")
    ap.add_argument("--batch-size", type=int, default=16, help="distilabel batch size (lower to 4 or 8 to reduce peak heat)")
    ap.add_argument("--weeks", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offline", action="store_true",
                    help="use canned child replies instead of distilabel. NOT network-free on its own: "
                         "the FSM's INTERCEPT state still calls LM Studio unless --stub-generator is also set")
    ap.add_argument("--stub-generator", action="store_true",
                    help="swap GrammarConstrainedGenerator for a deterministic stub that uses the template "
                         "pool for INTERCEPT instead of the live model (record fields do not depend on that text)")
    ap.add_argument("--backend", choices=["lmstudio", "ollama"], default="lmstudio",
                    help="distilabel backend for child replies (ignored with --offline)")
    ap.add_argument("--model", default=None,
                    help="model id. Default: first chat model from LM Studio's /v1/models, or gemma3:1b for ollama")
    ap.add_argument("--base-url", default=LM_STUDIO_BASE,
                    help=f"LM Studio server root, /v1 is appended (default from src/model_client.py: {LM_STUDIO_BASE})")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama host (--backend ollama)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--limit", type=int, default=0, help="only run the first N sessions (for smoke tests)")
    ap.add_argument("--rescore", action="store_true",
                    help="re-run the FSM over the seeds + replies already saved in --out (no model calls) "
                         "and rewrite the session outputs; use after changing FSM/evaluation logic")
    ap.add_argument("--mock-dir", default=str(DEFAULT_MOCK_DIR),
                    help="folder with trigger_from_component1.json / trigger_from_component2.json")
    return ap


def run(**overrides):
    """Programmatic entry point (used by tests). Accepts the CLI options as
    keyword arguments, e.g. run(offline=True, stub_generator=True, limit=10, out=tmp)."""
    a = build_parser().parse_args([])
    unknown = set(overrides) - set(vars(a))
    if unknown:
        raise TypeError(f"unknown option(s): {sorted(unknown)}")
    for k, v in overrides.items():
        setattr(a, k, v)
    return _run(a)


def _run(a):
    if a.rescore:
        return _rescore(a)
    t0 = time.time()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    seeds = build_seeds(a.children, a.weeks, rng, start_child=a.start_child, total_population=a.total_population)
    if a.limit:
        seeds = seeds[: a.limit]

    # Step 2: ALL child replies first ...
    if a.offline:
        replies = replies_offline(seeds, rng)
        reply_source = "offline templates"
    else:
        if a.backend == "lmstudio":
            model = a.model or lmstudio_default_model(a.base_url)
        else:
            model = a.model or "gemma3:1b"
        reply_source = f"distilabel/{a.backend}/{model}"
        print(f"child replies: {reply_source}")
        replies = replies_distilabel(seeds, make_llm(a.backend, model, a.base_url, a.host), batch_size=a.batch_size)
    t_replies = time.time() - t0
    # Seeds and replies are written together, only once the replies exist: if
    # distilabel crashes, nothing is written, so re-running the chunk with
    # --append cannot duplicate its seeds. If the FSM step below fails instead,
    # use --rescore rather than regenerating the replies.
    write_csv(out / "scenario_seeds.csv", seeds, append=a.append)
    jsonl_mode = "a" if a.append else "w"
    with open(out / "child_replies.jsonl", jsonl_mode, encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps({"session_id": s["session_id"], "archetype": s["archetype"],
                                "replies": replies[s["session_id"]]}, ensure_ascii=False) + "\n")

    # Step 3: ... then the sessions (no overlap with distilabel's model calls).
    records = run_sessions(seeds, replies, load_templates(a.mock_dir), stub_generator=a.stub_generator)
    export, cmp_rows = write_session_outputs(out, seeds, records, append=a.append)

    print(f"child replies: {reply_source} | generator: {'stub (template INTERCEPT)' if a.stub_generator else 'real (LM Studio INTERCEPT)'}")
    print_summary(out, seeds, cmp_rows)
    print(f"time: replies {t_replies:.1f}s, total {time.time() - t0:.1f}s")
    return {"seeds": seeds, "replies": replies, "records": records, "export": export,
            "comparison": cmp_rows, "out": out}


def _rescore(a):
    """--rescore: re-run the FSM over the seeds + child replies already saved in
    --out, without calling any model. Use it after changing FSM/evaluation logic
    so the existing (LLM-written) replies are re-scored rather than regenerated.
    Rewrites educator_sessions.*, seed_vs_output.csv and a de-duplicated
    scenario_seeds.csv; child_replies.jsonl is left untouched."""
    out = Path(a.out)
    seeds, seen = [], set()
    with open(out / "scenario_seeds.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["session_id"] not in seen:  # chunk re-runs could append a seed twice
                seen.add(row["session_id"])
                row["confidence_score"] = float(row["confidence_score"])
                seeds.append(row)
    with open(out / "child_replies.jsonl", encoding="utf-8") as f:
        replies = {r["session_id"]: r["replies"] for r in map(json.loads, f)}
    missing = [s["session_id"] for s in seeds if s["session_id"] not in replies]
    if missing:
        raise ValueError(f"{len(missing)} seeds have no saved replies, e.g. {missing[:3]}")
    old_path = out / "educator_sessions.jsonl"
    old = {}
    if old_path.exists():
        with open(old_path, encoding="utf-8") as f:
            old = {r["session_id"]: r for r in map(json.loads, f)}

    # No EvaluateOutput field depends on the INTERCEPT text, so the network-free
    # stub generator gives the same records as the live one.
    records = run_sessions(seeds, replies, load_templates(a.mock_dir), stub_generator=True)
    write_csv(out / "scenario_seeds.csv", seeds)
    export, cmp_rows = write_session_outputs(out, seeds, records, append=False)

    changes = [(r["session_id"], k, old[r["session_id"]][k], r[k])
               for r in records if r["session_id"] in old
               for k in r if k in old[r["session_id"]] and old[r["session_id"]][k] != r[k]]
    print(f"rescored {len(records)} sessions from saved replies (no model calls)")
    print(f"field changes vs previous educator_sessions.jsonl: {len(changes)}")
    for sid, k, before, after in changes:
        print(f"  {sid} {k}: {before} -> {after}")
    print_summary(out, seeds, cmp_rows)
    return {"seeds": seeds, "replies": replies, "records": records, "export": export,
            "comparison": cmp_rows, "out": out, "changes": changes}


def write_session_outputs(out, seeds, records, append=False):
    child_of = {s["session_id"]: s["child_id"] for s in seeds}
    export = [{"child_id": child_of[r["session_id"]], **r} for r in records]  # child_id added by harness
    with open(out / "educator_sessions.jsonl", "a" if append else "w", encoding="utf-8") as f:
        for r in export:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_csv(out / "educator_sessions.csv", export, append=append)

    by_sid = {r["session_id"]: r for r in records}
    cmp_rows = [{
        "session_id": s["session_id"], "child_id": s["child_id"], "archetype": s["archetype"],
        "intended_state": ARCHETYPES[s["archetype"]][1], "output_state": by_sid[s["session_id"]]["emotional_state"],
        "intended_risk": s["intended_risk_level"], "output_risk": by_sid[s["session_id"]]["risk_level"],
    } for s in seeds]
    write_csv(out / "seed_vs_output.csv", cmp_rows, append=append)
    return export, cmp_rows


def print_summary(out, seeds, cmp_rows):
    n = len(cmp_rows)
    state_ok = sum(r["intended_state"] == r["output_state"] for r in cmp_rows)
    risk_ok = sum(r["intended_risk"] == r["output_risk"] for r in cmp_rows)
    print(f"{n} sessions, {len({s['child_id'] for s in seeds})} children -> {out}/")
    print("emotional_state counts:", dict(Counter(r["output_state"] for r in cmp_rows)))
    print("risk_level counts     :", dict(Counter(r["output_risk"] for r in cmp_rows)))
    print(f"intended vs output emotional_state agreement: {state_ok / n:.1%}")
    print(f"intended vs output risk_level agreement     : {risk_ok / n:.1%}")
    for arch in ARCHETYPES:
        rows = [r for r in cmp_rows if r["archetype"] == arch]
        if rows:
            got = dict(Counter(r["output_state"] for r in rows))
            print(f"  {arch:<16} intended={ARCHETYPES[arch][1]:<10} n={len(rows):<4} output={got}")


def main():
    # distilabel logs emoji through rich; on a Windows cp1252 console/pipe that
    # floods stderr with UnicodeEncodeError tracebacks. Force UTF-8 output.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    _run(build_parser().parse_args())


if __name__ == "__main__":
    main()
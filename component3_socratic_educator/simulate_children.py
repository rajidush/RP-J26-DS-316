"""
simulate_children.py

Simulated children aged 8-15 talking to the real Component 3 controller
(ticket pp2-validation/06; spec stories 29, 30). A model plays each child and
answers every educator line as it comes. The sessions give two pools:

    classifier_test  one session per classifier-test scenario of the PP2 draw
                     (40 per age band), the human raters' reply labels (ticket 09)
    train_dev        one session per training or validation scenario (24 per band),
                     the owner's labels for the fine-tuned arm (ticket 14); never evidence

About 1 in 5 children is played by Gemma (the generator-sensitivity slice), the
rest by Gemini; results are reported per generator. Held-out children and
scenarios never appear in the train/dev pool.

    plan    before any child exists: each session's pool, scenario, band, archetype,
            interest, generator and trigger, from the scenario draw and a recorded seed
    run     simulate one generator's planned sessions; a model error (a quota running
            out) stops the run cleanly, and the same command resumes it
    finish  check every planned session is there and that the pools match the draw,
            then write one file per pool (the label-sheets input) and a manifest

The educator is the template arm with INTERCEPT's template line, because the
notebook has no local model (EDUCATOR_SETUP). Run from component3_socratic_educator/:

    python simulate_children.py plan --draw <scenario_draw.json> [--seed N] [--tag x] [--out <dir>]
    python simulate_children.py run --plan <dir>/children.json --generator gemini|gemma
                                    --model <id> --backend colab|aistudio|hf|local [--base-url <url>]
                                    [--pool classifier_test|train_dev] [--per-band N] [--limit N] [--fold-system]
    python simulate_children.py finish --dir <dir> --draw <scenario_draw.json>
"""
import argparse
import dataclasses
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from chat_client import BACKENDS, LOCAL_BASE_URL, base_url_of, is_resumable, make_chat, runtime
from generate_educator_dataset import (ARCHETYPES, RISK_BANDS, RISK_WEIGHTS, SCHEMA_FILE, StubInterceptGenerator,
                                       load_templates, make_trigger, normalize_text, pick, session_record)
from src import config
from src.fsm_controller import FSMController
from src.schemas import _validate_against_schema

HERE = Path(__file__).resolve().parent
PLAN_SEED = 20261124
POOLS = {"classifier_test": ("classifier_test",), "train_dev": ("train", "validation")}
CHILD_PREFIX = {"classifier_test": "CT", "train_dev": "CD"}
GENERATORS = ("gemini", "gemma")
GEMMA_SHARE = 0.2   # spec: about 1 in 5 children, the generator-sensitivity slice
CHILD_TEMPERATURE = 0.9
# Thinking tokens can count against the limit, and Colab's proxy reserves quota for it (8 Oct).
CHILD_MAX_TOKENS = 1024
CHILD_SEED = "not recorded (the chat endpoints take no seed)"
MAX_REPLY_CHARS = 300   # the stored reply; the raw output is kept whole in child_outputs
REPLY_TRIES = 3
EDUCATOR_SETUP = ("FSMController with the rules assessor; INQUIRE template arm; INTERCEPT template line "
                  "(StubInterceptGenerator: no local model in the notebook)")

# How a child of each band types, and how long a message runs (spec story 29: child prompts per band).
# Attitude comes from the archetype, never from the band.
BAND_VOICE = {
    "8-10": ("You are a child aged 8 to 10. Type like a child that age: simple everyday words, one idea at a "
             "time, little punctuation, now and then a spelling slip.", 15),
    "11-13": ("You are a child aged 11 to 13. Type like a child that age: short and casual, some slang, "
              "lowercase is fine.", 20),
    "14-15": ("You are a teenager aged 14 to 15. Type like a teen texting: brief and casual.", 30),
}
# Seeded per child, so children of the same band, archetype and category are not the same child.
INTERESTS = ("football", "drawing", "video games", "music", "dancing", "animals", "building things", "videos",
             "reading", "swimming")
# What the app noticed, in plain words; never the category code, never the content itself.
SITUATIONS = {
    "explicit_visual": "a rude adult picture",
    "violence": "a violent video",
    "self_harm_imagery": "a picture about someone hurting themselves",
    "hate_speech": "hateful messages about a group of people",
    "cyberbullying": "mean messages aimed at you or a friend",
    "grooming_language": "messages from an older stranger who wanted to keep things secret",
    "self_harm_language": "posts about hurting yourself",
    "unknown_flagged": "something the app thought was unsafe",
}
_SPEAKER = re.compile(r"^\s*(?:child|me|kid|reply|answer|response)\s*:\s*", re.I)
_NOT_THE_CHILD = re.compile(r"\bas an ai\b|\blanguage model\b|^\s*(?:here is|here's)\b", re.I)


def _refuse(problem, details, command):
    sys.exit(f"{command} refused: {problem}\n" + "\n".join(f"  {d}" for d in details))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, text):
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _load_draw(path, command):
    draw = json.loads(Path(path).read_text(encoding="utf-8"))
    if draw.get("dataset") != "inquire-pp2" or "scenarios" not in draw:
        _refuse(f"{Path(path).name} is not a PP2 scenario draw",
                ["pass the scenario_draw.json from inquire_finetune.py pp2-draw"], command)
    return draw


# ---------------------------------------------------------------- plan
def _spread_gemma(children, rng):
    """About 1 in 5 children to Gemma, taken in turn from each archetype, so every archetype has its share."""
    by_archetype = {a: [c for c in children if c["archetype"] == a] for a in ARCHETYPES}
    for group in by_archetype.values():
        rng.shuffle(group)
    order = list(ARCHETYPES)
    rng.shuffle(order)
    wanted, picked, turn = round(len(children) * GEMMA_SHARE), [], 0
    while len(picked) < wanted:
        group = by_archetype[order[turn % len(order)]]
        if group:
            picked.append(group.pop())
        turn += 1
    gemma = {c["session_id"] for c in picked}
    for c in children:
        c["generator"] = "gemma" if c["session_id"] in gemma else "gemini"


def plan_children(draw, rng):
    """One child per scenario of each pool's roles: archetypes balanced within each band, Gemma about 1 in 5.
    `risk_level` is the trigger's drawn level; its confidence score is drawn inside that level's range."""
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    children = []
    for pool, roles in POOLS.items():
        number = 0
        for band in config.AGE_BANDS:
            scenarios = sorted((s for s in draw["scenarios"] if s["role"] in roles and s["age_band"] == band),
                               key=lambda s: s["scenario_id"])
            archetypes = [list(ARCHETYPES)[i % len(ARCHETYPES)] for i in range(len(scenarios))]
            rng.shuffle(archetypes)
            group = []
            for scenario, archetype in zip(scenarios, archetypes):
                number += 1
                level = pick(rng, RISK_WEIGHTS)
                low, high = RISK_BANDS[level]
                child_id = f"{CHILD_PREFIX[pool]}{number:03d}"
                when = start + timedelta(days=rng.randrange(28), hours=rng.randint(15, 21), minutes=rng.randint(0, 59))
                group.append({"session_id": f"SIM-{child_id}", "child_id": child_id, "pool": pool,
                              "scenario_id": scenario["scenario_id"], "scenario_role": scenario["role"],
                              "age_band": band, "risk_category": scenario["risk_category"], "archetype": archetype,
                              "interest": rng.choice(INTERESTS), "risk_level": level,
                              "confidence_score": round(rng.uniform(low, high), 2),
                              "timestamp": when.strftime("%Y-%m-%dT%H:%M:%SZ")})
            _spread_gemma(group, rng)
            children += group
    return children


def cmd_plan(args):
    name = f"simulated_children_{date.today().isoformat()}" + (f"_{args.tag}" if args.tag else "")
    out = Path(args.out or HERE / "evidence" / "evaluate" / name)
    if (out / "children.json").exists():
        _refuse(f"{out} already holds a plan", ["pass --tag or a new --out; a recorded plan is never redrawn"], "plan")
    draw = _load_draw(args.draw, "plan")
    children = plan_children(draw, random.Random(args.seed))
    plan = {
        "dataset": "pp2-simulated-children",
        "seed": args.seed,
        "draw": {"file": Path(args.draw).name, "sha256": _sha256(args.draw), "seed": draw["seed"]},
        "rule": "one child per scenario: classifier_test scenarios -> classifier_test pool, train and validation "
                "scenarios -> train_dev pool; archetypes balanced within each band; about 1 in 5 children per "
                "band to Gemma, taken in turn from each archetype; drawn before any child reply exists",
        "children": children,
    }
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "children.json", json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out / 'children.json'}: {len(children)} sessions, seed {args.seed}")
    for pool in POOLS:
        counts = Counter(c["generator"] for c in children if c["pool"] == pool)
        print(f"  {pool}: {sum(counts.values())} ({', '.join(f'{g} {n}' for g, n in sorted(counts.items()))})")


# ---------------------------------------------------------------- the child
def child_prompt(child):
    """Who the child model plays: how a child of the band types, the archetype, an interest, and what the app
    noticed, in plain words. Feelings may be strong; what was seen is never described."""
    voice, words = BAND_VOICE[child["age_band"]]
    return (f"{voice} You like {child['interest']}. A safety app on your laptop noticed "
            f"{SITUATIONS[child['risk_category']]} on your screen, and a friendly assistant is now chatting with "
            f"you about it. You are {ARCHETYPES[child['archetype']][0]}. Reply with only the message you would "
            f"type: one message, under {words} words, always at least one word, in your own words. Stay in "
            "character and never say you are an AI. You may say how you feel, however strongly, but never "
            "describe or quote what you saw, and use no slurs or graphic detail.")


def child_messages(child, history, line):
    """The chat the child model sees: its prompt as the system message, then the educator's lines as the user
    and the child's own replies as the assistant."""
    messages = [{"role": "system", "content": child_prompt(child)}]
    for said, reply in history:
        messages += [{"role": "user", "content": said}, {"role": "assistant", "content": reply}]
    return messages + [{"role": "user", "content": line}]


def fold_system(messages):
    """The same chat with the system message moved into the first user message, for models that reject a
    system role (AI Studio's Gemma: "Developer instruction is not enabled")."""
    system, first, rest = messages[0], messages[1], messages[2:]
    return [{"role": "user", "content": f"{system['content']}\n\n{first['content']}"}, *rest]


def clean_reply(text):
    """The child's message: its first line, without a speaker label or quotes; empty if the model stepped out."""
    first = next((line for line in (text or "").splitlines() if line.strip()), "")
    reply = " ".join(normalize_text(_SPEAKER.sub("", first)).strip().strip("\"'").split())
    return "" if _NOT_THE_CHILD.search(reply) else reply[:MAX_REPLY_CHARS]


class ModelError(RuntimeError):
    """The child model could not be reached or refused the request (a quota, the network): stop and resume later."""


class ChildFailed(RuntimeError):
    """The child model answered, but gave no usable reply in REPLY_TRIES tries: skip this child for now."""


def make_ask(backend, model, base_url=None):
    """The child model: messages -> text through `chat_client` (Colab, AI Studio, Hugging Face or a local server)."""
    return make_chat(backend, model, CHILD_TEMPERATURE, CHILD_MAX_TOKENS, base_url)


def simulate_session(child, ask, controller, templates, model_id):
    """One session through the real controller, the child model answering each line. The record keeps the
    child's prompt and every raw model output (retries included) beside the cleaned replies."""
    history, outputs = [], []

    def respond(line):
        raw = []
        for _ in range(REPLY_TRIES):
            try:
                text = ask(child_messages(child, history, line))
            except Exception as error:
                if not is_resumable(error):   # a bad key or model name: waiting will not help
                    raise
                raise ModelError(f"{type(error).__name__}: {str(error)[:200]}") from error
            raw.append(text)
            reply = clean_reply(text)
            if reply:
                history.append((line, reply))
                outputs.append(raw)
                return reply
        raise ChildFailed(f"no usable reply in {REPLY_TRIES} tries; last output {raw[-1][:120]!r}")

    out, transcript = controller.run(make_trigger(child, templates), respond, age_band=child["age_band"])
    turns = [{**dataclasses.asdict(t), "state": t.state.name} for t in transcript.turns]
    return {**{k: child[k] for k in ("session_id", "child_id", "pool", "scenario_id", "scenario_role", "age_band",
                                     "archetype", "interest", "risk_category", "risk_level", "generator")},
            "model_id": model_id, "child_prompt": child_prompt(child), "child_outputs": outputs, "turns": turns,
            "record": session_record(out, child["timestamp"])}


def _finished_sessions(path):
    """The sessions written so far. A torn last line (the runtime died mid-write) is cut off, so the run resumes;
    a broken line anywhere else is refused."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    sessions = []
    for n, line in enumerate(lines):
        try:
            sessions.append(json.loads(line))
        except json.JSONDecodeError:
            if n != len(lines) - 1:
                _refuse(f"{path.name} line {n + 1} is broken", ["it is not the last line; fix the file by hand"], "run")
            _write(path, "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in sessions))
            print(f"  cut a torn last line from {path.name}; that session will be simulated again")
    return sessions


def _append(path, session):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(session, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def cmd_run(args, ask=None):
    plan_path = Path(args.plan)
    plan, folder = json.loads(plan_path.read_text(encoding="utf-8")), plan_path.parent
    sessions_path = folder / "sessions.jsonl"
    done = {s["session_id"] for s in _finished_sessions(sessions_path)}
    mine = [c for c in plan["children"] if c["generator"] == args.generator]
    todo = [c for c in mine if c["session_id"] not in done and (args.pool is None or c["pool"] == args.pool)]
    if args.per_band is not None:
        todo = [c for band in config.AGE_BANDS for c in [c for c in todo if c["age_band"] == band][:args.per_band]]
    if args.limit is not None:
        todo = todo[:args.limit]
    if args.base_url and args.backend != "local":
        _refuse("--base-url is only for --backend local", ["colab, aistudio and hf have fixed addresses"], "run")
    ask = ask or make_ask(args.backend, args.model, args.base_url or LOCAL_BASE_URL)
    send = (lambda chat: ask(fold_system(chat))) if args.fold_system else ask
    controller, templates = FSMController(generator=StubInterceptGenerator()), load_templates()

    started, written, failed, stopped = datetime.now(timezone.utc).isoformat(timespec="seconds"), 0, [], None
    for child in todo:
        try:
            session = simulate_session(child, send, controller, templates, args.model)
        except ModelError as error:   # a quota or the network: keep what is done, resume later
            stopped = f"{child['session_id']}: {error}"
            break
        except ChildFailed as error:  # this child only: the next run tries it again
            failed.append(f"{child['session_id']}: {error}")
            continue
        _append(sessions_path, session)
        written += 1

    record_path = folder / f"run_{args.generator}.json"
    earlier = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else None
    served = getattr(ask, "served", None)
    record = {"generator": args.generator, "model": args.model,
              "served_models": sorted(served) if served else "not recorded", "backend": args.backend,
              "base_url": base_url_of(args.backend, args.base_url),
              "settings": {"temperature": CHILD_TEMPERATURE, "max_tokens": CHILD_MAX_TOKENS},
              "child_seed": CHILD_SEED, "fold_system": args.fold_system, "plan_seed": plan["seed"],
              "educator": EDUCATOR_SETUP, "runtime": runtime(), "started": started,
              "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sessions_written": written,
              "children_without_a_usable_reply": failed, "stopped_by": stopped,
              "earlier_runs": ([{k: v for k, v in earlier.items() if k != "earlier_runs"}]
                               + earlier.get("earlier_runs", []) if earlier else [])}
    _write(record_path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    left = sum(1 for c in mine if c["session_id"] not in done) - written
    print(f"{args.generator}: wrote {written} sessions to {sessions_path}; {left} still to simulate")
    for line in failed:
        print(f"  no usable reply, try again later: {line}")
    if stopped:
        print(f"  stopped at {stopped}\n  finished sessions are kept: rerun the same command to resume")


# ---------------------------------------------------------------- finish
def _raw_reply_in(session):
    """A child's multi-word reply found in the Component 4 record (single words such as "bully" can occur in a
    label like "cyberbullying" by chance)."""
    record = json.dumps(session["record"], ensure_ascii=False)
    return any(t["child_response"] and " " in t["child_response"].strip() and t["child_response"] in record
               for t in session["turns"])


def cmd_finish(args):
    folder = Path(args.dir)
    plan = json.loads((folder / "children.json").read_text(encoding="utf-8"))
    if (folder / "manifest.json").exists():
        _refuse(f"{folder} is already finished", ["a finished run is never rewritten"], "finish")
    draw = _load_draw(args.draw, "finish")
    if _sha256(args.draw) != plan["draw"]["sha256"]:
        _refuse("this is not the draw the plan was made from", [f"{Path(args.draw).name}: hash differs"], "finish")
    sessions = _finished_sessions(folder / "sessions.jsonl")
    counts = Counter(s["session_id"] for s in sessions)
    planned = {c["session_id"]: c for c in plan["children"]}
    missing = [sid for sid in planned if sid not in counts]
    if missing:
        _refuse(f"{len(missing)} planned sessions are not simulated yet",
                [", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""),
                 "run each generator until it has none left"], "finish")
    roles = {s["scenario_id"]: s["role"] for s in draw["scenarios"]}
    problems = [f"{sid}: appears {n} times" for sid, n in sorted(counts.items()) if n > 1]
    problems += [f"{sid}: not in the plan" for sid in counts if sid not in planned]
    problems += [f"{s['session_id']}: scenario {s['scenario_id']} is {roles.get(s['scenario_id'])} in the draw, not "
                 f"a {s['pool']} scenario" for s in sessions if roles.get(s["scenario_id"]) not in POOLS[s["pool"]]]
    for key, label in (("child_id", "child"), ("scenario_id", "scenario")):
        pools = {}
        for s in sessions:
            pools.setdefault(s[key], set()).add(s["pool"])
        problems += [f"{label} {value} is in both pools" for value, p in sorted(pools.items()) if len(p) > 1]
    for s in sessions:
        try:
            _validate_against_schema(s["record"], SCHEMA_FILE)
        except Exception as error:
            problems.append(f"{s['session_id']}: record breaks the contract: {str(error)[:120]}")
    problems += [f"{s['session_id']}: the Component 4 record holds a raw child reply" for s in sessions
                 if _raw_reply_in(s)]
    if problems:
        _refuse("the sessions do not pass the checks", problems, "finish")

    outputs = {pool: folder / f"{pool}_sessions.jsonl" for pool in POOLS}
    by_pool = {pool: sorted((s for s in sessions if s["pool"] == pool), key=lambda s: s["session_id"])
               for pool in POOLS}
    for pool, path in outputs.items():
        _write(path, "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in by_pool[pool]))
    runs = {g: json.loads((folder / f"run_{g}.json").read_text(encoding="utf-8"))
            for g in GENERATORS if (folder / f"run_{g}.json").exists()}
    manifest = {
        "dataset": "pp2-simulated-children",
        "plan": {"file": "children.json", "sha256": _sha256(folder / "children.json"), "seed": plan["seed"],
                 "draw": plan["draw"]},
        "outputs": {path.name: _sha256(path) for path in outputs.values()},
        "counts": {pool: {band: dict(sorted(Counter(s["generator"] for s in rows if s["age_band"] == band).items()))
                          for band in config.AGE_BANDS} for pool, rows in by_pool.items()},
        "archetypes": {pool: {band: dict(sorted(Counter(s["archetype"] for s in rows if s["age_band"] == band).items()))
                              for band in config.AGE_BANDS} for pool, rows in by_pool.items()},
        "train_dev_roles": dict(sorted(Counter(s["scenario_role"] for s in by_pool["train_dev"]).items())),
        "models": {g: sorted({s["model_id"] for s in sessions if s["generator"] == g}) for g in GENERATORS},
        "educator": EDUCATOR_SETUP,
        "runs": runs,
        "checks": {"held_out_children_in_train_dev": 0, "held_out_scenarios_in_train_dev": 0,
                   "sessions_off_their_draw_role": 0, "records_failing_the_contract": 0,
                   "records_holding_a_raw_reply": 0},
        "limitations": ["every child reply is simulated by a language model, not written by a child",
                        "the educator lines are the template arm's, from before any ticket 27 rewording",
                        "the Gemma slice has 2 children per archetype per band in the test pool"],
    }
    _write(folder / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {folder}: " + ", ".join(f"{p.name} ({len(by_pool[pool])} sessions)" for pool, p in outputs.items())
          + ", manifest.json")


# ---------------------------------------------------------------- cli
def build_parser():
    ap = argparse.ArgumentParser(description="Simulated children 8-15 through the real controller (plan, run, "
                                             "finish).")
    sub = ap.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="draw each session's pool, scenario, band, archetype and generator")
    plan.add_argument("--draw", required=True, help="the scenario_draw.json from inquire_finetune.py pp2-draw")
    plan.add_argument("--seed", type=int, default=PLAN_SEED)
    plan.add_argument("--tag", help="added to the default folder name, for a second plan on the same day")
    plan.add_argument("--out", help="default: evidence/evaluate/simulated_children_<date>[_<tag>]")
    plan.set_defaults(func=cmd_plan)
    run = sub.add_parser("run", help="simulate one generator's planned sessions (resumable)")
    run.add_argument("--plan", required=True, help="the children.json written by plan")
    run.add_argument("--generator", required=True, choices=GENERATORS)
    run.add_argument("--model", required=True, help="the model id, e.g. google/gemini-2.5-pro or gemma-3-27b-it")
    run.add_argument("--backend", required=True, choices=BACKENDS,
                     help="hf: Hugging Face Inference Providers (HF_TOKEN); local: an OpenAI-compatible server on "
                          "this machine, such as LM Studio (see --base-url)")
    run.add_argument("--base-url", help=f"the local server's address (default {LOCAL_BASE_URL}, LM Studio)")
    run.add_argument("--pool", choices=tuple(POOLS), help="only this pool's children (a trial on train_dev)")
    run.add_argument("--per-band", type=int, help="at most this many children per age band (a trial)")
    run.add_argument("--limit", type=int, help="simulate at most this many sessions (a quota chunk)")
    run.add_argument("--fold-system", action="store_true",
                     help="send the child's prompt in the first user message (models with no system role, e.g. Gemma)")
    run.set_defaults(func=cmd_run)
    finish = sub.add_parser("finish", help="check the sessions; write one file per pool and a manifest")
    finish.add_argument("--dir", required=True, help="the folder holding children.json and sessions.jsonl")
    finish.add_argument("--draw", required=True, help="the scenario_draw.json the plan was made from")
    finish.set_defaults(func=cmd_finish)
    return ap


def main(argv=None, ask=None):
    """`ask` (messages -> text) replaces the model client in `run`; tests pass a fake child."""
    args = build_parser().parse_args(argv)
    if args.command == "run":
        cmd_run(args, ask)
    else:
        args.func(args)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()

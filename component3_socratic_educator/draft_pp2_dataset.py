"""
draft_pp2_dataset.py

Drafts the PP2 INQUIRE rows with a teacher model (ticket pp2-validation/08; spec
stories 31, 32). The owner reviews every row with `inquire_finetune.py review`,
then `inquire_finetune.py build-pp2` builds the dataset. Three resumable stages,
all writing into one folder:

    contexts  one scenario context per ID of scenario_slots.json (all 216), written
              per age band x risk category without knowing any ID's role (the
              draw's rule: no context is written knowing it is held out)
    rows      for every training and validation scenario, one row per Socratic
              move (owner, 25 Nov), its (easier, reply type) drawn from the
              combinations the controller can ask for, so training covers every
              stratum build-pp2 requires. Before clarify the child answers one of
              INTERCEPT's real fallback lines; otherwise the teacher writes the line
              the child answered, which must pass the live check. The teacher writes
              the child's reply of the row's type.
    drafts    two educator lines per row for the exact training prompt
              (grammar_decoder.inquire_prompt, ADR 0010); a line the live check
              would block (inquire_finetune.row_problems) is drafted again, and the
              band and move's most repeated closing questions are named to avoid

A quota or an overload stops a stage cleanly and the same command resumes it; a bad
key or model name stops it for good. Once the owner has started reviewing, drafting
is closed. Run from component3_socratic_educator/, or from
notebooks/draft_pp2_dataset_colab.ipynb:

    python draft_pp2_dataset.py contexts --slots <scenario_slots.json> --out <dir> --model <id> --backend <b>
    python draft_pp2_dataset.py rows --draw <scenario_draw.json> --out <dir> --model <id> --backend <b>
    python draft_pp2_dataset.py drafts --out <dir> --model <id> --backend <b>
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from chat_client import BACKENDS, base_url_of, is_resumable, make_chat, runtime
from draft_inquire_dataset import _REPLY_GUIDE, near_duplicates, parse_json_block
from inquire_finetune import _refuse, closing_sentence, dump_jsonl, load_jsonl, reachable_strata, row_problems, write_atomic
from simulate_children import BAND_VOICE
from src import config
from src.grammar_decoder import _ALERT_WORDS, _CATEGORY_WORDS, _FORBIDDEN_SUBSTRINGS
from src.grammar_decoder import (CONVERSATION_FENCE, INQUIRE_EASIER, INQUIRE_GUIDE, INQUIRE_PROMPT,
                                 GrammarConstrainedGenerator, inquire_prompt, inquire_violation)

PLAN_SEED = 20261125
DRAFTS_PER_ROW = 2
DRAFT_TRIES = 3   # attempts per draft slot; a line the live check blocks is drafted again
REQUEST_TRIES = 3   # attempts for a context batch or a row's conversation that comes back unusable
OVERUSED_AT = 3   # a closing question used this often in a band and move is named for the teacher to avoid
TEACHER_TEMPERATURE = 0.9
TEACHER_MAX_TOKENS = 4096   # thinking tokens can count against it
PROMPT_VERSION = "pp2-draft-v4"
# The small model's prompt, as text: a change to it changes this hash, recorded with every row.
TRAINING_PROMPT_SHA256 = hashlib.sha256(json.dumps([INQUIRE_PROMPT, INQUIRE_GUIDE, INQUIRE_EASIER, CONVERSATION_FENCE],
                                                   sort_keys=True).encode("utf-8")).hexdigest()
CONTEXTS_FILE, ROWS_FILE, RUN_FILE = "contexts.json", "pp2_drafts.jsonl", "draft_run.json"
NEAR_DUPLICATE = 0.7   # Jaccard over words, as the PP1 drafter
# Before clarify, the child answers what the controller actually says when no model is used.
OPENING_LINES = tuple(GrammarConstrainedGenerator._CANDIDATES["INTERCEPT"])
REPLY_GUIDE = {**_REPLY_GUIDE,
               "dont_know": "says they don't know or aren't sure (vary the wording; not always 'I don't know')",
               "explanation": "describes what they were doing or why, without describing what was on the screen"}
_FENCE_MARKS = re.compile(r"<{3}|>{3}")


class TeacherError(RuntimeError):
    """The teacher model could not be reached or refused for now (a quota, an overload): stop and resume later."""


def _call(ask, messages, command):
    """The teacher's answer. A quota or an overload stops the stage for a rerun; a bad request, key or model
    name stops it for good, since waiting will not help."""
    try:
        return ask(messages)
    except Exception as error:
        if not is_resumable(error):
            _refuse("the teacher refused the request", [f"{type(error).__name__}: {str(error)[:200]}",
                                                        "check the key and the model name"], command)
        raise TeacherError(f"{type(error).__name__}: {str(error)[:200]}") from error


def _rows_path(out):
    return Path(out) / ROWS_FILE


def _refuse_once_reviewed(out, command):
    """Drafting is closed once the owner has started reviewing: review decisions are never drafted over."""
    path = _rows_path(out)
    if path.exists() and any(r["review"]["status"] != "pending" for r in load_jsonl(path)):
        _refuse("the owner has started reviewing these rows", ["drafting is closed; copy the folder to draft again"],
                command)


# ---------------------------------------------------------------- contexts
def context_request(band, category, n, written=()):
    """A batch of contexts for one band and category. When some are already written (a context deleted because it
    nearly repeated another), the teacher sees them all, so it does not write the same typical scenario again."""
    shown = ("\n\nThese contexts are already written for this category. Each new one must describe a different "
             "situation from all of them, on another platform where you can, in different words:\n"
             + "\n".join(f"- {w}" for w in written)) if written else ""
    return [{"role": "user", "content": f"""You are helping build a sanitized, synthetic training dataset for a child digital-safety educator talking with children aged {band}.

Write {n} different one-sentence scenario contexts for the risk category "{category.replace('_', ' ')}". Each says, at a high level, what an on-screen detector flagged and what the child was doing, as it could happen to a child aged {band}.

Rules:
- Start every line with "A risk trigger was received for".
- High level only: never graphic, sexual or self-harm method details, slurs or quoted harmful text.
- No names, usernames, schools, cities or other identifying details.
- Vary the platform (games, video, social media, messaging, browser, school apps) and the situation.{shown}

Return ONLY a JSON array of {n} strings."""}]


def _repeats(fresh, contexts):
    """Whether a new context nearly repeats one already written, or another new one. Roles are never looked at."""
    pool = {**contexts, **{f"\0new{i}": text for i, text in enumerate(fresh)}}
    return any(a.startswith("\0new") or b.startswith("\0new")
               for a, b, _ in near_duplicates(pool, NEAR_DUPLICATE))


def _request_key(requests, band, category):
    """The first request of a cell keeps its key; later ones (rewrites) are kept beside it, never over it."""
    key = f"{band} / {category}"
    done = sum(1 for k in requests if k == key or k.startswith(f"{key} / rewrite "))
    return key if not done else f"{key} / rewrite {done}"


def cmd_contexts(args, ask):
    _refuse_once_reviewed(args.out, "contexts")
    slots = json.loads(Path(args.slots).read_text(encoding="utf-8"))
    if any("role" in s for s in slots.get("scenarios", [])):
        _refuse("this file holds the scenario roles", ["pass scenario_slots.json, not scenario_draw.json: no context "
                                                       "is written knowing its scenario is held out"], "contexts")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / CONTEXTS_FILE
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    contexts, requests = saved.get("contexts", {}), saved.get("requests", {})
    cells = {}
    for s in slots["scenarios"]:
        cells.setdefault((s["age_band"], s["risk_category"]), []).append(s["scenario_id"])
    stopped, unusable = None, []
    for (band, category), ids in cells.items():
        missing = sorted(i for i in ids if i not in contexts)
        if not missing:
            continue
        request = context_request(band, category, len(missing), [contexts[i] for i in sorted(ids) if i in contexts])
        try:
            for _ in range(REQUEST_TRIES):
                try:
                    lines = parse_json_block(_call(ask, request, "contexts"))
                except ValueError:
                    continue
                if isinstance(lines, list) and len(lines) == len(missing) and all(
                        isinstance(x, str) and x.strip() for x in lines) and not _repeats(
                        [x.strip() for x in lines], contexts):
                    contexts.update(zip(missing, (x.strip() for x in lines)))
                    requests[_request_key(requests, band, category)] = request[0]["content"]
                    break
            else:
                unusable.append(f"{band} / {category}")
        except TeacherError as error:
            stopped = f"{band} / {category}: {error}"
            break
        write_atomic(path, json.dumps({"dataset": "inquire-pp2", "prompt_version": PROMPT_VERSION,
                                       "contexts": dict(sorted(contexts.items())),
                                       "near_duplicates": near_duplicates(contexts, NEAR_DUPLICATE),
                                       "requests": requests}, ensure_ascii=False, indent=2) + "\n")
    left = sum(1 for s in slots["scenarios"] if s["scenario_id"] not in contexts)
    _record_run(out, "contexts", args, ask, {"contexts_written": len(contexts), "left": left, "unusable": unusable,
                                             "stopped_by": stopped})
    print(f"contexts: {len(contexts)} written to {path}; {left} still to write")
    _say_rerun(unusable, stopped, "no usable answer for")


# ---------------------------------------------------------------- rows
def plan_rows(draw, contexts, rng, seed):
    """One row per move for every training and validation scenario. Each move's (easier, reply type) runs through
    the combinations the controller can ask for, in a shuffled order per band, so every combination gets rows.
    A clarify row answers one of the opening lines; the others' answered line is written by the teacher."""
    combos = {m: [(easier, reply) for move, easier, reply in reachable_strata() if move == m]
              for m in config.SOCRATIC_MOVES}
    rows = []
    for role in ("train", "validation"):
        for band in config.AGE_BANDS:
            scenarios = sorted((s for s in draw["scenarios"] if s["role"] == role and s["age_band"] == band),
                               key=lambda s: s["scenario_id"])
            for move in config.SOCRATIC_MOVES:
                order = list(combos[move])
                rng.shuffle(order)
                for i, s in enumerate(scenarios):
                    easier, reply_type = order[i % len(order)]
                    opening = rng.choice(OPENING_LINES) if move == "clarify" and not easier else None
                    rows.append({"scenario_id": s["scenario_id"], "age_band": band, "risk_category": s["risk_category"],
                                 "move": move, "easier": easier, "reply_type": reply_type, "split": "train_or_val",
                                 "context": contexts[s["scenario_id"]], "previous_message": opening})
    moves = {m: i for i, m in enumerate(config.SOCRATIC_MOVES)}
    rows.sort(key=lambda r: (r["scenario_id"], moves[r["move"]]))
    return [{"example_id": f"P2R{n:04d}", **r, "child_reply": None, "prompt": None, "drafts": [], "completion": None,
             "review": {"status": "pending", "chosen_draft": None, "rewritten": False, "notes": ""},
             "provenance": {"prompt_version": PROMPT_VERSION, "plan_seed": seed,
                            "training_prompt_sha256": TRAINING_PROMPT_SHA256}}
            for n, r in enumerate(rows, 1)]


# The live check's word lists, told to the teacher: on the first run a row about a link in a messaging app failed
# 36 times because the teacher kept writing "that message" in an educator line.
BANNED_WORDS = (f"Never use these words: {', '.join(sorted(set(_ALERT_WORDS) | set(_CATEGORY_WORDS)))}; "
                f"and no word containing {', '.join(_FORBIDDEN_SUBSTRINGS)} (so not even \"skills\").")


def _answered_move(row):
    """The move of the line the child answered: the same move for an easier form, otherwise the move before."""
    moves = config.SOCRATIC_MOVES
    return row["move"] if row["easier"] else moves[moves.index(row["move"]) - 1]


def conversation_request(row):
    voice, words = BAND_VOICE[row["age_band"]]
    child = (f'"child_reply": the child\'s reply to that line. The child {REPLY_GUIDE[row["reply_type"]]}. {voice} '
             f"Under {words} words. Never describe or quote what was on the screen; no slurs, graphic detail or "
             "self-harm methods.")
    if row["previous_message"] is not None:
        return [{"role": "user", "content": f"""You are helping build a sanitized, synthetic training dataset for a child digital-safety educator talking with a child aged {row['age_band']}.

Situation (the educator never names it): {row['context']}

The educator has just said: "{row['previous_message']}"

Write {child}

Return ONLY a JSON object: {{"child_reply": "..."}}"""}]
    move = _answered_move(row)
    return [{"role": "user", "content": f"""You are helping build a sanitized, synthetic training dataset for a child digital-safety educator talking with a child aged {row['age_band']}.

Situation (the educator never names it): {row['context']}

Write two things:
1. "previous_message": the educator's last line, which the child is answering: the {move} question, which should {INQUIRE_GUIDE[move][row['age_band']]} At most one short, kind sentence, then one question ending in "?". Never name or describe what was on the screen, never name a feeling, and offer no choice ("... or ...?") unless it is the alternatives question. {BANNED_WORDS}
2. {child}

Return ONLY a JSON object: {{"previous_message": "...", "child_reply": "..."}}"""}]


def conversation_problems(row, previous, child):
    """Why a written conversation cannot go into a row: the line the child answered must pass the live check for
    its move (it goes into the training prompt), and the reply must be text that stays inside the prompt's fence."""
    problems = [] if child else ["no child reply"]
    if _FENCE_MARKS.search(child):
        problems.append("the child reply holds the fence's marks")
    if row["previous_message"] is None:
        reason = inquire_violation(previous, _answered_move(row), "") if previous else "no previous message"
        if reason:
            problems.append(f"previous message: {reason}")
    return problems


def cmd_rows(args, ask):
    _refuse_once_reviewed(args.out, "rows")
    out = Path(args.out)
    path = _rows_path(out)
    if not path.exists():
        draw = json.loads(Path(args.draw).read_text(encoding="utf-8"))
        if draw.get("dataset") != "inquire-pp2" or "roles" not in draw:
            _refuse(f"{Path(args.draw).name} is not a PP2 scenario draw", ["pass scenario_draw.json"], "rows")
        contexts = json.loads((out / CONTEXTS_FILE).read_text(encoding="utf-8"))["contexts"] \
            if (out / CONTEXTS_FILE).exists() else {}
        if set(contexts) != {s["scenario_id"] for s in draw["scenarios"]}:
            _refuse("contexts.json does not hold one context for each scenario of this draw",
                    ["run the contexts stage on this draw's scenario_slots.json"], "rows")
        roles = {s["scenario_id"]: s["role"] for s in draw["scenarios"]}
        crossed = [f"{a} ({roles[a]}) ~ {b} ({roles[b]}), {score}" for a, b, score in near_duplicates(contexts, NEAR_DUPLICATE)
                   if (roles[a] in ("train", "validation")) != (roles[b] in ("train", "validation"))]
        if crossed:
            _refuse("a training or validation context nearly repeats a held-out one", [
                *crossed, "delete the training or validation IDs' contexts from contexts.json and rerun contexts"], "rows")
        write_atomic(path, dump_jsonl(plan_rows(draw, contexts, random.Random(args.seed), args.seed)))
    rows = load_jsonl(path)
    stopped, unusable = None, []
    for row in rows:
        if row["child_reply"] is not None:
            continue
        request = conversation_request(row)
        try:
            for _ in range(REQUEST_TRIES):
                try:
                    obj = parse_json_block(_call(ask, request, "rows"))
                except ValueError:
                    continue
                obj = obj if isinstance(obj, dict) else {}
                child = str(obj.get("child_reply", "")).strip()
                previous = row["previous_message"] or str(obj.get("previous_message", "")).strip()
                if not conversation_problems(row, previous, child):
                    row.update(previous_message=previous, child_reply=child,
                               prompt=[{"role": "user", "content": inquire_prompt(
                                   row["move"], row["age_band"], row["easier"], previous, child)}])
                    row["provenance"]["conversation_request"] = request[0]["content"]
                    break
            else:
                unusable.append(row["example_id"])
        except TeacherError as error:
            stopped = f"{row['example_id']}: {error}"
            break
        write_atomic(path, dump_jsonl(rows))
    left = sum(r["child_reply"] is None for r in rows)
    _record_run(out, "rows", args, ask, {"rows": len(rows), "left": left, "unusable": unusable, "stopped_by": stopped},
                plan_seed=rows[0]["provenance"]["plan_seed"])
    print(f"rows: {len(rows) - left} of {len(rows)} have their conversation; {left} still to write")
    _say_rerun(unusable, stopped, "no usable answer for")


# ---------------------------------------------------------------- drafts
_DRAFT_GUIDE = {
    "stated_emotion": "Acknowledge the feeling with the child's exact word, then ask the move's question.",
    "dont_know": "Accept that they don't know, warmly and briefly, then ask the easier version of the question.",
    "off_topic": "Gently come back from the new topic, then ask the move's question.",
    "own_safer_plan": "Acknowledge the child's own plan in their words, then ask the move's question; never grant "
                      "permission to go back to what was stopped.",
    "explanation": "Build on what the child just said, in their words, then ask the move's question.",
}


def overused_closings(rows, row):
    """The closing questions already used OVERUSED_AT times or more in this row's band and move (ticket 07)."""
    used = Counter(closing_sentence(d) for r in rows if (r["age_band"], r["move"]) == (row["age_band"], row["move"])
                   for d in r["drafts"])
    return [text for text, n in used.most_common(3) if n >= OVERUSED_AT]


def draft_request(row, earlier, overused):
    differ = f"\n- Make it clearly different from: {' | '.join(earlier)}" if earlier else ""
    avoid = f"\n- Do not reuse these overused questions: {' | '.join(overused)}" if overused else ""
    return [{"role": "user", "content": f"""You are drafting reviewed training data for a small on-device digital-safety educator model. Write the ideal educator line for the input at the end, following its instructions exactly.

Hard rules (a line that breaks one is thrown away):
- At most one short sentence, then exactly one question, which ends the line; 30 words at most; plain words.
- Name a feeling only if the child used that exact word. Never invent facts, motives or the educator's feelings.
- Offer a choice ("... or ...?") only in the alternatives move. No tag questions ("..., right?"). Speak to the child as "you".
- Never name, describe or hint at what was on the screen; never mention alerts, warnings or that anyone was told.
- {BANNED_WORDS} If the child used one, say it another way.
- Never blame or lecture. Mention a trusted adult only in the alternatives move.
- The child's reply is "{row['reply_type']}": {_DRAFT_GUIDE[row['reply_type']]}{differ}{avoid}

Input (exactly what the small model will see):
---
{row['prompt'][0]['content']}
---

Return ONLY the educator's line, with no quotes, labels or JSON."""}]


def clean_line(text):
    """The drafted line: a JSON {"line": ...} unwrapped, the first line kept, quotes and labels removed."""
    text = (text or "").strip()
    try:
        obj = parse_json_block(text)
        if isinstance(obj, dict) and isinstance(obj.get("line"), str):
            text = obj["line"]
    except ValueError:
        pass
    first = next((line for line in text.splitlines() if line.strip()), "")
    first = re.sub(r"^\s*(?:educator|line|response)\s*:\s*", "", first, flags=re.I)
    return " ".join(first.strip().strip("\"'“”").split())


def cmd_drafts(args, ask):
    _refuse_once_reviewed(args.out, "drafts")
    out = Path(args.out)
    path = _rows_path(out)
    if not path.exists():
        _refuse("there are no rows yet", ["run the rows stage first"], "drafts")
    rows = load_jsonl(path)
    unwritten = [r["example_id"] for r in rows if r["prompt"] is None]
    if unwritten:
        _refuse(f"{len(unwritten)} rows have no conversation yet", [", ".join(unwritten[:12]),
                                                                    "finish the rows stage first"], "drafts")
    stopped = None
    for row in rows:
        if len(row["drafts"]) >= DRAFTS_PER_ROW:
            continue
        blocked = row["provenance"].setdefault("blocked_drafts", [])
        try:
            while len(row["drafts"]) < DRAFTS_PER_ROW:
                for _ in range(DRAFT_TRIES):
                    request = draft_request(row, row["drafts"], overused_closings(rows, row))
                    row["provenance"].setdefault("draft_request", request[0]["content"])
                    line = clean_line(_call(ask, request, "drafts"))
                    problems = row_problems(line, row) if line else ["empty"]
                    if not problems and line not in row["drafts"]:
                        row["drafts"].append(line)
                        break
                    blocked.append({"line": line, "problems": problems or ["same as the other draft"]})
                else:
                    break
        except TeacherError as error:
            stopped = f"{row['example_id']}: {error}"
            break
        finally:
            write_atomic(path, dump_jsonl(rows))
    short = [r["example_id"] for r in rows if len(r["drafts"]) < DRAFTS_PER_ROW]
    _record_run(out, "drafts", args, ask, {"rows": len(rows), "short_of_drafts": short, "stopped_by": stopped})
    print(f"drafts: {len(rows) - len(short)} of {len(rows)} rows have {DRAFTS_PER_ROW} drafts; {len(short)} short")
    _say_rerun([] if stopped else short, stopped, "every draft was blocked by the live check for")
    if not short:
        print(f"  ready for review: python inquire_finetune.py review --drafts {path}")


# ---------------------------------------------------------------- run record and cli
def _say_rerun(unusable, stopped, what):
    if unusable:
        print(f"  {what}: {', '.join(unusable[:12])}{' ...' if len(unusable) > 12 else ''}\n"
              "  rerun the same command to try them again")
    if stopped:
        print(f"  stopped at {stopped}\n  rerun the same command to resume")


def _record_run(out, stage, args, ask, result, plan_seed="not applicable"):
    path = Path(out) / RUN_FILE
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
    served = getattr(ask, "served", None)
    record["runs"].append({
        "stage": stage, "model": args.model, "served_models": sorted(served) if served else "not recorded",
        "backend": args.backend, "base_url": base_url_of(args.backend, args.base_url),
        "settings": {"temperature": TEACHER_TEMPERATURE, "max_tokens": TEACHER_MAX_TOKENS},
        "teacher_seed": "not recorded (the chat endpoints take no seed)", "plan_seed": plan_seed,
        "prompt_version": PROMPT_VERSION, "training_prompt_sha256": TRAINING_PROMPT_SHA256,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "runtime": runtime(),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), **result})
    write_atomic(path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def build_parser():
    ap = argparse.ArgumentParser(description="Draft the PP2 INQUIRE rows with a teacher model (contexts, rows, drafts).")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_teacher_options(p):
        p.add_argument("--out", required=True, help="the drafting folder (kept on Drive in the notebook)")
        p.add_argument("--model", required=True, help="the teacher model, e.g. gemini-3.8-flash")
        p.add_argument("--backend", required=True, choices=BACKENDS)
        p.add_argument("--base-url", help="the local server's address (--backend local)")

    contexts = sub.add_parser("contexts", help="one scenario context per scenario ID, written without roles")
    contexts.add_argument("--slots", required=True, help="scenario_slots.json from inquire_finetune.py pp2-draw")
    add_teacher_options(contexts)
    contexts.set_defaults(func=cmd_contexts)
    rows = sub.add_parser("rows", help="plan the rows; the teacher writes each row's conversation")
    rows.add_argument("--draw", required=True, help="scenario_draw.json (the roles: rows only for train/validation)")
    rows.add_argument("--seed", type=int, default=PLAN_SEED)
    add_teacher_options(rows)
    rows.set_defaults(func=cmd_rows)
    drafts = sub.add_parser("drafts", help="two educator lines per row for the exact training prompt")
    add_teacher_options(drafts)
    drafts.set_defaults(func=cmd_drafts)
    return ap


def main(argv=None, ask=None):
    """`ask` (messages -> text) replaces the teacher model; tests pass a fake."""
    args = build_parser().parse_args(argv)
    if args.base_url and args.backend != "local":
        _refuse("--base-url is only for --backend local", ["colab, aistudio and hf have fixed addresses"], args.command)
    ask = ask or make_chat(args.backend, args.model, TEACHER_TEMPERATURE, TEACHER_MAX_TOKENS, args.base_url)
    args.func(args, ask)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()

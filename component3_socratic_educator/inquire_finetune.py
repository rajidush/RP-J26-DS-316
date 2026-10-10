"""
inquire_finetune.py

Turns the drafted INQUIRE rows (draft_inquire_dataset.py) into a fine-tuning
pilot for Component 3. Plain JSON/JSONL in and out.

    review   step through the drafted rows in the terminal: choose draft A or B,
             rewrite, reject or skip; every decision is saved to the drafts file
             straight away, and a re-run resumes at the first pending row
    build    reviewed rows + INQUIRE seed examples (with recorded overrides and
             labels) -> train / validation / test JSONL grouped by scenario,
             plus a manifest; refuses pending rows and failing completions
    sheet    both arms' raw outputs on the test prompts -> shuffled blind CSV
             with opaque IDs, plus a separate key (ID -> arm)
    unblind  a fully scored sheet + key -> summary.json / summary.md per arm

format_check() is the shared format / stated-emotion contract (ADR 0003).
Nothing here touches generate(), the FSM controller or the controller grammar.

Usage:
    python inquire_finetune.py review [--drafts <file>] [--split all|train_or_val|test]
    python inquire_finetune.py build [--drafts <file>] [--seeds <file>] [--scenarios <file>]
                                     [--overrides <file>] [--labels <file>] [--out <dir>]
    python inquire_finetune.py sheet --outputs <control.jsonl> <adapted.jsonl> [--test <file>] [--seed N] [--out <dir>]
    python inquire_finetune.py unblind --sheet <csv> --key <json> --outputs <control.jsonl> <adapted.jsonl>
                                       [--test <file>] [--out <dir>]
"""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from draft_inquire_dataset import REPLY_TYPES, RISK_CATEGORIES, load_jsonl, parse_inquire_prompt
from src.grammar_decoder import _FORBIDDEN_SUBSTRINGS

HERE = Path(__file__).resolve().parent
DRAFTS_DIR = HERE / "evidence" / "inquire_drafts_2026-10-08"
DRAFTS_FILE = DRAFTS_DIR / "inquire_drafts.jsonl"
SCENARIOS_FILE = DRAFTS_DIR / "scenarios.json"
SEED_FILE = HERE / "data" / "educator_seed.jsonl"
SEED_INPUTS_DIR = HERE / "evidence" / "inquire_seed_inputs"

# The controller grammar's hard backstop, shared rather than copied (ADR 0003).
FORBIDDEN_SUBSTRINGS = _FORBIDDEN_SUBSTRINGS

MAX_SENTENCES = 2
MAX_WORDS = 30

# Fixed list for the stated-emotion check: words that attribute a feeling. Words
# that are mostly used another way ("hurt someone", "don't worry", "weird")
# are left out, so a paraphrase can slip through (see README limitations).
EMOTION_WORDS = (
    "afraid", "angry", "anger", "annoyed", "anxious", "anxiety", "ashamed", "bored",
    "confused", "creeped out", "depressed", "disappointed", "disgusted", "distressed",
    "embarrassed", "excited", "fear", "frightened", "frightening", "frustrated",
    "frustrating", "frustration", "furious", "grossed out", "guilty", "happy",
    "heartbroken", "helpless", "hopeless", "horrified", "irritated", "jealous",
    "lonely", "mad", "miserable", "nervous", "overwhelmed", "panicked", "sad",
    "sadness", "scared", "scary", "shaken", "shocked", "stressed", "terrified",
    "uncomfortable", "uneasy", "unhappy", "upset", "upsetting", "worried",
)

_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")
_DOUBLE_QUOTES = ('"', "“", "”")
# A single quote opening a word is quoting; one inside a word is an apostrophe.
_SINGLE_QUOTE_OPEN = re.compile(r"(?:^|\s)['‘]")
_META = (
    re.compile(r"^\s*(educator|assistant|response|answer|reply|child|ai|model|output|system|user)\s*:", re.I),
    re.compile(r"\b(here is|here's) (a|an|the|my) (response|reply|answer|question)\b", re.I),
    re.compile(r"\bas an ai\b", re.I),
    re.compile(r"[\[\]*#]|\(note\b", re.I),
)


def _normalise(text):
    return text.replace("’", "'").replace("‘", "'").lower()


def _has_phrase(phrase, text):
    return re.search(rf"(?<![a-z']){re.escape(phrase)}(?![a-z'])", text) is not None


def unstated_emotions(text, child_reply):
    """Emotion words in `text` that the child's reply does not contain verbatim."""
    said, child = _normalise(text), _normalise(child_reply)
    return [w for w in EMOTION_WORDS if _has_phrase(w, said) and not _has_phrase(w, child)]


@dataclass(frozen=True)
class CheckResult:
    sentences: int
    questions: int
    question_last: bool
    words: int
    forbidden: tuple
    quoted: bool
    meta: bool
    unstated_emotions: list

    @property
    def rules(self):
        """Each format rule on its own: True = passed."""
        return {
            "sentences": 1 <= self.sentences <= MAX_SENTENCES,
            "question": self.questions == 1 and self.question_last,
            "words": self.words <= MAX_WORDS,
            "forbidden": not self.forbidden,
            "quotes": not self.quoted,
            "meta": not self.meta,
        }

    @property
    def format_ok(self):
        return all(self.rules.values())

    @property
    def emotion_ok(self):
        return not self.unstated_emotions

    @property
    def ok(self):
        return self.format_ok and self.emotion_ok

    def failures(self):
        out = [f"{name}" for name, passed in self.rules.items() if not passed]
        if self.forbidden:
            out[out.index("forbidden")] = f"forbidden: {', '.join(self.forbidden)}"
        if self.unstated_emotions:
            out.append(f"unstated emotion: {', '.join(self.unstated_emotions)}")
        return out


def format_check(text, child_reply):
    """ADR 0003 format check plus the stated-emotion check, each rule reported separately."""
    text = text.strip()
    lowered = text.lower()
    return CheckResult(
        sentences=len([s for s in _SENTENCE_END.split(text) if re.search(r"\w", s)]),
        questions=text.count("?"),
        question_last=text.endswith("?"),
        words=len([t for t in text.split() if re.search(r"\w", t)]),
        forbidden=tuple(bad for bad in FORBIDDEN_SUBSTRINGS if bad in lowered),
        quoted=any(q in text for q in _DOUBLE_QUOTES) or bool(_SINGLE_QUOTE_OPEN.search(text)),
        meta=any(p.search(text) for p in _META),
        unstated_emotions=unstated_emotions(text, child_reply),
    )


# ---------------------------------------------------------------- io
def dump_jsonl(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def write_atomic(path, text):
    """Write a temporary file next to `path`, then replace it, so a crash never leaves half a file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def child_reply(row):
    return parse_inquire_prompt(row["prompt"][0]["content"])["child"]


def completion_text(row):
    return row["completion"][0]["content"]


# ---------------------------------------------------------------- review
TRUSTED_ADULT = re.compile(
    r"\b(trusted (?:adult|grown-up)s?|parents?|teachers?|grown-ups?|adults?|guardians?|"
    r"mum|mom|dad|caregivers?|counsell?ors?|family)\b", re.I)

_TRAIN_MENU = "[a] approve A  [b] approve B  [r] rewrite  [x] reject  [n] note  [s] skip  [q] quit"
_TEST_MENU = "[k] keep prompt  [x] reject prompt  [n] note  [s] skip  [q] quit"


def draft_order(example_id):
    """Draft indices shown as A and B: shuffled per row (seeded by example_id) so A is not always one drafter."""
    return [1, 0] if random.Random(example_id).random() < 0.5 else [0, 1]


def highlight_trusted_adults(text):
    return TRUSTED_ADULT.sub(lambda m: f"[[{m.group(0)}]]", text)


def describe_checks(text, child):
    result = format_check(text, child)
    rule_failures = [name for name, passed in result.rules.items() if not passed]
    parts = ["format ok" if result.format_ok else "format FAIL: " + ", ".join(rule_failures),
             "emotion ok" if result.emotion_ok else "emotion FAIL: " + ", ".join(result.unstated_emotions)]
    if TRUSTED_ADULT.search(text):
        parts.append("mentions trusted adult")
    return " | ".join(parts)


def _ask(prompt, initial=""):
    """One line from stdin. In a real terminal with readline, `initial` is pre-filled for editing."""
    if initial and getattr(sys.stdin, "isatty", lambda: False)():
        try:
            import readline
        except ImportError:  # Windows: no readline, the owner retypes the line
            pass
        else:
            readline.set_startup_hook(lambda: readline.insert_text(initial))
            try:
                return input(prompt)
            finally:
                readline.set_startup_hook()
    print(prompt, end="", flush=True)
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\r\n")


def review_progress(rows):
    lines = []
    for split in ("train_or_val", "test"):
        status = Counter(r["review"]["status"] for r in rows if r["split"] == split)
        total = sum(status.values())
        if not total:
            continue
        pending = status["pending"]
        if split == "train_or_val":
            detail = (f"{status['approved']} approved as drafted, {status['rewritten']} rewritten, "
                      f"{status['rejected']} rejected")
        else:
            detail = f"{status['approved']} kept, {status['rejected']} rejected"
        lines.append(f"{split}: {total - pending} reviewed, {pending} pending ({detail})")
    return "\n".join(lines)


def _show_row(row, position, total):
    parts = parse_inquire_prompt(row["prompt"][0]["content"])
    print(f"\n[{row['example_id']}] {position}/{total}  {row['split']}  reply type: {row['reply_type']}"
          f"  risk: {row['risk_category']}")
    print(f"Context: {parts['context']}")
    print(f"Previous educator message: {parts['previous']}")
    print(f"Child's latest reply: {parts['child']}")
    if row["split"] == "test":
        return
    for letter, idx in zip("AB", draft_order(row["example_id"])):
        text = row["drafts"][idx]
        print(f"  {letter}: {highlight_trusted_adults(text)}")
        print(f"     {describe_checks(text, parts['child'])}")


def _decide(row, status, chosen=None, completion=None, rewritten=False, notes=""):
    row["completion"] = completion
    row["review"] = {"status": status, "chosen_draft": chosen, "rewritten": rewritten, "notes": notes,
                     "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _join_notes(*notes):
    return "; ".join(n for n in notes if n)


def _rewrite(row, idx, child, note):
    print(f"editing: {row['drafts'][idx]}")
    while True:
        text = _ask("new text (empty cancels)> ", initial=row["drafts"][idx]).strip()
        if not text:
            print("rewrite cancelled")
            return False
        problems = format_check(text, child).failures()
        if not problems:
            _decide(row, "rewritten", chosen=idx, completion=text, rewritten=True, notes=note)
            return True
        print(f"refused: {', '.join(problems)}")


def _review_row(row):
    """Ask until a decision is made. Returns 'decided', 'skip' or 'quit'."""
    child = child_reply(row)
    is_test = row["split"] == "test"
    order = None if is_test else draft_order(row["example_id"])
    note = ""
    while True:
        print(_TEST_MENU if is_test else _TRAIN_MENU)
        key = _ask("> ").strip().lower()
        if key == "q":
            return "quit"
        if key == "s":
            return "skip"
        if key == "n":
            note = _join_notes(note, _ask("note> ").strip())
        elif key == "x":
            _decide(row, "rejected", notes=_join_notes(note, _ask("note (optional)> ").strip()))
            return "decided"
        elif is_test and key == "k":
            _decide(row, "approved", notes=note)
            return "decided"
        elif not is_test and key in ("a", "b"):
            idx = order["ab".index(key)]
            problems = format_check(row["drafts"][idx], child).failures()
            if problems:
                print(f"draft {key.upper()} fails the checks ({', '.join(problems)}); rewrite it instead")
                continue
            _decide(row, "approved", chosen=idx, completion=row["drafts"][idx], notes=note)
            return "decided"
        elif not is_test and key == "r":
            start = _ask("start from [a/b]> ").strip().lower()
            if start not in ("a", "b"):
                print("rewrite cancelled")
            elif _rewrite(row, order["ab".index(start)], child, note):
                return "decided"
        else:
            print(f"unknown key {key!r}")


def cmd_review(args):
    path = Path(args.drafts)
    rows = load_jsonl(path)
    splits = ("train_or_val", "test") if args.split == "all" else (args.split,)
    queue = [r for split in splits for r in rows if r["split"] == split and r["review"]["status"] == "pending"]
    print(review_progress(rows))
    try:
        for n, row in enumerate(queue, 1):
            _show_row(row, n, len(queue))
            outcome = _review_row(row)
            if outcome == "quit":
                break
            if outcome == "decided":
                write_atomic(path, dump_jsonl(rows))
    except EOFError:
        pass
    print(f"\nreview file (saved after every decision): {path}")
    print(review_progress(rows))


# ---------------------------------------------------------------- build
# Seed reply types add "explanation" (the child describes what happened or why); drafted rows use the five.
SEED_REPLY_TYPES = REPLY_TYPES + ("explanation",)
# Validation mirrors test (decision 4, ticket 03): only new drafted scenarios, one per risk
# category, never a category's only scenario; 5 x 5 rows is about 20% of training rows.
VALIDATION_SCENARIOS = 5
VALIDATION_SEED = 20261008
# Seen while iterating prompts or probing the inference baseline: never test evidence.
DEVELOPMENT_EXAMPLES = ("INQ001", "INQ026", "INQ027", "INQ028")
SPLITS = ("train", "validation", "test")


def _refuse(problem, details, command="build"):
    sys.exit(f"{command} refused: {problem}\n" + "\n".join(f"  {d}" for d in details))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _example(row, completion, source):
    """A conversational prompt/completion record (completion None for test: prompt only)."""
    out = {"example_id": row["example_id"], "scenario_id": row["scenario_id"], "reply_type": row["reply_type"],
           "risk_category": row["risk_category"], "source": source,
           "prompt": [{"role": "user", "content": row["prompt"][0]["content"]}]}
    if completion is not None:
        out["completion"] = [{"role": "assistant", "content": completion}]
    return out


def _seed_examples(seed_rows, overrides, labels):
    """INQUIRE seed rows with their labels and overrides applied; the seed rows themselves are not modified."""
    inquire = [r for r in seed_rows if r["example_id"].startswith("INQ")]
    by_id = {r["example_id"]: r for r in inquire}
    reply_types, risk = labels.get("reply_types", {}), labels.get("risk_categories", {})
    problems = [f"{r['example_id']}: no reply type label" for r in inquire if r["example_id"] not in reply_types]
    problems += [f"{s}: no risk category label" for s in sorted({r["scenario_id"] for r in inquire}) if s not in risk]
    problems += [f"{k}: unknown reply type label {v!r}" for k, v in reply_types.items() if v not in SEED_REPLY_TYPES]
    problems += [f"{k}: unknown risk category label {v!r}" for k, v in risk.items() if v not in RISK_CATEGORIES]
    if problems:
        _refuse("seed labels are missing or invalid", problems)

    completions = {eid: completion_text(r) for eid, r in by_id.items()}
    for o in overrides:
        eid = o["example_id"]
        if eid not in by_id:
            problems.append(f"{eid}: override for an example that is not an INQUIRE seed")
        elif o["original"] != completions[eid]:
            problems.append(f"{eid}: stale override, its original no longer matches the seed completion")
        else:
            completions[eid] = o["completion"]
    if problems:
        _refuse("seed overrides do not apply", problems)

    return [_example({**r, "reply_type": reply_types[r["example_id"]], "risk_category": risk[r["scenario_id"]]},
                     completions[r["example_id"]], "seed") for r in inquire]


def _check_problems(examples):
    problems = []
    for ex in examples:
        if "completion" not in ex:
            problems.append(f"{ex['example_id']}: no completion")
            continue
        failures = format_check(completion_text(ex), child_reply(ex)).failures()
        if failures:
            problems.append(f"{ex['example_id']}: {', '.join(failures)}")
    return problems


def _counts(rows):
    def tally(key):
        return dict(sorted(Counter(r[key] for r in rows).items()))
    return {"rows": len(rows), "scenarios": len({r["scenario_id"] for r in rows}), "reply_type": tally("reply_type"),
            "risk_category": tally("risk_category"), "source": tally("source")}


def _check_counts(rows):
    results = [format_check(completion_text(r), child_reply(r)) for r in rows]
    passed = sum(r.ok for r in results)
    return {"rows": len(rows), "format_ok": sum(r.format_ok for r in results),
            "emotion_ok": sum(r.emotion_ok for r in results),
            "pass_rate": round(passed / len(rows), 3) if rows else None}


def _trusted_adult_share(rows):
    if not rows or "completion" not in rows[0]:
        return None
    return round(sum(bool(TRUSTED_ADULT.search(completion_text(r))) for r in rows) / len(rows), 3)


def draw_validation(new_scenarios, k=VALIDATION_SCENARIOS, seed=VALIDATION_SEED):
    """Stratified draw: k categories with at least 2 new scenarios, then one scenario from each."""
    by_category = {}
    for s in sorted(new_scenarios, key=lambda s: s["scenario_id"]):
        by_category.setdefault(s["risk_category"], []).append(s["scenario_id"])
    rng = random.Random(seed)
    eligible = sorted(c for c, ids in by_category.items() if len(ids) >= 2)
    return sorted(rng.choice(by_category[c]) for c in sorted(rng.sample(eligible, k)))


def cmd_build(args):
    inputs = {"drafts": args.drafts, "seeds": args.seeds, "scenarios": args.scenarios,
              "overrides": args.overrides, "labels": args.labels}
    drafts, seed_rows = load_jsonl(args.drafts), load_jsonl(args.seeds)
    scenarios, overrides, labels = _load_json(args.scenarios), _load_json(args.overrides), _load_json(args.labels)
    test_ids = list(scenarios["test_scenario_ids"])

    pending = [r["example_id"] for r in drafts if r["review"]["status"] == "pending"]
    if pending:
        _refuse(f"{len(pending)} drafted rows are still pending review", [", ".join(pending)])
    misplaced = [f"{r['example_id']}: split {r['split']!r} but scenario {r['scenario_id']}"
                 f"{' is' if r['scenario_id'] in test_ids else ' is not'} a frozen test scenario"
                 for r in drafts if (r["split"] == "test") != (r["scenario_id"] in test_ids)]
    if misplaced:
        _refuse("drafted rows disagree with the frozen test scenarios", misplaced)

    seeds = _seed_examples(seed_rows, overrides["overrides"], labels)
    seed_scenarios = sorted({r["scenario_id"] for r in seeds})
    leaked = sorted(set(seed_scenarios) & set(test_ids))
    if leaked:
        _refuse("seed scenarios are in the frozen test set", leaked)

    kept = [r for r in drafts if r["review"]["status"] != "rejected"]
    rejected = sorted(r["example_id"] for r in drafts if r["review"]["status"] == "rejected")
    reviewed = [_example(r, r["completion"], "reviewed") for r in kept if r["split"] == "train_or_val"]
    test = [_example(r, None, "reviewed") for r in kept if r["split"] == "test"]

    problems = _check_problems(seeds + reviewed)
    if problems:
        _refuse("training/validation completions fail the format or stated-emotion check", problems)

    # Drawn over the scenarios file, not the kept rows, so a rejection never moves another
    # scenario between train and validation. Seed scenarios always train.
    new_scenarios = [s for s in scenarios["scenarios"] if s["scenario_id"] not in test_ids]
    val_ids = draw_validation(new_scenarios)
    train_ids = sorted(set(seed_scenarios) | {s["scenario_id"] for s in new_scenarios} - set(val_ids))
    empty = [s for s in val_ids if not any(r["scenario_id"] == s for r in reviewed)]
    if empty:
        _refuse("validation scenarios have no kept rows (all rejected)", empty)

    def ordered(rows):
        return sorted(rows, key=lambda r: r["example_id"])
    splits = {"train": ordered(r for r in seeds + reviewed if r["scenario_id"] not in val_ids),
              "validation": ordered(r for r in seeds + reviewed if r["scenario_id"] in val_ids),
              "test": ordered(test)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_atomic(out / f"{name}.jsonl", dump_jsonl(rows))

    covered = {s["risk_category"] for s in scenarios["scenarios"] if s["scenario_id"] in test_ids}
    manifest = {
        "dataset": "inquire-finetune",
        "inputs": {k: {"file": Path(p).name, "sha256": _sha256(p)} for k, p in inputs.items()},
        "outputs": {f"{name}.jsonl": _sha256(out / f"{name}.jsonl") for name in SPLITS},
        "split_seeds": {"test": scenarios["test_split_seed"], "validation": VALIDATION_SEED},
        "validation_rule": {
            "rule": "new drafted scenarios only; seeded draw of categories with >= 2 new scenarios, "
                    "one scenario per category; seed scenarios always train",
            "scenarios": VALIDATION_SCENARIOS,
            "row_share": round(len(splits["validation"]) / (len(splits["train"]) + len(splits["validation"])), 3),
        },
        "scenarios": {"train": train_ids, "validation": val_ids, "test": test_ids},
        "development_examples": {eid: next(n for n, rows in splits.items()
                                           if any(r["example_id"] == eid for r in rows))
                                 for eid in DEVELOPMENT_EXAMPLES},
        "counts": {name: _counts(rows) for name, rows in splits.items()},
        "rejected": {"count": len(rejected), "example_ids": rejected},
        "excluded_seed_rows": dict(sorted(Counter(r["example_id"][:3] for r in seed_rows
                                                  if not r["example_id"].startswith("INQ")).items())),
        "checks": {name: _check_counts(splits[name]) for name in ("train", "validation")},
        "trusted_adult_share": {name: _trusted_adult_share(rows) for name, rows in splits.items()},
        "seed_overrides": [{k: o[k] for k in ("example_id", "reason", "proposed_by")} for o in overrides["overrides"]],
        "seed_labels": {"labelled_by": labels.get("labelled_by", "not recorded")},
        "known_gaps": {"test_missing_risk_categories": [c for c in RISK_CATEGORIES if c not in covered]},
    }
    write_atomic(out / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"wrote {out}")
    for name in SPLITS:
        c = manifest["counts"][name]
        print(f"  {name}.jsonl: {c['rows']} rows, {c['scenarios']} scenarios")
    print(f"  rejected: {len(rejected)}  excluded non-INQUIRE seed rows: {sum(manifest['excluded_seed_rows'].values())}")
    print(f"  trusted-adult share: {manifest['trusted_adult_share']}")
    print(f"  test set has no scenario for: {', '.join(manifest['known_gaps']['test_missing_risk_categories'])}")


# ---------------------------------------------------------------- sheet / unblind
# Raw-output record (one JSONL file per arm, written by the ticket 06 notebook cells):
#   example_id, arm ("control" | "adapted"), model_id, adapter (None for control),
#   generation_settings (dict, identical for both arms), response, latency_seconds
ARMS = ("control", "adapted")
# Behavior rubric, one 0/1 column each (README: "Blind comparison").
RUBRIC = ("handles_reply_type", "no_shame_invention_or_flagged", "natural_for_age")
SHEET_COLUMNS = ("response_id", "reply_type", "context", "previous_message", "child_reply", "response",
                 *RUBRIC, "notes")
SHEET_SEED = 20261012
NOT_MEASURED = "not measured — adapted model not wired into the FSM"


def _load_raw_outputs(paths, test, command):
    """{arm: {example_id: record}}; refuses unless both arms answer every test prompt exactly once, alike."""
    by_arm, problems = {}, []
    for path in paths:
        rows = load_jsonl(path)
        arms = {r.get("arm") for r in rows}
        if len(arms) != 1 or not arms <= set(ARMS):
            _refuse(f"{Path(path).name}: expected one arm out of {ARMS}, found {sorted(map(str, arms))}", [], command)
        arm = arms.pop()
        if arm in by_arm:
            _refuse(f"two output files for arm {arm!r}", [], command)
        by_arm[arm] = {}
        for r in rows:
            if r["example_id"] in by_arm[arm]:
                problems.append(f"{arm}: {r['example_id']} answered twice")
            by_arm[arm][r["example_id"]] = r
    if set(by_arm) != set(ARMS):
        _refuse(f"need one output file per arm {ARMS}, got {sorted(by_arm)}", [], command)
    expected = {r["example_id"] for r in test}
    for arm, records in by_arm.items():
        problems += [f"{arm}: no output for {eid}" for eid in sorted(expected - set(records))]
        problems += [f"{arm}: {eid} is not a test prompt" for eid in sorted(set(records) - expected)]
    for field, label in (("generation_settings", "generation settings"), ("model_id", "model ID")):
        values = {json.dumps(r[field], sort_keys=True) for records in by_arm.values() for r in records.values()}
        if len(values) > 1:
            problems.append(f"{label} differ between outputs (only the adapter may differ): {sorted(values)}")
    if problems:
        _refuse("raw outputs do not match the test split", problems, command)
    return by_arm


def _read_sheet(path):
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except UnicodeDecodeError:
        _refuse(f"{path} is not UTF-8", ['re-save it as "CSV UTF-8 (Comma delimited)", not plain "CSV"'], "unblind")


def cmd_sheet(args):
    test = load_jsonl(args.test)
    outputs = _load_raw_outputs(args.outputs, test, "sheet")
    rows = []
    for r in test:
        parts = parse_inquire_prompt(r["prompt"][0]["content"])
        for arm in ARMS:
            rows.append(({"arm": arm, "example_id": r["example_id"]},
                         {"reply_type": r["reply_type"], "context": parts["context"],
                          "previous_message": parts["previous"], "child_reply": parts["child"],
                          "response": outputs[arm][r["example_id"]]["response"]}))
    random.Random(args.seed).shuffle(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = {"seed": args.seed, "sheet": "scoring_sheet.csv", "rubric": list(RUBRIC), "responses": {}}
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SHEET_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for n, (hidden, shown) in enumerate(rows, 1):
        rid = f"R{n:03d}"
        key["responses"][rid] = hidden
        writer.writerow({"response_id": rid, **shown, **{c: "" for c in RUBRIC}, "notes": ""})
    # utf-8-sig so Excel shows curly quotes and apostrophes correctly
    write_atomic(out / "scoring_sheet.csv", "﻿" + buffer.getvalue())
    write_atomic(out / "key.json", json.dumps(key, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out / 'scoring_sheet.csv'} ({len(rows)} responses, shuffle seed {args.seed})")
    print(f"wrote {out / 'key.json'}: do not open until every row is scored")


def _mean(values):
    return round(sum(values) / len(values), 3) if values else None


def closing_sentence(text):
    """The last sentence of a response (spec story 14 watch item: one closing question for every reply)."""
    return re.split(r"(?<=[.!?])\s+", text.strip())[-1]


def _arm_summary(records, scored, test_by_id):
    checks = [format_check(r["response"], child_reply(test_by_id[eid])) for eid, r in records.items()]
    failed = Counter(rule for c in checks for rule, ok in c.rules.items() if not ok)
    latencies = [r["latency_seconds"] for r in records.values() if r.get("latency_seconds") is not None]
    by_type = {}
    for row in scored:
        by_type.setdefault(row["reply_type"], []).append(row)

    def rubric(rows):
        means = {c: _mean([int(r[c]) for r in rows]) for c in RUBRIC}
        return {**means, "overall": _mean([int(r[c]) for r in rows for c in RUBRIC]), "n": len(rows)}

    overall = rubric(scored)
    first = next(iter(records.values()))
    responses = [r["response"] for r in records.values()]
    adults = sum(bool(TRUSTED_ADULT.search(text)) for text in responses)
    closing, repeats = Counter(closing_sentence(text) for text in responses).most_common(1)[0]
    return {
        "model_id": first["model_id"],
        "adapter": first.get("adapter"),
        "responses": len(records),
        "format_check": {"pass_rate": _mean([c.format_ok for c in checks]), "failed_rules": dict(sorted(failed.items()))},
        "stated_emotion_violations": sum(not c.emotion_ok for c in checks),
        "mean_words": _mean([c.words for c in checks]),
        "trusted_adult": {"mentions": adults, "share": round(adults / len(responses), 3)},
        "most_repeated_closing": {"text": closing, "count": repeats},
        "rubric": {"overall": overall["overall"],
                   "per_criterion": {c: overall[c] for c in RUBRIC},
                   "per_reply_type": {t: rubric(rows) for t, rows in sorted(by_type.items())}},
        "latency_seconds": {"mean": _mean(latencies), "median": round(statistics.median(latencies), 3) if latencies else None,
                            "max": max(latencies, default=None),
                            "note": "reported separately; not a behavior measure"},
        "not_measured": {"json_validity": NOT_MEASURED, "controller_grammar_compliance": NOT_MEASURED,
                         "controller_correctness": NOT_MEASURED},
    }


def _pct(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _summary_markdown(summary):
    s = summary["sample"]
    lines = ["# INQUIRE comparison: blind scoring summary", "",
             f"Sample: {s['scenarios']} scenarios, {s['prompts']} prompts per arm. {s['note']}", ""]
    for arm, a in summary["arms"].items():
        lat = a["latency_seconds"]
        lines += [f"## {arm}", "", f"Model: `{a['model_id']}`" + (f" + adapter `{a['adapter']}`" if a["adapter"] else ""), "",
                  "| Measure | Value |", "|---|---|",
                  f"| Format-check pass rate | {_pct(a['format_check']['pass_rate'])} |",
                  f"| Stated-emotion violations | {a['stated_emotion_violations']} / {a['responses']} |",
                  f"| Rubric overall (behavior) | {_pct(a['rubric']['overall'])} |",
                  *[f"| Rubric: {c} | {_pct(v)} |" for c, v in a["rubric"]["per_criterion"].items()],
                  f"| Mean words | {a['mean_words']} |",
                  f"| Trusted-adult mentions | {a['trusted_adult']['mentions']} / {a['responses']} "
                  f"({_pct(a['trusted_adult']['share'])}) |",
                  f"| Most repeated closing sentence | {a['most_repeated_closing']['count']}× "
                  f"\"{a['most_repeated_closing']['text']}\" |",
                  f"| Latency mean / median / max (s, separate) | {lat['mean']} / {lat['median']} / {lat['max']} |",
                  *[f"| {k.replace('_', ' ').capitalize()} | {v} |" for k, v in a["not_measured"].items()],
                  "", "| Reply type | n | " + " | ".join(RUBRIC) + " | overall |",
                  "|---|---|" + "---|" * (len(RUBRIC) + 1),
                  *[f"| {t} | {r['n']} | " + " | ".join(_pct(r[c]) for c in RUBRIC) + f" | {_pct(r['overall'])} |"
                    for t, r in a["rubric"]["per_reply_type"].items()], ""]
    return "\n".join(lines)


def cmd_unblind(args):
    sheet = _read_sheet(args.sheet)
    key = _load_json(args.key)
    unscored = [f"{row['response_id']}: {c}={row.get(c, '')!r}" for row in sheet for c in RUBRIC
                if (row.get(c) or "").strip() not in ("0", "1")]
    if unscored:
        _refuse(f"{len(unscored)} rubric cells are not scored 0 or 1", unscored, "unblind")
    counts = Counter(row["response_id"] for row in sheet)
    if any(n > 1 for n in counts.values()) or set(counts) != set(key["responses"]):
        _refuse("the sheet's response IDs do not match the key", [
            f"duplicated: {sorted(i for i, n in counts.items() if n > 1)}",
            f"missing from sheet: {sorted(set(key['responses']) - set(counts))}",
            f"not in key: {sorted(set(counts) - set(key['responses']))}"], "unblind")

    test = load_jsonl(args.test)
    test_by_id = {r["example_id"]: r for r in test}
    outputs = _load_raw_outputs(args.outputs, test, "unblind")
    scored, mismatched = {arm: [] for arm in ARMS}, []
    for row in sheet:
        hidden = key["responses"][row["response_id"]]
        if row["response"].strip() != outputs[hidden["arm"]][hidden["example_id"]]["response"].strip():
            mismatched.append(row["response_id"])
        # reply type comes from the test split via the key, never from the editable sheet
        scored[hidden["arm"]].append({c: row[c].strip() for c in RUBRIC}
                                     | {"reply_type": test_by_id[hidden["example_id"]]["reply_type"]})
    if mismatched:
        _refuse(f"{len(mismatched)} sheet responses do not match the raw outputs given (wrong files?)",
                [", ".join(mismatched)], "unblind")

    summary = {
        "sample": {"scenarios": len({r["scenario_id"] for r in test}), "prompts": len(test),
                   "note": "This is a pilot on held-out scenarios, not a significance test."},
        "shuffle_seed": key["seed"],
        "rubric": list(RUBRIC),
        "arms": {arm: _arm_summary(outputs[arm], scored[arm], test_by_id) for arm in ARMS},
    }
    out = Path(args.out) if args.out else Path(args.sheet).parent
    out.mkdir(parents=True, exist_ok=True)
    write_atomic(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_atomic(out / "summary.md", _summary_markdown(summary))
    print(f"wrote {out / 'summary.json'} and summary.md")
    for arm, a in summary["arms"].items():
        print(f"  {arm:<8} format {_pct(a['format_check']['pass_rate'])}  emotion violations "
              f"{a['stated_emotion_violations']}  rubric {_pct(a['rubric']['overall'])}  "
              f"mean words {a['mean_words']}  latency {a['latency_seconds']['mean']} s")


# ---------------------------------------------------------------- cli
def build_parser():
    ap = argparse.ArgumentParser(description="INQUIRE fine-tuning pilot: review, build, sheet, unblind.")
    sub = ap.add_subparsers(dest="command", required=True)

    review = sub.add_parser("review", help="review drafted INQUIRE rows in the terminal")
    review.add_argument("--drafts", default=str(DRAFTS_FILE))
    review.add_argument("--split", choices=["all", "train_or_val", "test"], default="all",
                        help="default: training/validation rows first, then the prompt check on test rows")
    review.set_defaults(func=cmd_review)

    build = sub.add_parser("build", help="build grouped train/validation/test JSONL and a manifest")
    build.add_argument("--drafts", default=str(DRAFTS_FILE))
    build.add_argument("--seeds", default=str(SEED_FILE))
    build.add_argument("--scenarios", default=str(SCENARIOS_FILE), help="holds the frozen test scenario IDs")
    build.add_argument("--overrides", default=str(SEED_INPUTS_DIR / "seed_overrides.json"))
    build.add_argument("--labels", default=str(SEED_INPUTS_DIR / "seed_labels.json"))
    build.add_argument("--out", default=str(HERE / "evidence" / f"inquire_dataset_{date.today().isoformat()}"))
    build.set_defaults(func=cmd_build)

    sheet = sub.add_parser("sheet", help="shuffled blind scoring sheet + hidden key from both arms' raw outputs")
    sheet.add_argument("--test", default=str(HERE / "evidence" / "inquire_dataset_2026-10-10" / "test.jsonl"))
    sheet.add_argument("--outputs", nargs=2, required=True, metavar="RAW_OUTPUTS_JSONL",
                       help="one raw-output file per arm (control and adapted), in any order")
    sheet.add_argument("--seed", type=int, default=SHEET_SEED)
    sheet.add_argument("--out", default=str(HERE / "evidence" / f"inquire_comparison_{date.today().isoformat()}"))
    sheet.set_defaults(func=cmd_sheet)

    unblind = sub.add_parser("unblind", help="join a fully scored sheet with the key; write summary.json/.md")
    unblind.add_argument("--sheet", required=True)
    unblind.add_argument("--key", required=True)
    unblind.add_argument("--test", default=str(HERE / "evidence" / "inquire_dataset_2026-10-10" / "test.jsonl"))
    unblind.add_argument("--outputs", nargs=2, required=True, metavar="RAW_OUTPUTS_JSONL")
    unblind.add_argument("--out", help="default: the sheet's folder")
    unblind.set_defaults(func=cmd_unblind)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

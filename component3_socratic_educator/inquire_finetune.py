"""
inquire_finetune.py

Turns the drafted INQUIRE rows (draft_inquire_dataset.py) into a fine-tuning
pilot for Component 3, and makes and scores the blind rating sheets of the PP2
validation. Plain JSON/JSONL/CSV in and out.

    review   step through the drafted rows in the terminal: choose draft A or B,
             rewrite, reject or skip; every decision is saved to the drafts file
             straight away, and a re-run resumes at the first pending row
    build    reviewed rows + INQUIRE seed examples (with recorded overrides and
             labels) -> train / validation / test JSONL grouped by scenario,
             plus a manifest; refuses pending rows and failing completions
    sheet    both arms' raw outputs on the test prompts -> shuffled blind CSV
             with opaque IDs, plus a separate key (ID -> arm)
    unblind  a fully scored sheet + key -> summary.json / summary.md per arm
    label-sheets  simulated sessions -> blind reply-label sheets (classmate: all,
             counsellor: stratified 40% overlap; or --owner: the owner's
             train/dev copy, never evidence) plus a hidden key
    label-score   returned reply-label sheets (+ adjudication) -> agreement,
             final labels, summary; refuses invalid or mismatched sheets

format_check() is the shared format / stated-emotion contract (ADR 0003).
Nothing here touches generate(), the FSM controller or the controller grammar.

Usage:
    python inquire_finetune.py review [--drafts <file>] [--split all|train_or_val|test]
    python inquire_finetune.py build [--drafts <file>] [--seeds <file>] [--scenarios <file>]
                                     [--overrides <file>] [--labels <file>] [--out <dir>]
    python inquire_finetune.py sheet --outputs <control.jsonl> <adapted.jsonl> [--test <file>] [--seed N] [--out <dir>]
    python inquire_finetune.py unblind --sheet <csv> --key <json> --outputs <control.jsonl> <adapted.jsonl>
                                       [--test <file>] [--out <dir>]
    python inquire_finetune.py label-sheets --sessions <sessions.jsonl> [--owner] [--seed N] [--out <dir>]
    python inquire_finetune.py label-score --key <key.json> --rated NAME=<csv> [--rated NAME=<csv>]
                                           [--adjudication <csv>] [--out <dir>]
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
from src.codebook import (DISTRESS_SEVERITIES, DISTRESS_SEVERITY, EXPRESSED_EMOTION, EXPRESSED_EMOTIONS, STANCE,
                          STANCES)
from src.agreement import krippendorff_alpha_nominal, percent_agreement, prevalence
from src.grammar_decoder import _FORBIDDEN_SUBSTRINGS

HERE = Path(__file__).resolve().parent
DRAFTS_DIR = HERE / "evidence" / "inquire" / "inquire_drafts_2026-10-08"
DRAFTS_FILE = DRAFTS_DIR / "inquire_drafts.jsonl"
SCENARIOS_FILE = DRAFTS_DIR / "scenarios.json"
SEED_FILE = HERE / "data" / "educator_seed.jsonl"
SEED_INPUTS_DIR = HERE / "evidence" / "inquire" / "inquire_seed_inputs"

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


def _read_sheet(path, command="unblind"):
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except UnicodeDecodeError:
        _refuse(f"{path} is not UTF-8", ['re-save it as "CSV UTF-8 (Comma delimited)", not plain "CSV"'], command)


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


# ---------------------------------------------------------------- reply-label sheets
# Session record (one JSONL line per simulated session, written by the ticket 06 notebook):
#   session_id, pool ("classifier_test" | "train_dev"), age_band, generator, risk_category,
#   turns (asdict(Turn): state, system_text, child_response, ...); other fields are ignored
POOLS = {"raters": "classifier_test", "owner": "train_dev"}
LABEL_PARTS = (EXPRESSED_EMOTION, DISTRESS_SEVERITY, STANCE)
LABEL_VALUES = {EXPRESSED_EMOTION: EXPRESSED_EMOTIONS, DISTRESS_SEVERITY: DISTRESS_SEVERITIES, STANCE: STANCES}
SKIP = "skip"   # recorded as missing, never as a judgement
LABEL_SHOWN = ("sensitive", "dialogue_so_far", "child_reply")
LABEL_COLUMNS = ("item", *LABEL_SHOWN, *LABEL_PARTS, "notes")
LABEL_SHEET_SEED = 20261030
TURN_FIELDS = ("state", "system_text", "child_response")
RATER_PAIR = ("counsellor", "classmate")   # the two independent raters whose overlap gives agreement
AGREEMENT_METHOD = ("on the items both raters were given, before adjudication; skips are missing and never "
                    "compared; Krippendorff's alpha for nominal labels (None when every rating has one value)")
# The counsellor labels a stratified share of the test sessions; the classmate labels all of them (stories 45-46).
COUNSELLOR_SHARE = 0.4
OVERLAP_STRATA = ("age_band", "generator")
# Marked on the sheet so a rater can skip them (story 42): self-harm and sexual-content triggers.
SENSITIVE_CATEGORIES = ("explicit_visual", "grooming_language", "self_harm_imagery", "self_harm_language")
# The rater's one-line definitions (CONTEXT.md; evidence/research/emotion_taxonomy_research_2026-10-25.md).
# Worked examples come in the window 0 calibration round (ticket 17), not here.
LABEL_GUIDE = {
    EXPRESSED_EMOTION: ("The feeling the reply conveys, named or not (GoEmotions' Ekman grouping).", {
        "anger": "anger, annoyance, disapproval",
        "disgust": "disgust, being grossed out",
        "fear": "fear, nervousness, worry",
        "joy": "joy, amusement, relief, excitement, gratitude, pride, optimism",
        "sadness": "sadness, disappointment, embarrassment, grief, remorse",
        "surprise": "surprise, confusion, realisation, curiosity",
        "neutral": "no feeling conveyed"}),
    DISTRESS_SEVERITY: ("How much distress the reply shows (after the ReachOut triage scheme).", {
        "none": "no distress",
        "past": "distress the child says is over (\"it scared me then, I'm okay now\")",
        "current_mild": "distress now, softened or manageable (\"kinda scary\", \"a bit worried\")",
        "current_acute": "strong distress now, being overwhelmed, or any self-harm language"}),
    STANCE: ("How the reply relates to taking a safer step (Motivational Interviewing client language).", {
        "change_talk": "the child moves toward a safer step: wants to, can, or will take one",
        "neutral": "answers or describes without leaning toward or away from a safer step",
        "sustain_talk": "argues for the risky behaviour or plays it down (\"it's just a joke\")",
        "discord": "friction with the educator (\"stop asking me\", \"it's not my fault\" said at the educator)",
        "disengaged": "non-answers and opting out (\"idk\", \"nothing\", \"leave me alone\"); a project code, not MI"}),
}


def _sheet_csv(columns, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return "﻿" + buffer.getvalue()   # utf-8-sig, as the comparison sheet


def _label_instructions():
    lines = ["# Reply-label sheet: instructions", "",
             "Each row is one child reply. Read `dialogue_so_far`, then label `child_reply` only, on three parts. "
             "Type one value from each list below. To leave a cell out, type `skip`: it is recorded as missing, "
             "never as a judgement. Rows marked `sensitive` = yes come from self-harm or sexual-content situations; "
             "skip them whenever you prefer. Use `notes` for anything else. Do not change any other column, "
             "and save the file as CSV UTF-8.", ""]
    for part in LABEL_PARTS:
        about, values = LABEL_GUIDE[part]
        lines += [f"## `{part}`", "", about, "", *[f"- `{v}`: {values[v]}" for v in LABEL_VALUES[part]], ""]
    return "\n".join(lines)


def _label_items(sessions):
    """One item per child reply, in session order: what the sheet shows and what the key hides."""
    items = []
    for s in sessions:
        dialogue, reply_index = [], 0
        for turn_index, turn in enumerate(s["turns"]):
            if turn["system_text"]:
                dialogue.append(f"Educator: {turn['system_text']}")
            if turn["child_response"] is None:
                continue
            shown = {"sensitive": "yes" if s["risk_category"] in SENSITIVE_CATEGORIES else "",
                     "dialogue_so_far": "\n".join(dialogue), "child_reply": turn["child_response"]}
            hidden = {"session_id": s["session_id"], "reply_index": reply_index, "turn_index": turn_index,
                      **{k: s[k] for k in ("age_band", "generator", "risk_category")}}
            items.append((hidden, shown))
            dialogue.append(f"Child: {turn['child_response']}")
            reply_index += 1
    return items


def _load_sessions(path, pool):
    sessions = load_jsonl(path)
    problems = [f"{s.get('session_id')}: no {field}" for s in sessions
                for field in ("session_id", "pool", "age_band", "generator", "risk_category", "turns") if field not in s]
    problems += [f"{s['session_id']}: turn {n} has no {field}" for s in sessions if "turns" in s
                 for n, turn in enumerate(s["turns"]) for field in TURN_FIELDS if field not in turn]
    if problems:
        _refuse("session records are incomplete", problems, "label-sheets")
    counts = Counter(s["session_id"] for s in sessions)
    problems = [f"{sid}: appears {n} times" for sid, n in sorted(counts.items()) if n > 1]
    problems += [f"{s['session_id']}: pool {s['pool']!r}" for s in sessions if s["pool"] != pool]
    if problems:
        _refuse(f"sessions must be unique and all from the {pool!r} pool", problems, "label-sheets")
    return sessions


def draw_overlap(sessions, rng, share=COUNSELLOR_SHARE):
    """{stratum: (sessions, drawn session IDs)}: the share of each band x generator stratum, rounded half up, at least one."""
    strata = {}
    for s in sorted(sessions, key=lambda s: s["session_id"]):
        strata.setdefault(" / ".join(s[k] for k in OVERLAP_STRATA), []).append(s["session_id"])
    return {name: (len(ids), sorted(rng.sample(ids, max(1, int(len(ids) * share + 0.5)))))
            for name, ids in sorted(strata.items())}


def cmd_label_sheets(args):
    copy_for = "owner" if args.owner else "raters"
    sessions = _load_sessions(args.sessions, POOLS[copy_for])
    # The owner's copy never shares the raters' folder (story 44).
    out = Path(args.out or HERE / "evidence" / "evaluate" /
               f"{'owner_' if args.owner else ''}reply_labels_{date.today().isoformat()}")
    if (out / "key.json").exists():
        _refuse(f"{out} already holds sheets", ["pass a new --out; recorded runs are never overwritten"], "label-sheets")

    rng = random.Random(args.seed)
    overlap = draw_overlap(sessions, rng) if copy_for == "raters" else {}
    items = _label_items(sessions)
    rng.shuffle(items)
    ids = [f"L{n:04d}" for n in range(1, len(items) + 1)]
    if copy_for == "owner":
        copies = {"owner": ids}
    else:
        drawn = {sid for _, chosen in overlap.values() for sid in chosen}
        counsellor = [i for i, (hidden, _) in zip(ids, items) if hidden["session_id"] in drawn]
        rng.shuffle(counsellor)
        copies = {"classmate": ids, "counsellor": counsellor}

    shown_by_id = {i: shown for i, (_, shown) in zip(ids, items)}
    key = {
        "sheet_kind": "reply_labels",
        "evidence": copy_for == "raters",
        "note": ("blind rater sheets: the classmate labels every item, the counsellor a stratified overlap"
                 if copy_for == "raters" else "owner labels for training and development: never evidence"),
        "seed": args.seed,
        "sessions_file": {"file": Path(args.sessions).name, "sha256": _sha256(args.sessions)},
        "pool": POOLS[copy_for],
        "overlap": {"share": COUNSELLOR_SHARE, "strata": list(OVERLAP_STRATA),
                    "per_stratum": {name: {"sessions": n, "counsellor": len(chosen)}
                                    for name, (n, chosen) in overlap.items()}} if overlap else None,
        "copies": {name: {"sheet": f"{name}_reply_labels.csv", "items": order} for name, order in copies.items()},
        "items": {i: hidden for i, (hidden, _) in zip(ids, items)},
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, order in copies.items():
        rows = [{"item": i, **shown_by_id[i], **{p: "" for p in LABEL_PARTS}, "notes": ""} for i in order]
        write_atomic(out / key["copies"][name]["sheet"], _sheet_csv(LABEL_COLUMNS, rows))
        key["copies"][name]["sha256"] = _sha256(out / key["copies"][name]["sheet"])
    write_atomic(out / "instructions.md", _label_instructions())
    write_atomic(out / "key.json", json.dumps(key, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out}: {len(items)} child replies from {len(sessions)} sessions, shuffle seed {args.seed}")
    for name, order in copies.items():
        print(f"  {key['copies'][name]['sheet']}: {len(order)} items")
    print("  key.json: do not open until every sheet is labelled")


def _label_value(raw):
    """A rater's cell as a codebook value: "Current mild" and "current-mild" are current_mild."""
    return re.sub(r"[\s-]+", "_", (raw or "").strip().lower())


def _same_text(a, b):
    """Shown text compared as Excel may save it: other line endings or trailing spaces are not an edit."""
    return " ".join((a or "").split()) == " ".join((b or "").split())


def _check_labelled(name, blank_rows, rows):
    """Everything wrong with one returned sheet against its blank copy, each problem naming its item."""
    blank = {r["item"]: r for r in blank_rows}
    counts = Counter((r.get("item") or "").strip() for r in rows)
    problems = [f"{name}: missing: {i}" for i in blank if i not in counts]
    problems += [f"{name}: duplicated: {i}" for i, n in counts.items() if n > 1 and i in blank]
    problems += [f"{name}: not on the blank sheet: {i}" for i in counts if i not in blank]
    for row in rows:
        item = (row.get("item") or "").strip()
        if item not in blank:
            continue
        problems += [f"{name}: {item}: {c} differs from the blank sheet" for c in LABEL_SHOWN
                     if not _same_text(row.get(c), blank[item][c])]
        problems += [f"{name}: {item}: {p} {row.get(p) or ''!r} is not one of {', '.join(LABEL_VALUES[p])} or {SKIP}"
                     for p in LABEL_PARTS if _label_value(row.get(p)) not in (*LABEL_VALUES[p], SKIP)]
    return problems


def _load_adjudication(path, key):
    """{(item, part): {label, reason}} from the adjudication CSV (item, part, label, reason); every change needs a reason."""
    out, problems = {}, []
    for line, row in enumerate(_read_sheet(path, "label-score"), 2):
        item, part, label = (row.get("item") or "").strip(), _label_value(row.get("part")), _label_value(row.get("label"))
        reason, where = (row.get("reason") or "").strip(), f"line {line} ({item} {part})"
        if item not in key["items"]:
            problems.append(f"{where}: no such item")
        elif part not in LABEL_VALUES:
            problems.append(f"{where}: part must be one of {', '.join(LABEL_PARTS)}")
        elif label not in LABEL_VALUES[part]:
            problems.append(f"{where}: label {row.get('label')!r} is not one of {', '.join(LABEL_VALUES[part])}")
        elif not reason:
            problems.append(f"{where}: no reason given")
        elif (item, part) in out:
            problems.append(f"{where}: adjudicated twice")
        else:
            out[item, part] = {"label": label, "reason": reason}
    if problems:
        _refuse(f"{Path(path).name}: adjudication rows are invalid", problems, "label-score")
    return out


def _round3(value):
    return None if value is None else round(value, 3)


def _pair_agreement(labels, overlap):
    """Per part, on the items both raters had: percent agreement, prevalence and nominal alpha."""
    parts = {}
    for part in LABEL_PARTS:
        pairs = [tuple(labels[name][i][part] for name in RATER_PAIR) for i in overlap]
        compared = sum(None not in pair for pair in pairs)
        parts[part] = {"compared": compared, "skipped": len(pairs) - compared,
                       "percent_agreement": _round3(percent_agreement(pairs)),
                       "prevalence": {v: _round3(share) for v, share in prevalence(pairs).items()},
                       "krippendorff_alpha": _round3(krippendorff_alpha_nominal(pairs))}
    return parts


def _final_label(part, ratings, adjudicated):
    """(label, source): the adjudicated label, else the raters' one value; a disagreement or a skip is no label."""
    if adjudicated:
        return adjudicated["label"], "adjudicated"
    given = [r[part] for r in ratings.values() if r[part] is not None]
    if not ratings:
        return None, "not_rated"
    if not given:
        return None, "skipped"
    if len(set(given)) > 1:
        return None, "disagreement"
    return given[0], "agreed" if len(given) > 1 else "one_rater"


def _label_summary_markdown(s):
    lines = ["# Reply labels: scoring summary", ""]
    if not s["evidence"]:
        lines += ["**Not evidence.** " + s["note"], ""]
    lines += [f"Shuffle seed {s['shuffle_seed']}. Sheets: " +
              ", ".join(f"{n} ({v['items']} items)" for n, v in s["sheets"].items()) + ".", ""]
    a = s["agreement"]
    if a:
        lines += [f"## Agreement: {' vs '.join(a['between'])} ({a['items']} shared items)", "", a["method"], "",
                  "| Part | Compared | Skipped | Percent agreement | Krippendorff's α | Prevalence |",
                  "|---|---|---|---|---|---|",
                  *[f"| {p} | {v['compared']} | {v['skipped']} | {_pct(v['percent_agreement'])} | "
                    f"{'n/a' if v['krippendorff_alpha'] is None else v['krippendorff_alpha']} | "
                    + ", ".join(f"{k} {_pct(x)}" for k, x in v["prevalence"].items()) + " |"
                    for p, v in a["parts"].items()], ""]
    else:
        lines += ["No agreement: fewer than two raters' sheets.", ""]
    adj = s["adjudication"]
    lines += ["## Labels" + (f" after adjudication ({adj['changes']} changes, {adj['file']})" if adj else ""), "",
              "| Part | Labelled | Disagreements left | Missing | Distribution |", "|---|---|---|---|---|",
              *[f"| {p} | {v['labelled']} | {len(v['disagreements'])} | {v['missing']} | "
                + ", ".join(f"{k} {n}" for k, n in v["distribution"].items()) + " |"
                for p, v in s["labels"].items()], ""]
    return "\n".join(lines)


def cmd_label_score(args):
    key_path = Path(args.key)
    key, run = _load_json(key_path), key_path.parent
    rated = {}
    for spec in args.rated:
        name, sep, path = spec.partition("=")
        if not sep or name not in key["copies"] or name in rated:
            _refuse(f"--rated {spec!r}: give each of the key's copies ({', '.join(key['copies'])}) at most once "
                    "as NAME=PATH", [], "label-score")
        rated[name] = Path(path)

    changed = [copy["sheet"] for copy in key["copies"].values() if _sha256(run / copy["sheet"]) != copy["sha256"]]
    if changed:
        _refuse("blank sheets have changed since they were written (filled in place?)",
                [*changed, "label a copy of the blank sheet, and keep the blank as written"], "label-score")
    blanks = {name: _read_sheet(run / copy["sheet"], "label-score") for name, copy in key["copies"].items()}
    problems, labels = [], {}
    for name, path in rated.items():
        rows = _read_sheet(path, "label-score")
        problems += _check_labelled(name, blanks[name], rows)
        cells = {(r.get("item") or "").strip(): {p: _label_value(r.get(p)) for p in LABEL_PARTS} for r in rows}
        labels[name] = {item: {p: None if v == SKIP else v for p, v in values.items()} for item, values in cells.items()}
    if problems:
        _refuse("rated sheets do not match their blank sheets", problems, "label-score")
    adjudication = _load_adjudication(args.adjudication, key) if args.adjudication else {}

    out = Path(args.out) if args.out else run / f"scores_{date.today().isoformat()}"
    if any((out / f).exists() for f in ("labels.jsonl", "summary.json", "summary.md")):
        _refuse(f"{out} already holds scores", ["pass a new --out; recorded scores are never overwritten"],
                "label-score")

    shown = {r["item"]: r for rows in blanks.values() for r in rows}
    records = []
    for item, hidden in sorted(key["items"].items(), key=lambda kv: (kv[1]["session_id"], kv[1]["reply_index"])):
        ratings = {name: by_item[item] for name, by_item in labels.items() if item in by_item}
        final = {p: _final_label(p, ratings, adjudication.get((item, p))) for p in LABEL_PARTS}
        records.append({"item": item, **hidden, "child_reply": shown[item]["child_reply"], "evidence": key["evidence"],
                        "labels": {p: label for p, (label, _) in final.items()},
                        "label_source": {p: source for p, (_, source) in final.items()},
                        "ratings": ratings,
                        "adjudication": {p: adjudication[item, p]["reason"] for p in LABEL_PARTS
                                         if (item, p) in adjudication}})

    overlap = sorted(set(labels[RATER_PAIR[0]]) & set(labels[RATER_PAIR[1]])) if set(RATER_PAIR) <= set(labels) else []
    summary = {
        "sheet_kind": key["sheet_kind"],
        "evidence": key["evidence"],
        "note": key["note"],
        "shuffle_seed": key["seed"],
        "key": {"file": key_path.name, "sha256": _sha256(key_path)},
        "sheets": {name: {"file": path.name, "sha256": _sha256(path), "items": len(labels[name]),
                          "skipped": {p: sum(r[p] is None for r in labels[name].values()) for p in LABEL_PARTS}}
                   for name, path in rated.items()},
        "agreement": {"between": list(RATER_PAIR), "items": len(overlap), "method": AGREEMENT_METHOD,
                      "parts": _pair_agreement(labels, overlap)} if overlap else None,
        "adjudication": {"file": Path(args.adjudication).name, "sha256": _sha256(args.adjudication),
                         "changes": len(adjudication)} if args.adjudication else None,
        "labels": {p: {"labelled": sum(r["labels"][p] is not None for r in records),
                       "disagreements": [r["item"] for r in records if r["label_source"][p] == "disagreement"],
                       "missing": sum(r["label_source"][p] in ("skipped", "not_rated") for r in records),
                       "distribution": dict(sorted(Counter(r["labels"][p] for r in records
                                                           if r["labels"][p] is not None).items()))}
                   for p in LABEL_PARTS},
    }
    out.mkdir(parents=True, exist_ok=True)
    write_atomic(out / "labels.jsonl", dump_jsonl(records))
    write_atomic(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_atomic(out / "summary.md", _label_summary_markdown(summary))
    print(f"wrote {out}: labels.jsonl, summary.json, summary.md" + ("" if key["evidence"] else "  (NOT evidence)"))
    if summary["agreement"]:
        for p, v in summary["agreement"]["parts"].items():
            print(f"  {p:<18} agreement {_pct(v['percent_agreement'])}  alpha {v['krippendorff_alpha']}  "
                  f"(compared {v['compared']}, skipped {v['skipped']})")
    for p, v in summary["labels"].items():
        print(f"  {p:<18} labelled {v['labelled']}  disagreements left {len(v['disagreements'])}  missing {v['missing']}")


# ---------------------------------------------------------------- cli
def build_parser():
    ap = argparse.ArgumentParser(description="INQUIRE fine-tuning pilot (review, build, sheet, unblind) and PP2 "
                                             "rating sheets (label-sheets, label-score).")
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
    build.add_argument("--out", default=str(HERE / "evidence" / "inquire" / f"inquire_dataset_{date.today().isoformat()}"))
    build.set_defaults(func=cmd_build)

    sheet = sub.add_parser("sheet", help="shuffled blind scoring sheet + hidden key from both arms' raw outputs")
    sheet.add_argument("--test", default=str(HERE / "evidence" / "inquire" / "inquire_dataset_2026-10-10" / "test.jsonl"))
    sheet.add_argument("--outputs", nargs=2, required=True, metavar="RAW_OUTPUTS_JSONL",
                       help="one raw-output file per arm (control and adapted), in any order")
    sheet.add_argument("--seed", type=int, default=SHEET_SEED)
    sheet.add_argument("--out", default=str(HERE / "evidence" / "inquire" / f"inquire_comparison_{date.today().isoformat()}"))
    sheet.set_defaults(func=cmd_sheet)

    unblind = sub.add_parser("unblind", help="join a fully scored sheet with the key; write summary.json/.md")
    unblind.add_argument("--sheet", required=True)
    unblind.add_argument("--key", required=True)
    unblind.add_argument("--test", default=str(HERE / "evidence" / "inquire" / "inquire_dataset_2026-10-10" / "test.jsonl"))
    unblind.add_argument("--outputs", nargs=2, required=True, metavar="RAW_OUTPUTS_JSONL")
    unblind.add_argument("--out", help="default: the sheet's folder")
    unblind.set_defaults(func=cmd_unblind)

    label_sheets = sub.add_parser("label-sheets", help="blind reply-label sheets + hidden key from simulated sessions")
    label_sheets.add_argument("--sessions", required=True, help="session records JSONL (ticket 06)")
    label_sheets.add_argument("--owner", action="store_true",
                              help="the owner's copy of train/dev sessions (never evidence), not the raters' copies")
    label_sheets.add_argument("--seed", type=int, default=LABEL_SHEET_SEED)
    label_sheets.add_argument("--out", help="default: evidence/evaluate/reply_labels_<date>, or "
                                            "owner_reply_labels_<date> with --owner")
    label_sheets.set_defaults(func=cmd_label_sheets)

    label_score = sub.add_parser("label-score", help="validate labelled reply-label sheets; agreement, adjudication, labels")
    label_score.add_argument("--key", required=True, help="the key.json written with the sheets")
    label_score.add_argument("--rated", action="append", required=True, metavar="NAME=PATH",
                             help="a returned sheet, e.g. classmate=classmate_filled.csv (repeat per rater)")
    label_score.add_argument("--adjudication", help="CSV of item, part, label, reason; the rated sheets stay unchanged")
    label_score.add_argument("--out", help="default: scores_<date> next to the key")
    label_score.set_defaults(func=cmd_label_score)
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

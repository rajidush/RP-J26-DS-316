"""
inquire_finetune.py

Turns the drafted INQUIRE rows (draft_inquire_dataset.py) into a fine-tuning
pilot for Component 3. Plain JSON/JSONL in and out.

    review   step through the drafted rows in the terminal: choose draft A or B,
             rewrite, reject or skip; every decision is saved to the drafts file
             straight away, and a re-run resumes at the first pending row

format_check() is the shared format / stated-emotion contract (ADR 0003).
Nothing here touches generate(), the FSM controller or the controller grammar.

Usage:
    python inquire_finetune.py review [--drafts <file>] [--split all|train_or_val|test]
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from draft_inquire_dataset import load_jsonl, parse_inquire_prompt
from src.grammar_decoder import _FORBIDDEN_SUBSTRINGS

HERE = Path(__file__).resolve().parent
DRAFTS_FILE = HERE / "evidence" / "inquire_drafts_2026-10-08" / "inquire_drafts.jsonl"

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


# ---------------------------------------------------------------- cli
def build_parser():
    ap = argparse.ArgumentParser(description="INQUIRE fine-tuning pilot: review drafted rows.")
    sub = ap.add_subparsers(dest="command", required=True)

    review = sub.add_parser("review", help="review drafted INQUIRE rows in the terminal")
    review.add_argument("--drafts", default=str(DRAFTS_FILE))
    review.add_argument("--split", choices=["all", "train_or_val", "test"], default="all",
                        help="default: training/validation rows first, then the prompt check on test rows")
    review.set_defaults(func=cmd_review)
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

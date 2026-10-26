"""
Shadow view of the fine-tuned INQUIRE adapter, for the viva demo.

The adapter is a CANDIDATE, not part of the dialogue: INQUIRE's grammar is still
exactly INQUIRE_TEMPLATES (grammar_decoder.py), and nothing here imports or calls the
FSM controller or the generator. The dashboard shows what the base and adapted
models would say next to the template line the child actually received.

Everything is pure (no Streamlit, no network) except `live_lines`, which takes the
model call as an argument so tests can pass a fake.

Evidence read from disk (never rewritten):
    evidence/inquire/inquire_dataset_2026-10-10/test.jsonl         held-out prompts (never trained on)
    evidence/inquire/inquire_comparison_2026-10-10/raw_outputs_*   "as evaluated": Colab, fp32, greedy
    evidence/inquire/inquire_served_<date>/raw_outputs_*           "as served": the GGUFs in LM Studio (optional)
"""
import json
import re
from collections import Counter
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "inquire"
TEST_FILE = EVIDENCE / "inquire_dataset_2026-10-10" / "test.jsonl"
EVALUATED_DIR = EVIDENCE / "inquire_comparison_2026-10-10"

# Same word list as TRUSTED_ADULT in inquire_finetune.py (that module needs the
# component folder on sys.path, so the dashboard keeps its own copy; a test asserts they agree).
TRUSTED_ADULT = re.compile(
    r"\b(trusted (?:adult|grown-up)s?|parents?|teachers?|grown-ups?|adults?|guardians?|"
    r"mum|mom|dad|caregivers?|counsell?ors?|family)\b", re.I)

_CONTEXT_MARK = "Context:\n"
_PREVIOUS_MARK = "\n\nPrevious educator message:\n"
_CHILD_MARK = "\n\nChild's latest reply:\n"

# Where each limitation is planned to be fixed (ticket 07, after PP1).
PP2_FIXES = [
    ("Trusted adult in almost every reply",
     "Lower the trusted-adult share in the train and validation completions (cap to be decided, for example 50%), "
     "keeping an adult where the scenario needs one (for example grooming)."),
    ("One closing question repeated",
     "Vary the closing questions in the training completions; report the most repeated closing line per arm."),
    ("Not in the child-facing loop",
     "Retrain with the same fixed LoRA config and re-run the same frozen comparison (same test split, settings, "
     "rubric, blind sheet). Only then decide whether the adapter may replace the template lines."),
    ("JSON validity, grammar compliance and controller correctness not measured",
     "Measure them once the adapter is wired into the FSM behind the controller grammar."),
]


# ------------------------------------------------------------------ prompts
def split_prompt(prompt_text):
    """A training-format INQUIRE prompt -> header, context, previous, child."""
    header, rest = prompt_text.split(_CONTEXT_MARK, 1)
    context, rest = rest.split(_PREVIOUS_MARK, 1)
    previous, child = rest.split(_CHILD_MARK, 1)
    return {"header": header, "context": context, "previous": previous, "child": child}


def build_prompt(header, context, previous, child):
    """Same layout the adapter was trained on (draft_inquire_dataset.build_inquire_prompt)."""
    return f"{header}{_CONTEXT_MARK}{context}{_PREVIOUS_MARK}{previous}{_CHILD_MARK}{child}"


def load_test_cases(path=TEST_FILE):
    """The held-out cases, in file order. `prompt` is the full single user message."""
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row["prompt"][0]["content"]
            parts = split_prompt(text)
            cases.append({"example_id": row["example_id"], "scenario_id": row["scenario_id"],
                          "reply_type": row["reply_type"], "risk_category": row["risk_category"],
                          "prompt": text, **parts})
    return cases


def session_prompt(header, risk_category, previous, child):
    """Prompt for a live session turn. The context is a generic line built from the trigger category, an
    approximation of the scenario descriptions the adapter was trained on (the dashboard says so)."""
    context = (f"A risk trigger was received for {risk_category.replace('_', ' ')}; "
               "the content was blocked.")
    return build_prompt(header, context, previous, child)


# ------------------------------------------------------------------ flags (heuristic, string matching)
def closing_sentence(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())[-1]


def flags(text):
    return {"trusted_adult": bool(TRUSTED_ADULT.search(text)), "closing": closing_sentence(text)}


def tally(responses):
    """Trusted-adult count and the most repeated closing sentence over a list of response texts."""
    n = len(responses)
    adults = sum(bool(TRUSTED_ADULT.search(r)) for r in responses)
    closing, count = (Counter(closing_sentence(r) for r in responses).most_common(1) or [("", 0)])[0]
    return {"n": n, "trusted_adult": adults, "top_closing": closing, "top_closing_count": count}


# ------------------------------------------------------------------ replay (evidence on disk)
def load_outputs(path):
    """example_id -> {"response", "latency_seconds"} from a raw_outputs_*.jsonl file."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["example_id"]] = {"response": r["response"], "latency_seconds": r.get("latency_seconds")}
    return out


def served_dir(root=EVIDENCE):
    """Newest evidence/inquire/inquire_served_<date>/ that holds both files, or None."""
    found = sorted(p for p in Path(root).glob("inquire_served_*")
                   if (p / "raw_outputs_control.jsonl").exists() and (p / "raw_outputs_adapted.jsonl").exists())
    return found[-1] if found else None


def load_replay(directory):
    """{"control": {...}, "adapted": {...}} for one evidence folder."""
    d = Path(directory)
    return {"control": load_outputs(d / "raw_outputs_control.jsonl"),
            "adapted": load_outputs(d / "raw_outputs_adapted.jsonl")}


def evaluated_figures(directory=EVALUATED_DIR):
    """The static strip: quality, format and latency per arm from the saved blind scoring, kept apart,
    plus the trusted-adult and repeated-closing tallies recomputed from the raw outputs."""
    d = Path(directory)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    replay = load_replay(d)
    figures = {}
    for arm in ("control", "adapted"):
        a = summary["arms"][arm]
        figures[arm] = {
            "rubric_overall": a["rubric"]["overall"],
            "handles_reply_type": a["rubric"]["per_criterion"]["handles_reply_type"],
            "format_pass_rate": a["format_check"]["pass_rate"],
            "stated_emotion_violations": a["stated_emotion_violations"],
            "latency_mean": a["latency_seconds"]["mean"],
            **tally([v["response"] for v in replay[arm].values()]),
        }
    figures["sample"] = summary["sample"]
    return figures


# ------------------------------------------------------------------ live (LM Studio), shadow only
def live_lines(prompt, call, base_id, adapted_id, max_tokens=80):
    """Ask both models the same prompt at temperature 0. `call` is call_local_model-compatible and takes a
    `model` keyword. A failure is returned as text for the panel; it never raises into the dialogue."""
    out = {}
    for arm, model_id in (("control", base_id), ("adapted", adapted_id)):
        try:
            text, secs = call(prompt, model=model_id, temperature=0, max_tokens=max_tokens)
            out[arm] = {"response": text.strip(), "latency_seconds": secs, "error": None}
        except Exception as e:
            out[arm] = {"response": "", "latency_seconds": None, "error": repr(e)}
    return out

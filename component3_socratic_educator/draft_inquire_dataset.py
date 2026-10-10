"""
draft_inquire_dataset.py

Drafts new INQUIRE examples for the Component 3 fine-tuning pilot with a
stronger model (Gemini) through distilabel. Drafts are raw material: the owner
reviews every row before it becomes a seed example (see `review` in each row).

Runs in three resumable stages, each writing to --out:

    1. scenarios  30 new scenario contexts spread over RiskCategory, then the
                  10 test scenarios drawn with a fixed seed (frozen BEFORE any
                  child reply or educator draft exists)
    2. replies    per scenario: one previous educator message + one child reply
                  per reply type (stated emotion, don't know, refusal,
                  off-topic, own safer plan)
    3. drafts     per row: 2 educator drafts for the exact INQUIRE training
                  prompt, few-shot examples from training-only scenarios

Self-contained on purpose (stdlib + distilabel) so it runs on Colab with only
this file and data/educator_seed.jsonl uploaded.

Usage (Colab, after `from google.colab import ai` has set MODEL_PROXY_*):
    python draft_inquire_dataset.py scenarios --seed-file educator_seed.jsonl --out drafts
    python draft_inquire_dataset.py replies   --seed-file educator_seed.jsonl --out drafts
    python draft_inquire_dataset.py drafts    --seed-file educator_seed.jsonl --out drafts
Fallback backend (free AI Studio key in GEMINI_API_KEY):
    python draft_inquire_dataset.py scenarios ... --backend aistudio --model gemini-2.5-pro
"""
import argparse
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Mirrors src.schemas.RiskCategory; kept literal so this file runs on Colab alone.
RISK_CATEGORIES = (
    "explicit_visual", "violence", "self_harm_imagery", "hate_speech",
    "cyberbullying", "grooming_language", "self_harm_language", "unknown_flagged",
)
REPLY_TYPES = ("stated_emotion", "dont_know", "refusal", "off_topic", "own_safer_plan")

# Training-only scenarios: S026 never enters the test set, and S001 is excluded
# because INQ001 breaks the verbatim stated-emotion rule.
FEWSHOT_IDS = ("INQ011", "INQ013", "INQ017", "INQ026", "INQ027", "INQ028")

NEW_SCENARIOS = 30
FIRST_NEW_SCENARIO = 27   # S001-S026 already exist
FIRST_NEW_EXAMPLE = 30    # INQ001-INQ029 already exist
TEST_SCENARIOS = 10
TEST_SPLIT_SEED = 20261008
DRAFTS_PER_ROW = 2
DRAFT_PROMPT_VERSION = "inquire-draft-v1"

_CONTEXT_MARK = "Context:\n"
_PREVIOUS_MARK = "\n\nPrevious educator message:\n"
_CHILD_MARK = "\n\nChild's latest reply:\n"


# ---------------------------------------------------------------- io
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- INQUIRE prompt
def inquire_header(seed_rows):
    """The instruction block every INQUIRE seed prompt starts with (taken from the data, not retyped)."""
    headers = {r["prompt"][0]["content"].split(_CONTEXT_MARK)[0]
               for r in seed_rows if r["example_id"].startswith("INQ")}
    if len(headers) != 1:
        raise ValueError(f"expected one INQUIRE instruction header, found {len(headers)}")
    return headers.pop()


def build_inquire_prompt(header, context, previous, child):
    return f"{header}{_CONTEXT_MARK}{context}{_PREVIOUS_MARK}{previous}{_CHILD_MARK}{child}"


def parse_inquire_prompt(text):
    _, rest = text.split(_CONTEXT_MARK, 1)
    context, rest = rest.split(_PREVIOUS_MARK, 1)
    previous, child = rest.split(_CHILD_MARK, 1)
    return {"context": context, "previous": previous, "child": child}


def fewshot_examples(seed_rows):
    by_id = {r["example_id"]: r for r in seed_rows}
    shots = []
    for eid in FEWSHOT_IDS:
        row = by_id[eid]
        shots.append({"example_id": eid, **parse_inquire_prompt(row["prompt"][0]["content"]),
                      "completion": row["completion"][0]["content"]})
    return shots


# ---------------------------------------------------------------- stage 1: scenarios
def category_quota(categories, total):
    base, extra = divmod(total, len(categories))
    return {c: base + (1 if i < extra else 0) for i, c in enumerate(categories)}


def build_scenario_request(category, n, existing_contexts):
    examples = "\n".join(f"- {c}" for c in existing_contexts)
    return f"""You are helping build a sanitized, synthetic training dataset for a child digital-safety educator (children aged 11-15).

Write {n} NEW scenario context lines for the risk category "{category}".

Each line describes, in one sentence, what an upstream detector flagged and (optionally) what the system or child did. Follow the style of these existing lines exactly:
{examples}

Rules:
- Start every line with "A risk trigger was received for".
- Describe the situation at a high level only. Never include graphic, sexual, or self-harm method details, slurs, or quoted harmful text.
- No names, usernames, schools, cities, or other identifying details.
- Vary the platform (games, video, social media, messaging, browser, school apps) and the situation.
- Each line must be clearly different from the existing lines and from each other.

Return ONLY a JSON array of {n} strings."""


def parse_json_block(text):
    """First JSON array/object in a model reply, tolerating ``` fences and surrounding chatter."""
    if text is None:
        raise ValueError("empty model output")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidate = fenced.group(1) if fenced else text
    start = min((i for i in (candidate.find("["), candidate.find("{")) if i != -1), default=-1)
    if start == -1:
        raise ValueError(f"no JSON found in: {text[:120]!r}")
    obj, _ = json.JSONDecoder().raw_decode(candidate[start:])
    return obj


def assign_scenarios(drafted, start=FIRST_NEW_SCENARIO):
    """{category: [context, ...]} -> numbered scenarios, in RiskCategory order."""
    ordered = [c for c in RISK_CATEGORIES if c in drafted] + [c for c in drafted if c not in RISK_CATEGORIES]
    scenarios, n = [], start
    for category in ordered:
        for context in drafted[category]:
            scenarios.append({"scenario_id": f"S{n:03d}", "risk_category": category, "context": context})
            n += 1
    return scenarios


def select_test_scenarios(scenario_ids, k=TEST_SCENARIOS, seed=TEST_SPLIT_SEED):
    return sorted(random.Random(seed).sample(sorted(scenario_ids), k))


def _tokens(text):
    return set(re.findall(r"[a-z']+", text.lower()))


def near_duplicates(contexts, threshold=0.7):
    """Pairs of scenario ids whose contexts share most words (Jaccard), for the reviewer to merge or keep."""
    ids = sorted(contexts)
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ta, tb = _tokens(contexts[a]), _tokens(contexts[b])
            score = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
            if score >= threshold:
                pairs.append((a, b, round(score, 2)))
    return pairs


# ---------------------------------------------------------------- stage 2: replies
_REPLY_GUIDE = {
    "stated_emotion": "names how they feel in their own words (e.g. annoyed, scared, embarrassed, mad, confused)",
    "dont_know": "is unsure what they feel or what happened (vary the wording; not always 'I don't know')",
    "refusal": "declines to talk about it",
    "off_topic": "changes the subject (asks for something unrelated, talks about something else)",
    "own_safer_plan": "suggests their OWN safer next step (e.g. leaving the chat, telling a parent, blocking someone)",
}


def build_replies_request(scenario):
    guide = "\n".join(f'- "{t}": the child {g}' for t, g in _REPLY_GUIDE.items())
    return f"""You are helping build a sanitized, synthetic training dataset for a child digital-safety educator.

Scenario context: {scenario['context']}

1. Write "previous_educator_message": the educator's short, calm opening question to the child about this moment. Do not name or describe the flagged content, do not assume the child's emotion, do not describe the educator's own feelings.
2. Write one realistic child reply (age 11-15, at most 25 words) to that question for EACH reply type:
{guide}

Rules for child replies: natural and varied wording, no graphic or sexual detail, no slurs, no self-harm methods, no names or identifying details.

Return ONLY a JSON object:
{{"previous_educator_message": "...", "replies": {{{", ".join(f'"{t}": "..."' for t in REPLY_TYPES)}}}}}"""


def validate_replies(obj):
    if not isinstance(obj, dict) or not str(obj.get("previous_educator_message", "")).strip():
        raise ValueError("missing previous_educator_message")
    replies = obj.get("replies")
    missing = [t for t in REPLY_TYPES if not (isinstance(replies, dict) and str(replies.get(t, "")).strip())]
    if missing:
        raise ValueError(f"missing reply types: {missing}")
    return obj


# ---------------------------------------------------------------- stage 3: drafts
_DRAFT_GUIDE = {
    "stated_emotion": "Acknowledge the feeling using the child's exact emotion word, then ask one question toward a safe next step.",
    "dont_know": "Accept the uncertainty and offer an easy either/or choice.",
    "refusal": "Respect the refusal without asking for an explanation; offer a gentle choice of safe next steps.",
    "off_topic": "Do not follow the new topic; gently redirect to a safe next step for this situation.",
    "own_safer_plan": "Acknowledge the child's own plan in their words and invite them to take that step; do not grant permission to resume the flagged activity.",
}


def build_draft_request(training_prompt, reply_type, shots):
    examples = "\n\n".join(
        f"Context: {s['context']}\nPrevious educator message: {s['previous']}\n"
        f"Child's latest reply: {s['child']}\nGood response: {s['completion']}"
        for s in shots
    )
    return f"""You are drafting reviewed training data for a small on-device digital-safety educator model. Write the ideal educator response for the input at the end.

Hard rules:
- One or two short sentences, at most 30 words, exactly one question, and it ends the response.
- Only name an emotion if the child used that exact word; never paraphrase it ("annoyed" stays "annoyed").
- Never invent facts, events, emotions, or the educator's own feelings. No jokes. Never grant permission to resume the blocked activity or claim anyone was notified.
- Do not repeat flagged content or shame the child.
- If the context involves self-harm, threats, or grooming, steer toward a trusted adult.

This reply type is "{reply_type}": {_DRAFT_GUIDE[reply_type]}

Examples of good responses:

{examples}

Input (this is exactly what the small model will see):
---
{training_prompt}
---

Return ONLY the child-facing response text, with no quotes or labels."""


def assemble_rows(header, scenarios, replies, test_ids, start=FIRST_NEW_EXAMPLE):
    rows, n = [], start
    for sc in scenarios:
        rep = replies[sc["scenario_id"]]
        for reply_type in REPLY_TYPES:
            prompt = build_inquire_prompt(header, sc["context"], rep["previous_educator_message"],
                                          rep["replies"][reply_type])
            rows.append({
                "example_id": f"INQ{n:03d}",
                "scenario_id": sc["scenario_id"],
                "risk_category": sc["risk_category"],
                "reply_type": reply_type,
                "split": "test" if sc["scenario_id"] in test_ids else "train_or_val",
                "prompt": [{"role": "user", "content": prompt}],
                "drafts": [],
                "completion": None,
                "review": {"status": "pending", "chosen_draft": None, "rewritten": False, "notes": ""},
                "provenance": {},
            })
            n += 1
    return rows


# ---------------------------------------------------------------- distilabel
# distilabel's OpenAILLM always sends these; AI Studio's OpenAI-compatible endpoint
# rejects them with 400 INVALID_ARGUMENT "Unknown name" (8 Oct). It also rejects
# explicit nulls (stop=None -> "Value is not a string: null"), so unset fields go too.
GEMINI_UNSUPPORTED_FIELDS = ("logprobs", "top_logprobs", "frequency_penalty", "presence_penalty")


def _without_unsupported_fields(create):
    async def create_for_gemini(**kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None and k not in GEMINI_UNSUPPORTED_FIELDS}
        return await create(**kwargs)
    create_for_gemini.drops_unsupported_fields = True
    return create_for_gemini


try:  # optional at import time so the pure helpers stay testable without distilabel
    from distilabel.models import OpenAILLM as _OpenAILLM
except ImportError:  # pragma: no cover
    _OpenAILLM = None

if _OpenAILLM is not None:
    class AIStudioLLM(_OpenAILLM):
        """OpenAILLM for AI Studio's Gemini endpoint: strips request fields Gemini rejects."""

        def load(self):
            super().load()
            completions = self._aclient.chat.completions
            completions.create = _without_unsupported_fields(completions.create)


def make_llm(backend, model, temperature, max_new_tokens=8192):
    # Written against distilabel 1.5.x; check import paths if your version differs.
    from distilabel.models import OpenAILLM
    if backend == "aistudio":
        OpenAILLM = AIStudioLLM
    if backend == "colab":
        host, key = os.environ.get("MODEL_PROXY_HOST"), os.environ.get("MODEL_PROXY_API_KEY")
        if not (host and key):
            sys.exit("MODEL_PROXY_HOST / MODEL_PROXY_API_KEY not set: run `from google.colab import ai` "
                     "and one ai.generate_text() call in the notebook first.")
        base_url = f"{host.rstrip('/')}/models/openapi"
    else:  # aistudio: free Gemini API key, OpenAI-compatible endpoint
        key = os.environ.get("GEMINI_API_KEY") or sys.exit("GEMINI_API_KEY not set")
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # Thinking tokens can count against max_new_tokens. Colab's proxy reserved quota as if no limit was
    # sent (65,536 tokens, 8 Oct), and the reservation multiplies by --batch-size concurrent requests.
    return OpenAILLM(model=model, base_url=base_url, api_key=key, timeout=300, max_retries=3,
                     generation_kwargs={"temperature": temperature, "max_new_tokens": max_new_tokens})


def expand_requests(prompts, copies):
    """One independent request per copy. distilabel's num_generations relies on the OpenAI `n`
    parameter, which LM Studio ignores (smoke-tested 8 Oct) and Gemini's proxy may too."""
    return [{"key": k, "copy": i, "instruction": v} for k, v in prompts.items() for i in range(copies)]


def regroup_generations(table):
    out = {}
    for r in sorted(table, key=lambda r: (r["key"], r["copy"])):
        out.setdefault(r["key"], []).append(r["generation"])
    return out


def run_generation(llm, prompts, num_generations=1, name="inquire-drafts", batch_size=8):
    """{key: prompt} -> {key: [generation, ...]} (None for failed calls)."""
    from distilabel.pipeline import Pipeline
    from distilabel.steps import LoadDataFromDicts
    from distilabel.steps.tasks import TextGeneration

    with Pipeline(name=name) as pipe:
        load = LoadDataFromDicts(data=expand_requests(prompts, num_generations))
        gen = TextGeneration(llm=llm, input_batch_size=batch_size)
        load >> gen
    return regroup_generations(pipe.run(use_cache=False)["default"]["train"])


def provenance(args):
    return {"model": args.model, "backend": args.backend, "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
            "drafting_prompt_version": DRAFT_PROMPT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


# ---------------------------------------------------------------- stages
def stage_scenarios(args, seed_rows, out):
    path = out / "scenarios.json"
    if path.exists():
        print(f"{path} exists; scenarios and test set are frozen. Delete it only to start over.")
        return
    inquire_header(seed_rows)  # fail fast on a wrong seed file before spending model calls
    existing = [parse_inquire_prompt(r["prompt"][0]["content"])["context"]
                for r in seed_rows if r["example_id"].startswith("INQ")]
    existing = list(dict.fromkeys(existing))
    quota = category_quota(RISK_CATEGORIES, NEW_SCENARIOS)
    llm = make_llm(args.backend, args.model, args.temperature, args.max_new_tokens)
    raw = run_generation(llm, {c: build_scenario_request(c, n, existing) for c, n in quota.items()},
                         name="inquire-scenarios", batch_size=args.batch_size)
    drafted, problems = {}, []
    for category, n in quota.items():
        try:
            lines = [str(x).strip() for x in parse_json_block(raw[category][0])]
            if len(lines) < n:
                raise ValueError(f"got {len(lines)} of {n}")
            drafted[category] = lines[:n]
        except (ValueError, KeyError) as e:
            problems.append(f"{category}: {e}")
    write_json(out / "scenarios_raw.json", raw)
    if problems:
        sys.exit("Scenario drafting incomplete, nothing frozen. Re-run this stage.\n" + "\n".join(problems))
    scenarios = assign_scenarios(drafted)
    test_ids = select_test_scenarios([s["scenario_id"] for s in scenarios])
    dupes = near_duplicates({**{f"existing:{i}": c for i, c in enumerate(existing)},
                             **{s["scenario_id"]: s["context"] for s in scenarios}})
    write_json(path, {"provenance": provenance(args),
                      "scenarios": scenarios, "test_scenario_ids": test_ids,
                      "test_split_seed": TEST_SPLIT_SEED, "near_duplicates_for_review": dupes})
    print(f"{len(scenarios)} scenarios, test = {test_ids}, {len(dupes)} near-duplicate pairs to review")


def stage_replies(args, seed_rows, out):
    frozen = json.loads((out / "scenarios.json").read_text(encoding="utf-8"))
    path = out / "replies.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [s for s in frozen["scenarios"] if s["scenario_id"] not in done]
    if not todo:
        print("all replies drafted")
        return
    llm = make_llm(args.backend, args.model, args.temperature, args.max_new_tokens)
    raw = run_generation(llm, {s["scenario_id"]: build_replies_request(s) for s in todo},
                         name="inquire-replies", batch_size=args.batch_size)
    failed = []
    for s in todo:
        try:
            done[s["scenario_id"]] = {**validate_replies(parse_json_block(raw[s["scenario_id"]][0])),
                                      "provenance": provenance(args)}
        except (ValueError, KeyError) as e:
            failed.append(f"{s['scenario_id']}: {e}")
    write_json(path, done)
    print(f"{len(done)}/{len(frozen['scenarios'])} scenarios have replies")
    if failed:
        print("Re-run this stage to retry:\n" + "\n".join(failed))


def stage_drafts(args, seed_rows, out):
    frozen = json.loads((out / "scenarios.json").read_text(encoding="utf-8"))
    replies = json.loads((out / "replies.json").read_text(encoding="utf-8"))
    missing = [s["scenario_id"] for s in frozen["scenarios"] if s["scenario_id"] not in replies]
    if missing:
        sys.exit(f"replies missing for {missing}; finish the replies stage first")
    header = inquire_header(seed_rows)
    shots = fewshot_examples(seed_rows)
    path = out / "inquire_drafts.jsonl"
    rows = load_jsonl(path) if path.exists() else assemble_rows(
        header, frozen["scenarios"], replies, set(frozen["test_scenario_ids"]))
    for r in rows:
        r["provenance"].setdefault("replies", replies[r["scenario_id"]]["provenance"])
    todo = [r for r in rows if len([x for x in r["drafts"] if x]) < DRAFTS_PER_ROW]
    if todo:
        llm = make_llm(args.backend, args.model, args.temperature, args.max_new_tokens)
        raw = run_generation(llm, {r["example_id"]: build_draft_request(r["prompt"][0]["content"], r["reply_type"], shots)
                                   for r in todo}, num_generations=DRAFTS_PER_ROW, name="inquire-educator-drafts",
                             batch_size=args.batch_size)
        for r in todo:
            r["drafts"] = [g.strip().strip('"') for g in raw.get(r["example_id"], []) if g]
            r["provenance"]["drafts"] = provenance(args)
    write_jsonl(path, rows)
    short = [r["example_id"] for r in rows if len(r["drafts"]) < DRAFTS_PER_ROW]
    print(f"{len(rows) - len(short)}/{len(rows)} rows have {DRAFTS_PER_ROW} drafts -> {path}")
    if short:
        print("Re-run this stage to retry: " + ", ".join(short))


def build_parser():
    ap = argparse.ArgumentParser(description="Draft INQUIRE fine-tuning examples with Gemini via distilabel.")
    ap.add_argument("stage", choices=["scenarios", "replies", "drafts"])
    ap.add_argument("--seed-file", default=str(Path(__file__).resolve().parent / "data" / "educator_seed.jsonl"))
    ap.add_argument("--out", default="drafts")
    ap.add_argument("--backend", choices=["colab", "aistudio"], default="colab")
    ap.add_argument("--model", default="google/gemini-3.1-pro-preview")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=8192,
                    help="output token cap sent to the model (recorded in provenance)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="concurrent requests; lower (e.g. 2) if Colab reports the estimated cost exceeds your quota")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_rows = load_jsonl(args.seed_file)
    {"scenarios": stage_scenarios, "replies": stage_replies, "drafts": stage_drafts}[args.stage](args, seed_rows, out)


if __name__ == "__main__":
    main()

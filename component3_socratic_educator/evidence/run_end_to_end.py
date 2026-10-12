"""
Full mock end-to-end run (plan, Fri 9 Oct): trigger -> FSM -> constrained
dialogue -> schema-validated record -> Component 4.

Eight sessions: the Component 1 and Component 2 mock triggers, each with four
scripted children (reflective, defensive, distressed, withdrawn). Each session
goes through integration/end_to_end_pipeline.run_session; then all eight
Component 4 input rows go to Component 4 Function 1's real
run_emotional_pattern() for this device's child.

    --mode online   live model in LM Studio (localhost:1234); refuses to record
                    if any INTERCEPT call got no reply
    --mode offline  no model: every INTERCEPT line must be the fallback line

Run from the repo root with an interpreter that has pandas and joblib:
    python component3_socratic_educator/evidence/run_end_to_end.py --mode offline

Writes evidence/end_to_end_<date>_<mode>.jsonl (one row per session), a
.summary.json and a readable .log.txt; never overwrites a recorded run.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from component3_socratic_educator.src import config                                       # noqa: E402
from component3_socratic_educator.src.grammar_decoder import INTERCEPT_PROMPT, INTERCEPT_SCHEMA  # noqa: E402
from component3_socratic_educator.src.schemas import _validate_against_schema              # noqa: E402
from integration.comp3_to_comp4 import DEVICE_CHILD_ID                                     # noqa: E402
from integration.end_to_end_pipeline import build_trigger_from, offline_model, run_session  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
C4_MODELS = ROOT / "component4_profiling_xai" / "models"
SOURCES = ("component1", "component2")
# A reply under 4 words is not "complete", so INQUIRE asks again (up to 3 times).
CHILDREN = {
    "reflective": ["I was just watching videos", "I guess I should close it and tell my mum next time"],
    "defensive": ["It wasn't me", "whatever", "no big deal", "It was nothing, my friend sent it"],
    "distressed": ["It was really scary", "I feel sick", "so scared", "I can't stop crying about it"],
    "withdrawn": ["idk", "nothing", "leave me alone", "I don't want to talk"],
}


def run_sessions(model_call):
    logs = []
    for source in SOURCES:
        for child, script in CHILDREN.items():
            replies = iter(script)
            start = time.perf_counter()
            log = run_session(build_trigger_from(source), lambda _p, r=replies: next(r, "ok"), model_call=model_call)
            logs.append({"source": source, "scripted_child": child,
                         "seconds": round(time.perf_counter() - start, 3), **log})
    return logs


def component4_patterns(rows):
    """Component 4 Function 1 on every session row, or why it could not run."""
    if not (C4_MODELS / "emotional_pattern_kmeans.pkl").exists():
        return {"ran": False, "why": "not run: Component 4's trained artifacts are not in component4_profiling_xai/models/"}
    import pandas as pd
    from component4_profiling_xai.src.emotional_pattern import run_emotional_pattern
    patterns = run_emotional_pattern(pd.DataFrame(rows))
    return {"ran": True, "patterns": {child: {"sessions": int(r["sessions"]), "cluster": int(r["cluster"]),
                                              "pattern": r["pattern"]} for child, r in patterns.iterrows()},
            "artifacts_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in sorted(C4_MODELS.glob("emotional_pattern_*"))}}


def schema_valid(record):
    try:
        _validate_against_schema(record, "comp3_to_comp4.schema.json")
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["online", "offline"], required=True)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"end_to_end_{stamp}_{args.mode}.jsonl"
    outputs = [out, out.with_suffix(".summary.json"), out.with_suffix(".log.txt")]
    if any(o.exists() for o in outputs):   # check all three first, so a rerun cannot leave a partial set
        sys.exit(f"{out.name} (or its .summary.json/.log.txt) is already recorded; nothing was overwritten.")

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    logs = run_sessions(None if args.mode == "online" else offline_model)
    intercepts = [log["turns"][0] for log in logs]
    unreachable = sum(t["reason"] == "model_unreachable" for t in intercepts)
    if args.mode == "online" and unreachable:
        sys.exit(f"{unreachable}/{len(logs)} INTERCEPT calls got no model reply: start the LM Studio server "
                 "and rerun. Nothing was recorded.")
    if args.mode == "offline" and any(t["source"] != "fallback" for t in intercepts):
        sys.exit("offline run produced a non-fallback INTERCEPT line; nothing was recorded")

    c4 = component4_patterns([log["component4_row"] for log in logs])
    seconds = [log["seconds"] for log in logs]
    session_seconds = {"min": min(seconds), "median": round(statistics.median(seconds), 3), "max": max(seconds)}
    valid = sum(schema_valid(log["record"]) for log in logs)
    summary = {
        "mode": args.mode,
        "run_started": started,
        "sessions": len(logs),
        "records_schema_valid": f"{valid}/{len(logs)} (re-checked here against comp3_to_comp4.schema.json)",
        "intercept_sources": dict(Counter(t["source"] for t in intercepts)),
        "intercept_fallback_reasons": dict(Counter(t["reason"] for t in intercepts if t["reason"])),
        "emotional_state_by_scripted_child": {f"{log['source']}/{log['scripted_child']}": log["record"]["emotional_state"]
                                              for log in logs},
        "component4_function1": c4,
        "session_seconds": session_seconds,
        "session_seconds_note": "wall-clock per session, including scripted replies and the stub report; "
                                "not model latency (measured against the 3 s target on Tue 13 Oct)",
        "settings": {
            "triggers": "mock fixtures: component1 mock_inputs/sample_trigger.json, component2 "
                        "mock_inputs/sample_analysis.json (Component 2's real engine needs Hugging Face "
                        "downloads, unavailable offline)",
            "child_replies": CHILDREN, "child_id": f"{DEVICE_CHILD_ID} (device placeholder)",
            "model_id": config.LOCAL_MODEL_ID if args.mode == "online" else "none (offline)",
            "server": "LM Studio (llama.cpp engine), localhost:1234" if args.mode == "online" else "none (offline)",
            "lm_studio_version": "not recorded", "hardware": "not recorded",
            "response_format": "json_schema (strict)", "schema": INTERCEPT_SCHEMA, "prompt": INTERCEPT_PROMPT,
            "max_tokens": 100,
            "temperature": "server default (not set)" if args.mode == "online" else "n/a",
            "inquire": "template lines (no model until after PP1)",
            "component4_artifacts": "reproduced on 11 Oct by running Component 4's Function 1 notebook unchanged "
                                    "(models/ is gitignored); checked then against its committed outputs/ "
                                    "(40/40 children, trend file); not re-checked by this run (hashes above)",
        },
        "limitations": [
            "Triggers are the Component 1 and 2 mock fixtures, not live detections.",
            "Child replies are scripted; INQUIRE lines are templates and do not adapt to the reply.",
            "The parent-facing report is Component 4's stub (generate_report); only Function 1 is real.",
            "All 8 sessions share one device child_id, so Function 1 profiles a synthetic child mixing four "
            "personas; the pattern shows the chain works, not anything about a real child.",
        ],
    }

    text = [f"End-to-end run, {args.mode}: {len(logs)} sessions"]
    for log in logs:
        r = log["record"]
        text.append(f"\n=== {log['source']} trigger ({log['trigger']['risk_category']}, "
                    f"confidence {log['trigger']['confidence_score']}) / {log['scripted_child']} child, "
                    f"{log['seconds']} s ===")
        for t in log["turns"]:
            tag = f" [{t['source']}{': ' + t['reason'] if t['reason'] else ''}]" if t["source"] else ""
            text.append(f"  {t['state']:<9} {t['system_text']}{tag}")
            if t["child_response"] is not None:
                text.append(f"  {'child':<9} {t['child_response']}")
            if t["source"] == "fallback" and t["model_output"]:
                text.append(f"  {'(blocked)':<9} {t['model_output']}")
        text.append(f"  record    risk={r['risk_level']} emotion={r['emotional_state']} "
                    f"self_regulation={r['self_regulation_shown']} escalate={r['escalation_flag']} "
                    f"turns={r['dialogue_turns']}")
    text.append(f"\nINTERCEPT sources: {summary['intercept_sources']} {summary['intercept_fallback_reasons']}")
    text.append(f"Records schema-valid: {summary['records_schema_valid']}")
    text.append(f"Component 4 Function 1: {c4.get('patterns', c4.get('why'))}")
    text.append("Limitations:")
    text.extend(f"  - {item}" for item in summary["limitations"])
    text.append(f"Session seconds (min/median/max): {session_seconds['min']} / {session_seconds['median']} / {session_seconds['max']} (wall-clock, not latency)")

    with open(out, "x", encoding="utf-8", newline="\n") as f:   # never overwrite a recorded run
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
    with open(out.with_suffix(".summary.json"), "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with open(out.with_suffix(".log.txt"), "x", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(text) + "\n")
    print("\n".join(text))
    print(f"\n-> {out.name}, .summary.json, .log.txt")


if __name__ == "__main__":
    main()

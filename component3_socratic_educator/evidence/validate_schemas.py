"""
Re-run schema validation across all current outputs (plan, Thu 15 Oct; target 100%).

Checks every output that crosses a Component 3 contract, with src/schema_audit.py:

    in     triggers           comp1_to_comp3 / comp2_to_comp3 (mock fixtures, recorded runs, sweep)
    model  INTERCEPT replies  raw validity from the recorded fallback reason ("valid",
                              "parse_failure", "no_reply"); the delivered line against
                              the controller grammar and post-check
           INQUIRE lines      against today's INQUIRE grammar, for runs on today's
                              template lines (14 Oct onwards)
    out    records            comp3_to_comp4 (recorded end-to-end runs, sweep)

Recorded evidence is only read. A fresh sweep then runs today's FSM over every
risk category x scripted child x confidence 0.3 / 0.6 / 0.9. The 6 Oct samples
predate the INTERCEPT schema (unconstrained) and are listed as not applicable.

Run from component3_socratic_educator/ (online needs LM Studio):
    python -m evidence.validate_schemas [--mode offline|online] [--tag x]

Writes, in the run folder evidence/system/schema_validation_<date>[_<tag>]/, a .jsonl
(one row per checked item), a .summary.json and a .log.txt; never overwrites.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from evidence.run_info import lm_studio_version
from src import config, schema_audit
from src.fsm_controller import DialogueState, FSMController
from src.grammar_decoder import GrammarConstrainedGenerator
from src.schemas import _CONTRACTS_DIR, RiskCategory, TriggerPayload

EV = Path(__file__).resolve().parent
REPO = EV.parents[1]
TRIGGER, RECORD = "in: trigger", "out: record"
REPLY, INTERCEPT_LINE, INQUIRE_LINE = "model: INTERCEPT reply", "model: INTERCEPT delivered", "model: INQUIRE delivered"
# Component 1 and 2's own sample outputs (what they send Component 3)
UPSTREAM_FIXTURES = ("component1_screen_monitoring/mock_inputs/sample_trigger.json",
                     "component2_hate_speech_detection/mock_inputs/sample_analysis.json")
MOCK = EV.parent / "mock_inputs"
CHILDREN = {   # the scripted children of the end-to-end runs
    "reflective": ["I was just watching videos", "I guess I should close it and tell my mum next time"],
    "defensive": ["It wasn't me", "whatever", "no big deal", "It was nothing, my friend sent it"],
    "distressed": ["It was really scary", "I feel sick", "so scared", "I can't stop crying about it"],
    "withdrawn": ["idk", "nothing", "leave me alone", "I don't want to talk"],
}
TEXT_CATEGORIES = {"hate_speech", "cyberbullying", "grooming_language", "self_harm_language"}
CURRENT_INQUIRE_FROM = "2026-10-14"   # INQUIRE's template lines changed that day
# Runs recorded before the post-check's invites_description rule existed: the
# constrained samples (62389bc) and the first adversarial run, both 11 Oct; the rule
# came with the after-fix rerun (cf37750). A failure there by that rule is a line
# today's code blocks, reported apart from today's results.
BEFORE_RULE = {"adversarial_intercept_2026-10-11/adversarial_intercept_2026-10-11.jsonl":
               "invites_description", "intercept_samples_constrained_2026-10-11/"
               "intercept_samples_constrained_2026-10-11.jsonl": "invites_description"}


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _date_of(path):
    return next((part for part in path.stem.split("_") if part[:4] == "2026"), "")


def check_recorded(checks):
    """Every recorded file, read only."""
    for name in sorted(p.name for p in MOCK.glob("trigger_from_component*.json")):
        checks.append(("mock_inputs/" + name, TRIGGER, name,
                       schema_audit.trigger_error(json.loads((MOCK / name).read_text(encoding="utf-8")))))
    for rel in UPSTREAM_FIXTURES:
        checks.append((rel, TRIGGER, Path(rel).name,
                       schema_audit.trigger_error(json.loads((REPO / rel).read_text(encoding="utf-8")))))

    for path in sorted(EV.glob("system/end_to_end_*/*.jsonl")):
        src = f"{path.parent.name}/{path.name}"
        for row in _rows(path):
            sid = f"{row['source']}/{row['scripted_child']}"
            checks.append((src, TRIGGER, sid, schema_audit.trigger_error(row["trigger"])))
            checks.append((src, RECORD, sid, schema_audit.record_error(row["record"])))
            intercept = row["turns"][0]
            checks.append((src, REPLY, sid, schema_audit.intercept_outcome(intercept["source"], intercept["reason"])))
            checks.append((src, INTERCEPT_LINE, sid, schema_audit.delivered_intercept_error(intercept["system_text"])))
            if _date_of(path) >= CURRENT_INQUIRE_FROM:
                for j, t in enumerate(x for x in row["turns"] if x["state"] == "INQUIRE"):
                    checks.append((src, INQUIRE_LINE, f"{sid}#{j}",
                                   schema_audit.delivered_inquire_error(t["system_text"])))

    for path in sorted(EV.glob("intercept/adversarial_intercept_*/*.jsonl")):
        src = f"{path.parent.name}/{path.name}"
        for row in _rows(path):
            sid = f"{row['attack_id']}/{row['risk_category']}"
            checks.append((src, REPLY, sid, schema_audit.intercept_outcome(row["source"], row["reason"])))
            checks.append((src, INTERCEPT_LINE, sid, schema_audit.delivered_intercept_error(row["delivered"])))

    for path in sorted(EV.glob("intercept/intercept_samples_constrained_*/*.jsonl")):
        src = f"{path.parent.name}/{path.name}"
        for row in _rows(path):
            sid = f"{row['risk_category']}#{row['run']}"
            checks.append((src, REPLY, sid, schema_audit.intercept_outcome(row["source"], row["fallback_reason"])))
            checks.append((src, INTERCEPT_LINE, sid, schema_audit.delivered_intercept_error(row["text"])))

    for path in sorted(EV.glob("system/latency_*/*.jsonl")):
        src = f"{path.parent.name}/{path.name}"
        for i, row in enumerate(_rows(path)):
            w = row["waits"][0]
            checks.append((src, REPLY, f"{row['risk_category']}/{row['scripted_child']}#{i}",
                           schema_audit.intercept_outcome(w["source"], w["reason"])))


def _trigger(category, confidence):
    if category.value in TEXT_CATEGORIES:
        return TriggerPayload.new("component2_hate_speech_detection", "text_message", category, confidence)
    return TriggerPayload.new("component1_screen_monitoring", "image", category, confidence, app_or_window="browser")


def sweep(checks, mode):
    """Today's FSM over every category x scripted child x confidence."""
    def offline(*_a, **_k):
        raise ConnectionError("offline sweep: no model called")

    src = f"sweep ({mode}, today's code)"
    for category in RiskCategory:
        for child, script in CHILDREN.items():
            for confidence in (0.3, 0.6, 0.9):
                trigger = _trigger(category, confidence)
                sid = f"{category.value}/{child}/{confidence}"
                checks.append((src, TRIGGER, sid, schema_audit.trigger_error(trigger.to_dict())))
                lines = []
                generator = GrammarConstrainedGenerator(model_call=offline if mode == "offline" else None)
                generate = generator.generate_with_source

                def recording(state, *args, _generate=generate, _lines=lines, **kwargs):
                    line = _generate(state, *args, **kwargs)
                    _lines.append((state, line))
                    return line
                generator.generate_with_source = recording
                replies = iter(script)
                transcript = None
                try:
                    record, transcript = FSMController(generator=generator).run(
                        trigger, lambda _p, r=replies: next(r, "ok"))
                    checks.append((src, RECORD, sid, schema_audit.record_error(record.to_dict())))
                except Exception as e:   # a record that fails validate() raises inside run()
                    checks.append((src, RECORD, sid, f"run raised {e.__class__.__name__}: {e}"))
                for j, (state, line) in enumerate(lines):   # kept even if the run raised
                    if state == "INTERCEPT":
                        checks.append((src, REPLY, sid, schema_audit.intercept_outcome(line.source, line.reason)))
                        checks.append((src, INTERCEPT_LINE, sid, schema_audit.delivered_intercept_error(line.text)))
                    else:
                        checks.append((src, INQUIRE_LINE, f"{sid}#{j}", schema_audit.delivered_inquire_error(line.text)))


def tally(checks):
    """Per source and boundary: n, passed, and the failures (or raw-reply outcomes)."""
    table = defaultdict(lambda: defaultdict(int))
    for src, boundary, _sid, result in checks:
        t = table[(src, boundary)]
        t["n"] += 1
        if boundary == REPLY:
            t[result] += 1
        else:
            t["passed" if result is None else "failed"] += 1
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("offline", "online"), default="offline")
    ap.add_argument("--tag", help="suffix for a second run on the same day")
    args = ap.parse_args()

    run_name = f"schema_validation_{datetime.now():%Y-%m-%d}{'_' + args.tag if args.tag else ''}"
    out = EV / "system" / run_name / f"{run_name}.jsonl"
    outputs = [out, out.with_suffix(".summary.json"), out.with_suffix(".log.txt")]
    if any(p.exists() for p in outputs):
        sys.exit(f"{run_name} is already recorded: pass --tag. Nothing was overwritten.")

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    checks = []
    check_recorded(checks)
    sweep(checks, args.mode)
    if args.mode == "online" and any(s.startswith("sweep") and b == REPLY and r == "no_reply"
                                     for s, b, _, r in checks):
        sys.exit("online sweep: some INTERCEPT calls got no model reply; start LM Studio and rerun. Nothing recorded.")
    table = tally(checks)

    totals = defaultdict(lambda: defaultdict(int))
    for (src, boundary), t in table.items():
        for k, v in t.items():
            totals[boundary][k] += v
    def blocked_by_later_rule(src, boundary, result):
        return boundary == INTERCEPT_LINE and schema_audit.failed_only_by(result, BEFORE_RULE.get(src))

    counted = [c for c in checks if c[1] != REPLY and not blocked_by_later_rule(c[0], c[1], c[3])]
    failed = sum(r is not None for *_, r in counted)
    checked = len(counted)
    resolved = [{"source": s, "id": i, "rule": r} for s, b, i, r in checks if blocked_by_later_rule(s, b, r)]
    raw = totals[REPLY]
    replies = raw["valid"] + raw["parse_failure"]

    summary = {
        "run_started": started,
        "mode": args.mode,
        "result": f"{checked - failed}/{checked} contract and delivered-line checks passed ({failed} failed)",
        "blocked_by_a_later_rule": (f"{len(resolved)} INTERCEPT lines delivered on 11 Oct (62389bc, and the first "
                                    "adversarial run), before the post-check's invites_description rule existed "
                                    "(cf37750), fail it today: today's code blocks them. Not counted in the result "
                                    "above."),
        "model_raw_validity": (f"{raw['valid']}/{replies} INTERCEPT model replies fit INTERCEPT_SCHEMA "
                               f"({raw['parse_failure']} parse failures; {raw['no_reply']} calls with no reply, "
                               "not counted)") if replies else "no model replies recorded",
        "by_boundary": {b: dict(t) for b, t in sorted(totals.items())},
        "by_source": {f"{s} | {b}": dict(t) for (s, b), t in sorted(table.items())},
        "failures": [{"source": s, "boundary": b, "id": i, "error": r} for s, b, i, r in counted if r is not None],
        "resolved_lines": resolved,
        "not_applicable": {"intercept_samples_2026-10-06": "unconstrained samples from before the INTERCEPT schema "
                                                           "existed (6 Oct 'before' baseline)",
                           "inquire_* and eva001_*": "fine-tuning datasets and comparison outputs; they cross no "
                                                     "component contract",
                           "emotion_rules_eval_*, temperature_check_*": "console logs of other measurements; no "
                                                                         "contract payloads",
                           "component4_row in end-to-end runs": "not measured: Component 4's input row has no "
                                                                "contract schema (only comp3_to_comp4 does)",
                           "end_to_end_2026-10-12_* INQUIRE lines": "made with the template lines retired on 14 Oct; "
                                                                    "today's INQUIRE grammar is not theirs"},
        "settings": {"schemas_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(_CONTRACTS_DIR.glob("*.schema.json"))},
                     "sweep": {"categories": [c.value for c in RiskCategory], "children": list(CHILDREN),
                               "confidences": [0.3, 0.6, 0.9],
                               "model": config.LOCAL_MODEL_ID if args.mode == "online" else "none (offline)",
                               "temperature": config.INTERCEPT_TEMPERATURE if args.mode == "online" else "n/a",
                               "lm_studio_version": lm_studio_version() if args.mode == "online" else "n/a",
                               "hardware": "not recorded"},
                     "upstream_fixtures": list(UPSTREAM_FIXTURES)},
        "limitations": [
            "Raw model validity is read from each call's recorded fallback reason; most files keep the parsed "
            "line, not the raw JSON text.",
            "Recorded triggers are mock fixtures and simulated Component 1/2 output, not live detections.",
            "The sweep's child replies are scripted.",
        ],
    }

    log = [f"Schema validation, {started}: {summary['result']}",
           f"Model raw validity: {summary['model_raw_validity']}",
           f"Blocked by a later rule: {summary['blocked_by_a_later_rule']}", "",
           f"  {'source':<70}{'boundary':<28}{'n':>5}{'pass':>6}{'fail':>6}"]
    for (src, boundary), t in sorted(table.items()):
        if boundary == REPLY:
            log.append(f"  {src:<70}{boundary:<28}{t['n']:>5}   valid {t['valid']}, parse failures "
                       f"{t['parse_failure']}, no reply {t['no_reply']}")
        else:
            log.append(f"  {src:<70}{boundary:<28}{t['n']:>5}{t['passed']:>6}{t['failed']:>6}")
    log += ["", "Failures:"] + ([f"  {f['source']} | {f['boundary']} | {f['id']}: {f['error']}"
                                  for f in summary["failures"]] or ["  none"])
    log += ["", "Blocked by a later rule (11 Oct lines today's post-check blocks):"]
    log += [f"  {r['source']} | {r['id']}: {r['rule']}" for r in resolved]
    log += ["", "Not applicable:"] + [f"  {k}: {v}" for k, v in summary["not_applicable"].items()]

    out.parent.mkdir(exist_ok=True)

    def write(path, text):
        with open(path, "x", encoding="utf-8", newline="\n") as f:   # never overwrite a recorded run
            f.write(text)

    write(outputs[0], "".join(json.dumps({"source": s, "boundary": b, "id": i, "result": r}) + "\n"
                              for s, b, i, r in checks))
    write(outputs[1], json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write(outputs[2], "\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()

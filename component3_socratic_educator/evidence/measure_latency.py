"""
Per-state latency against the 3-second target (plan, Tue 13 Oct).

Runs full FSM sessions against the live model: every risk category, each with
scripted children, on the production INTERCEPT prompt (no upstream text). For
each line it records how long the child waited (src/latency.py), and for
INTERCEPT how much of that was the model's HTTP call. One warm-up INTERCEPT
call is made first and reported apart from the sessions.

It also times a bare TCP connect to every address the server's host name
resolves to, because a connect that is refused before falling back to the next
address costs time on every call.

Needs LM Studio running with the model loaded; refuses to record a run in which
the model was unreachable. Run from component3_socratic_educator/:
    python -m evidence.measure_latency [--sessions-per-category 5] [--tag after-fix]

Writes, in the run folder evidence/system/latency_<date>[_<tag>]/, a .jsonl (one row
per session), a .summary.json and a .log.txt; never overwrites a recorded run.
"""
import argparse
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from evidence.run_info import lm_studio_version
from src import config, latency, model_client
from src.fsm_controller import FSMController
from src.grammar_decoder import INTERCEPT_PROMPT, INTERCEPT_SCHEMA, GrammarConstrainedGenerator
from src.schemas import RiskCategory, TriggerPayload

OUT_DIR = Path(__file__).resolve().parent / "system"   # new runs go in their topic folder (evidence/README.md)
TARGET_SECONDS = 3.0
LMS = Path.home() / ".lmstudio" / "bin" / "lms"
# A reply under 4 words is not "complete", so INQUIRE asks again (up to 3 times).
CHILDREN = {
    "reflective": ["I was just watching videos", "I guess I should close it and tell my mum next time"],
    "defensive": ["It wasn't me", "whatever", "no big deal", "It was nothing, my friend sent it"],
    "distressed": ["It was really scary", "I feel sick", "so scared", "I can't stop crying about it"],
    "withdrawn": ["idk", "nothing", "leave me alone", "I don't want to talk"],
}


def _run(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip() or "not recorded"
    except (OSError, subprocess.SubprocessError):
        return "not recorded"


def _hardware() -> dict:
    def cim(cls, prop):
        return _run(["powershell", "-NoProfile", "-Command",
                     f"(Get-CimInstance {cls} | ForEach-Object {{ $_.{prop} }}) -join '; '"])
    ram = cim("Win32_ComputerSystem", "TotalPhysicalMemory")
    return {"os": platform.platform(), "cpu": cim("Win32_Processor", "Name"),
            "ram_gb": round(int(ram) / 2**30, 1) if ram.isdigit() else "not recorded",
            "gpu": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
            "lm_studio_engines": _run([str(LMS), "runtime", "ls"]),
            "loaded_model": _run([str(LMS), "ps"])}


def _connect_probe(url: str) -> list[dict]:
    """A bare TCP connect to each address the host resolves to, in the order tried."""
    host, port = urlparse(url).hostname, urlparse(url).port
    rows = []
    for family, _, _, _, address in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        s = socket.socket(family)
        s.settimeout(5)
        start = time.perf_counter()
        try:
            s.connect(address)
            outcome = "connected"
        except OSError as e:
            outcome = f"failed: {e.__class__.__name__}"
        finally:
            s.close()
        rows.append({"address": address[0], "seconds": round(time.perf_counter() - start, 3), "outcome": outcome})
    return rows


# Each category comes from the component whose contract allows it (docs/interface-contracts/).
_TEXT_CATEGORIES = {RiskCategory.HATE_SPEECH, RiskCategory.CYBERBULLYING, RiskCategory.GROOMING_LANGUAGE,
                    RiskCategory.SELF_HARM_LANGUAGE}


def _trigger(category: RiskCategory) -> TriggerPayload:
    if category in _TEXT_CATEGORIES:
        return TriggerPayload.new(source_component="component2_hate_speech_detection", content_type="text_message",
                                  risk_category=category, confidence_score=0.9)
    return TriggerPayload.new(source_component="component1_screen_monitoring", content_type="image",
                              risk_category=category, confidence_score=0.9, app_or_window="browser")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-per-category", type=int, default=5)
    ap.add_argument("--tag", help="suffix for a second run on the same day, e.g. after-fix")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d")
    run_name = f"latency_{stamp}{'_' + args.tag if args.tag else ''}"
    out = OUT_DIR / run_name / f"{run_name}.jsonl"           # one folder per run
    outputs = [out, out.with_suffix(".summary.json"), out.with_suffix(".log.txt")]
    if any(p.exists() for p in outputs):
        sys.exit(f"{out.name} (or its summary/log) already exists: pass --tag. Nothing was recorded.")

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    hardware = _hardware()
    probe = _connect_probe(model_client.LM_STUDIO_URL)
    generator = GrammarConstrainedGenerator()
    controller = FSMController(generator=generator)

    start = time.perf_counter()
    warm = generator.generate_with_source("INTERCEPT", RiskCategory.EXPLICIT_VISUAL)
    warm_up = {"seconds": round(time.perf_counter() - start, 3), "source": warm.source, "reason": warm.reason}

    rows, personas = [], list(CHILDREN)
    for category in RiskCategory:
        for i in range(args.sessions_per_category):
            child = personas[i % len(personas)]
            replies = iter(CHILDREN[child])
            timed = latency.time_session(controller, _trigger(category), lambda _line, r=replies: next(r, "ok"))
            rows.append({"risk_category": category.value, "scripted_child": child,
                         "waits": [{"state": w.state, "seconds": round(w.seconds, 4), "source": w.source,
                                    "reason": w.reason} for w in timed.waits],
                         "intercept_model_call_seconds": [round(s, 4) for s in timed.model_call_seconds]})

    unreachable = sum(r["waits"][0]["reason"] == "model_unreachable" for r in rows)
    if unreachable or warm.reason == "model_unreachable":
        sys.exit(f"{unreachable}/{len(rows)} sessions (or the warm-up) got no model reply: start the LM Studio "
                 "server and rerun. Nothing was recorded.")

    def waits_of(state):
        return [w["seconds"] for r in rows for w in r["waits"] if w["state"] == state]

    intercept = waits_of("INTERCEPT")
    model_call = [sum(r["intercept_model_call_seconds"]) for r in rows]   # every call in the step
    per_state = {
        "INTERCEPT": latency.summarise(intercept, TARGET_SECONDS),
        "INTERCEPT model call (HTTP)": latency.summarise(model_call, TARGET_SECONDS),
        "INTERCEPT outside the model call": latency.summarise([w - m for w, m in zip(intercept, model_call)],
                                                              TARGET_SECONDS),
        "INQUIRE": latency.summarise(waits_of("INQUIRE"), TARGET_SECONDS),
        "EVALUATE+CONTRACT": latency.summarise(waits_of("EVALUATE+CONTRACT"), TARGET_SECONDS),
    }
    sources = {s: sum(r["waits"][0]["source"] == s for r in rows) for s in ("model", "fallback")}
    summary = {
        "run_started": started,
        "target_seconds": TARGET_SECONDS,
        "definition": ("wait = from the child's last reply (or the session start) to the next line being ready; "
                       "the child's own reply time is excluded. EVALUATE+CONTRACT is the wait for the closing "
                       "line, which covers both states. 'INTERCEPT model call (HTTP)' is the whole HTTP request, "
                       "connect included, as the client sees it."),
        "sessions": len(rows),
        "intercept_sources": sources,
        "warm_up_intercept": warm_up,
        "per_state_seconds": per_state,
        "connect_probe": probe,
        "settings": {"url": model_client.LM_STUDIO_URL, "model_id": config.LOCAL_MODEL_ID,
                     "server": "LM Studio (llama.cpp engine)", "lm_studio_version": lm_studio_version(),
                     "hardware": hardware, "response_format": "json_schema (strict)", "schema": INTERCEPT_SCHEMA,
                     "prompt": INTERCEPT_PROMPT, "max_tokens": 100, "temperature": generator.temperature,
                     "other_sampling": "server defaults (not set)",
                     "sessions_per_category": args.sessions_per_category, "child_replies": CHILDREN,
                     "inquire": "template lines (no model until after PP1)"},
        "limitations": [
            "One laptop, one model load, back-to-back sessions: other machines, a cold start or a busy GPU "
            "will differ.",
            "Child replies are scripted and instant; INQUIRE, EVALUATE and CONTRACT make no model call today, "
            "so their numbers will grow when INQUIRE moves to the model after PP1.",
            "Timed in Python with time.perf_counter around the whole step; the server's own prediction time "
            "is not read here.",
        ],
    }

    log = [f"Latency, {len(rows)} sessions ({args.sessions_per_category} per category), target {TARGET_SECONDS} s, "
           f"{config.LOCAL_MODEL_ID} at temperature {generator.temperature}, url {model_client.LM_STUDIO_URL}",
           f"GPU: {hardware['gpu']}; CPU: {hardware['cpu']}; RAM: {hardware['ram_gb']} GB",
           f"INTERCEPT sources: {sources}",
           f"Warm-up INTERCEPT (not in the table): {warm_up['seconds']} s ({warm_up['source']})",
           "", "Connect probe (each address the host resolves to, in order):"]
    log += [f"  {p['address']:<12}{p['seconds']:>7} s  {p['outcome']}" for p in probe]
    log += ["", f"  {'seconds the child waits':<34}{'n':>5}{'min':>8}{'median':>8}{'p95':>8}{'max':>8}"
                f"{'> ' + str(TARGET_SECONDS) + ' s':>9}"]
    log += [f"  {state:<34}{s['n']:>5}{s['min']:>8}{s['median']:>8}{s['p95']:>8}{s['max']:>8}{s['over_target']:>9}"
            for state, s in per_state.items() if s["n"]]
    log += ["", "Limitations:"] + [f"  - {item}" for item in summary["limitations"]]

    out.parent.mkdir(exist_ok=True)

    def write(path, text):
        with open(path, "x", encoding="utf-8", newline="\n") as f:   # never overwrite a recorded run
            f.write(text)

    write(outputs[0], "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write(outputs[1], json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write(outputs[2], "\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()

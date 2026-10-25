"""
Which sampling temperature does LM Studio actually use? (plan, Tue 13 Oct)

Streams the llama.cpp runtime log (`lms log stream --source runtime`) while it
makes two calls on the INTERCEPT prompt: the production call (the generator,
which sends config.INTERCEPT_TEMPERATURE), then the same request without a
temperature, so the server default applies (as every run before 12 Oct did).
Each call's "Sampling params" log line is kept, and its temp value read.

Needs LM Studio running with the model loaded, and the lms CLI. Run from
component3_socratic_educator/:
    python -m evidence.check_temperature

Writes, in the run folder evidence/intercept/temperature_check_<date>/, the raw log
(.runtime_log.jsonl), a .summary.json and a .log.txt; never overwrites a
recorded run.
"""
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from evidence.run_info import lm_studio_version
from src import config
from src.grammar_decoder import INTERCEPT_PROMPT, INTERCEPT_SCHEMA, GrammarConstrainedGenerator
from src.model_client import LM_STUDIO_URL, call_local_model
from src.schemas import RiskCategory

OUT_DIR = Path(__file__).resolve().parent / "intercept"   # new runs go in their topic folder (evidence/README.md)
LMS = Path.home() / ".lmstudio" / "bin" / "lms"
SAMPLER = re.compile(r"top_k = (?P<top_k>[\d.]+), top_p = (?P<top_p>[\d.]+), min_p = (?P<min_p>[\d.]+).*?"
                     r"temp = (?P<temp>[\d.]+)", re.S)
REPEAT = re.compile(r"repeat_penalty = (?P<repeat_penalty>[\d.]+)")


def main():
    stamp = datetime.now().strftime("%Y-%m-%d")
    run_name = f"temperature_check_{stamp}"
    out = OUT_DIR / run_name / f"{run_name}.runtime_log.jsonl"
    outputs = [out, OUT_DIR / run_name / f"{run_name}.summary.json", OUT_DIR / run_name / f"{run_name}.log.txt"]
    if any(p.exists() for p in outputs):
        sys.exit(f"{run_name} is already recorded; nothing was overwritten.")

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    stream = subprocess.Popen([str(LMS), "log", "stream", "--source", "runtime", "--json"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
    time.sleep(3)   # let the stream attach before the first call
    generator = GrammarConstrainedGenerator()
    production = generator.generate_with_source("INTERCEPT", RiskCategory.EXPLICIT_VISUAL)
    call_local_model(INTERCEPT_PROMPT, max_tokens=100, response_format={"type": "json_schema", "json_schema": {
        "name": "intercept_line", "strict": True, "schema": INTERCEPT_SCHEMA}})
    time.sleep(3)   # let the last log lines arrive
    stream.terminate()
    raw = stream.communicate(timeout=10)[0]

    events = [json.loads(line) for line in raw.splitlines() if line.startswith("{")]
    sampler_lines = [e["data"]["message"] for e in events if e["data"].get("message", "").startswith("Sampling params")]
    if production.reason == "model_unreachable" or len(sampler_lines) != 2:
        sys.exit(f"expected 2 sampling-parameter log lines, got {len(sampler_lines)} "
                 f"(production line: {production.source}, {production.reason}). Nothing was recorded.")

    calls = []
    for label, sent, line in zip(("production INTERCEPT call", "same request, no temperature sent"),
                                 (generator.temperature, None), sampler_lines):
        used = {k: float(v) for k, v in {**SAMPLER.search(line).groupdict(),
                                         **REPEAT.search(line).groupdict()}.items()}
        calls.append({"call": label, "temperature_sent": sent, "engine_used": used})
    engine = next(e["data"] for e in events if "engineName" in e["data"])
    summary = {
        "run_started": started,
        "calls": calls,
        "engine": f"{engine['engineName']}@{engine['engineVersion']}",
        "settings": {"url": LM_STUDIO_URL, "model_id": config.LOCAL_MODEL_ID,
                     "lm_studio_version": lm_studio_version(), "config.INTERCEPT_TEMPERATURE": config.INTERCEPT_TEMPERATURE,
                     "prompt": INTERCEPT_PROMPT, "max_tokens": 100, "response_format": "json_schema (strict)",
                     "log_source": "lms log stream --source runtime --json"},
        "limitations": ["Reads LM Studio's own debug log, whose wording can change between versions.",
                        "The server default can change with the LM Studio version or a per-model preset; "
                        "none was set on this machine (no files in ~/.lmstudio/config-presets)."],
    }
    log = [f"Sampling parameters llama.cpp used ({summary['engine']}, {config.LOCAL_MODEL_ID}):"]
    log += [f"  {c['call']:<36} sent temperature {str(c['temperature_sent']):<5} -> engine temp {c['engine_used']['temp']}, "
            f"top_k {c['engine_used']['top_k']:g}, top_p {c['engine_used']['top_p']}, min_p {c['engine_used']['min_p']}, "
            f"repeat_penalty {c['engine_used']['repeat_penalty']}" for c in calls]

    out.parent.mkdir(exist_ok=True)

    def write(path, text):
        with open(path, "x", encoding="utf-8", newline="\n") as f:   # never overwrite a recorded run
            f.write(text)

    write(outputs[0], "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events))
    write(outputs[1], json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write(outputs[2], "\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()

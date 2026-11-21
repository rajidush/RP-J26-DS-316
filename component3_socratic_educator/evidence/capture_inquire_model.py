"""
Live capture of model-written INQUIRE lines (ticket pp2-validation/05, ADR 0010).

Scripted children, one set per age band, run through the real controller in every
arm (template, control, adapted). For each INQUIRE turn it records the line the
child saw, its source (model, template or fallback), the fallback reason and what
the model wrote, also when it was blocked. Same scripts for every arm, so the move
sequence is the same and only the wording differs.

Needs LM Studio running with the INTERCEPT model (config.LOCAL_MODEL_ID) and both
INQUIRE models (config.INQUIRE_MODEL_IDS). Run from component3_socratic_educator/:
    python -m evidence.capture_inquire_model [--tag <name>]

Writes evidence/inquire/inquire_model_capture_<date>[_<tag>]/: sessions.jsonl and run_record.json.
Never overwrites a recorded run. The adapter it calls was trained on the PP1 prompt,
not this one (it is retrained on PP2 rows in ticket 20), so this run shows the path
works and how often each check fires, not the adapted model's PP2 quality.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from component3_socratic_educator.evidence.capture_inquire_served import model_file, sha256  # noqa: E402
from component3_socratic_educator.evidence.run_info import lm_studio_version                # noqa: E402
from component3_socratic_educator.src import config                                          # noqa: E402
from component3_socratic_educator.src import grammar_decoder as gd                           # noqa: E402
from component3_socratic_educator.src.fsm_controller import DialogueState, FSMController     # noqa: E402
from component3_socratic_educator.src.model_client import LM_STUDIO_URL                      # noqa: E402
from component3_socratic_educator.src.schemas import RiskCategory, TriggerPayload            # noqa: E402

# Replies in order: the first answers the opening line, the rest answer INQUIRE's moves.
SCRIPTS = {
    "explains": ["I was just watching videos when it popped up.", "I clicked a link my friend sent me.",
                 "Maybe more of it would come up.", "I could close the app."],
    "dont_know": ["I don't know.", "I don't know.", "I don't know.", "I don't know."],
    "states_feeling": ["Honestly it made me feel scared.", "I was playing a game and it came up.",
                       "I felt scared and a bit sick.", "I might tell my dad."],
}
CATEGORY = RiskCategory.VIOLENCE   # never shown to the model (ADR 0004); fixed so runs compare


@dataclass
class CapturingGenerator(gd.GrammarConstrainedGenerator):
    """Keeps every INQUIRE line with what the model wrote (the transcript keeps only the delivered text)."""
    inquire_lines: list = field(default_factory=list)

    def generate_with_source(self, state, risk_category, *args, **kwargs):
        line = super().generate_with_source(state, risk_category, *args, **kwargs)
        if state == "INQUIRE":
            self.inquire_lines.append(line)
        return line


def run(arm, band, script):
    replies = iter(SCRIPTS[script])
    gen = CapturingGenerator(inquire_arm=arm)
    trigger = TriggerPayload.new("component1_screen_monitoring", "image", CATEGORY, 0.6)
    started = time.perf_counter()
    record, transcript = FSMController(generator=gen).run(trigger, lambda _line: next(replies), age_band=band)
    seconds = round(time.perf_counter() - started, 2)
    turns = transcript.turns
    inquire = [(i, t) for i, t in enumerate(turns) if t.state == DialogueState.INQUIRE]
    return {"arm": arm, "age_band": band, "script": script, "seconds": seconds,
            "emotional_state": record.emotional_state.value,
            "inquire": [{"move": t.move, "easier": t.easier, "answering": turns[i - 1].child_response,
                         "line": t.system_text, "source": t.source, "reason": t.fallback_reason,
                         "model_output": line.model_output}
                        for (i, t), line in zip(inquire, gen.inquire_lines)]}


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model(model_id):
    path = model_file(model_id)
    return {"model_id": model_id, "file": path.name if path else "not recorded",
            "sha256": sha256(path) if path else "not recorded"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", help="for a second run on the same day")
    args = ap.parse_args(argv)
    name = f"inquire_model_capture_{datetime.now().strftime('%Y-%m-%d')}" + (f"_{args.tag}" if args.tag else "")
    out = ROOT / "evidence" / "inquire" / name
    if out.exists():
        raise SystemExit(f"{out} already exists; recorded runs are never overwritten.")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    sessions = [run(arm, band, script) for arm in config.INQUIRE_ARMS for band in config.AGE_BANDS
                for script in SCRIPTS]

    sources = {arm: Counter(t["source"] for s in sessions if s["arm"] == arm for t in s["inquire"])
               for arm in config.INQUIRE_ARMS}
    reasons = {arm: Counter(t["reason"] for s in sessions if s["arm"] == arm for t in s["inquire"] if t["reason"])
               for arm in config.INQUIRE_ARMS}
    record = {
        "started_at": started, "runtime_seconds": round(time.perf_counter() - t0, 1),
        "scripts": SCRIPTS, "risk_category": CATEGORY.value, "age_bands": list(config.AGE_BANDS),
        "models": {"intercept": _model(config.LOCAL_MODEL_ID),
                   **{arm: _model(model_id) for arm, model_id in config.INQUIRE_MODEL_IDS.items()}},
        "seed": "none: INQUIRE decodes greedily (temperature 0); INTERCEPT samples at "
                f"{config.INTERCEPT_TEMPERATURE} with no seed set, so its lines vary between runs",
        "generation_settings": {"temperature": config.INQUIRE_TEMPERATURE, "max_tokens": gd.INQUIRE_MAX_TOKENS,
                                "grammar": gd.INQUIRE_SCHEMA, "other": "server defaults (top_p, top_k, repeat penalty)"},
        "prompt": {"template": gd.INQUIRE_PROMPT, "guide": gd.INQUIRE_GUIDE, "easier": gd.INQUIRE_EASIER,
                   "fence": gd.CONVERSATION_FENCE,
                   "sha256": _sha(json.dumps([gd.INQUIRE_PROMPT, gd.INQUIRE_GUIDE, gd.INQUIRE_EASIER,
                                              gd.CONVERSATION_FENCE], sort_keys=True))},
        "runtime": {"server": LM_STUDIO_URL, "lm_studio": lm_studio_version(), "python": sys.version.split()[0],
                    "machine": platform.platform(), "gpu": "not recorded"},
        "sources_per_arm": {arm: dict(c) for arm, c in sources.items()},
        "fallback_reasons_per_arm": {arm: dict(c) for arm, c in reasons.items()},
        "limitation": "the adapted model was trained on the PP1 prompt; it is retrained on PP2 rows in ticket 20",
    }
    out.mkdir(parents=True)
    (out / "sessions.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in sessions),
                                        encoding="utf-8", newline="\n")
    (out / "run_record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8", newline="\n")
    for arm in config.INQUIRE_ARMS:
        print(f"{arm:<9} sources {dict(sources[arm])}  fallback reasons {dict(reasons[arm])}")
    print("wrote", out)


if __name__ == "__main__":
    main()

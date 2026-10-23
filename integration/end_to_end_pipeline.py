"""
End-to-end integration pipeline.

This is what "the whole system running on a consumer laptop" actually is:
one Python process that imports each component's real module (or, right
now, its stub) and calls them in sequence. No servers, no containers, no
network calls between components -- everything is in-process function
calls passing dicts that match the shared JSON Schemas.

Run from the project root:
    python integration/end_to_end_pipeline.py
    python integration/end_to_end_pipeline.py --source component2
    python integration/end_to_end_pipeline.py --source component2 --text "nobody likes you, just leave the group" --age 10
    python integration/end_to_end_pipeline.py --offline     # no model: INTERCEPT uses its fallback line
    python integration/end_to_end_pipeline.py --offline --auto   # PP1: scripted replies, no input()

As each component owner finishes their real implementation, they replace
their stub import below with their real module -- nobody else's code
needs to change, because everyone was built against the same schemas.
"""
import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# One sys.path entry is all that's needed: every component is imported by
# its unique top-level package name (component1_screen_monitoring, etc.),
# never by the generic "src" name alone -- that's what avoids the module
# name collisions you'd get if two components' "src" packages were both
# added to sys.path directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from component1_screen_monitoring.src.component1 import simulate_screen_capture as comp1_capture  # noqa: E402
from component2_hate_speech_detection.src.component2 import analyze_text as comp2_analyze          # noqa: E402
from component4_profiling_xai.src.component4 import generate_report as comp4_report                # noqa: E402
from component3_socratic_educator.src.fsm_controller import DialogueState, FSMController           # noqa: E402
from component3_socratic_educator.src.emotion_rules import OTHER                                   # noqa: E402
from component3_socratic_educator.src.grammar_decoder import GrammarConstrainedGenerator           # noqa: E402
from component3_socratic_educator.src.schemas import RiskCategory, TriggerPayload                   # noqa: E402
from integration.comp3_to_comp4 import DEVICE_CHILD_ID, to_emotional_pattern_row                   # noqa: E402


# Scripted child replies for --auto / PP1 demos (no interactive input).
# Short early replies keep INQUIRE cycling; the last line completes the contract.
AUTO_SCRIPT = ["Hmm", "idk", "not sure", "I should've closed it instead, next time I will"]


def cli_respond(prompt: str) -> str:
    print(f"\n[Socratic Educator]: {prompt}")
    return input("[Child]: ")


def auto_respond(script=None):
    """Non-interactive child replies for demos and CI (no input())."""
    replies = iter(script or AUTO_SCRIPT)

    def respond(prompt: str) -> str:
        try:
            reply = next(replies)
        except StopIteration:
            reply = "ok"
        print(f"\n[Socratic Educator]: {prompt}")
        print(f"[Child]: {reply}")
        return reply

    return respond


def offline_model(*_args, **_kwargs):
    """A model call that always fails, so INTERCEPT takes its fallback line."""
    raise ConnectionError("offline run: no model called")


def build_trigger_from(source: str, text: Optional[str] = None, age: int = 10) -> Optional[TriggerPayload]:
    """C1's simulated capture, C2's mock fixture, or -- given `text` -- C2's real engine for a
    child of `age`. None means C2 checked the text and found nothing that should wake C3."""
    if source == "component1":
        raw = comp1_capture()
    else:
        raw = comp2_analyze(text, age=age) if text is not None else comp2_analyze()
        if raw is None:
            return None
    raw["risk_category"] = RiskCategory(raw["risk_category"])
    return TriggerPayload(**raw)


@dataclass
class RecordingGenerator(GrammarConstrainedGenerator):
    """The FSM only takes text from the generator; this keeps where each line
    came from (model, template or fallback, with the reason) for the run log."""
    lines: list = field(default_factory=list)

    def generate(self, state, risk_category, attempt=0, reply_type=OTHER, content_type=None, **_kwargs):
        line = self.generate_with_source(
            state, risk_category, attempt, reply_type=reply_type, content_type=content_type
        )
        self.lines.append((state, line))
        return line.text


C4_MODELS = ROOT / "component4_profiling_xai" / "models"


def try_emotional_pattern(component4_row: dict) -> Optional[dict]:
    """Run C4 Function 1 when trained artifacts exist; otherwise return None (stub-only path)."""
    if not (C4_MODELS / "emotional_pattern_kmeans.pkl").exists():
        return None
    try:
        import pandas as pd
        from component4_profiling_xai.src.emotional_pattern import run_emotional_pattern

        patterns = run_emotional_pattern(pd.DataFrame([component4_row]), artifact_dir=C4_MODELS)
        row = patterns.reset_index().iloc[0]
        return {
            "child_id": str(row.get("child_id", component4_row.get("child_id"))),
            "sessions": int(row["sessions"]),
            "cluster": int(row["cluster"]),
            "pattern": str(row["pattern"]),
        }
    except Exception as exc:  # noqa: BLE001 — demo must not crash if artifacts mismatch sklearn
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_session(trigger: TriggerPayload, respond, model_call=None, child_id: str = DEVICE_CHILD_ID) -> dict:
    """One session through the whole chain: trigger -> FSM (constrained INTERCEPT,
    template INQUIRE) -> schema-validated record -> Component 4 input row and report.
    model_call=None uses the live model; pass offline_model for a run without one."""
    generator = RecordingGenerator(model_call=model_call)
    evaluate_output, transcript = FSMController(generator=generator).run(trigger, respond)
    record = evaluate_output.to_dict()

    lines = iter(generator.lines)   # one per INTERCEPT / INQUIRE turn, in order
    turns = []
    for t in transcript.turns:
        line = None
        if t.state in (DialogueState.INTERCEPT, DialogueState.INQUIRE):
            state, line = next(lines)
            if state != t.state.name:   # fail loudly rather than label a turn with another line's source
                raise RuntimeError(f"generator line for {state} paired with a {t.state.name} turn")
        turns.append({"state": t.state.name, "system_text": t.system_text, "child_response": t.child_response,
                      "source": line.source if line else None, "reason": line.reason if line else None,
                      "model_output": line.model_output if line else None})
    component4_row = to_emotional_pattern_row(evaluate_output, child_id)
    return {"trigger": trigger.to_dict(), "turns": turns, "record": record,
            "component4_row": component4_row,
            "emotional_pattern": try_emotional_pattern(component4_row),
            "report": comp4_report(record)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full 4-component pipeline once, end to end.")
    parser.add_argument("--source", choices=["component1", "component2"], default="component1",
                         help="Which upstream detector triggers this session")
    parser.add_argument("--offline", action="store_true", help="Run without the model (fallback INTERCEPT line)")
    parser.add_argument("--auto", action="store_true",
                         help="Scripted child replies (no input) — use for PP1 / non-interactive demos")
    parser.add_argument("--with-model", action="store_true",
                         help="Allow live LM Studio even when --auto is set (default: --auto is offline)")
    parser.add_argument("--text", help="Component 2 only: a message to check with the real C2 engine "
                                       "(without it, C2's mock fixture is used)")
    parser.add_argument("--age", type=int, default=10, help="Component 2 only: the child's age, 8-15")
    args = parser.parse_args()

    # PP1-safe defaults: --auto implies offline unless --with-model; keep C2 on lexicon
    # so --text never tries to download HuggingFace head weights during a demo.
    use_offline = args.offline or (args.auto and not args.with_model)
    if use_offline or args.text is not None:
        os.environ.setdefault("C2_USE_HEADS", "0")

    print(f"--- Step 1/4: {args.source} detects a risk and emits a trigger ---")
    try:
        trigger = build_trigger_from(args.source, text=args.text, age=args.age)
    except Exception as exc:  # noqa: BLE001 — surface failure clearly for demos
        print(f"FAILED at trigger build: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if trigger is None:
        print("Component 2 checked the text and found nothing that should start a conversation. No session.")
        return 0
    print(f"Trigger: risk_category={trigger.risk_category.value}, confidence={trigger.confidence_score}, "
          f"content_type={trigger.content_type}")

    print("\n--- Step 2/4: Component 3 runs the Socratic dialogue (FSM + grammar-constrained decoding) ---")
    respond = auto_respond() if args.auto else cli_respond
    try:
        log = run_session(trigger, respond=respond, model_call=offline_model if use_offline else None)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED during C3/C4 session: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    intercept = log["turns"][0]
    print(f"\n(INTERCEPT line source: {intercept['source']}"
          f"{', ' + intercept['reason'] if intercept['reason'] else ''})")
    print(f"(INTERCEPT text: {intercept['system_text']})")

    # Integration guard: text/voice triggers must not use screen-worded INTERCEPT lines.
    if trigger.content_type in ("text_message", "voice_transcript") and "screen" in intercept["system_text"].lower():
        print("FAILED: C2 text trigger produced a screen-worded INTERCEPT line", file=sys.stderr)
        print(f"  line: {intercept['system_text']}", file=sys.stderr)
        return 1

    print("\n--- Step 3/4: Component 3 hands a validated structured record to Component 4 ---")
    print(f"EvaluateOutput: {log['record']}")
    print(f"Component 4 input row: {log['component4_row']}")
    if log.get("emotional_pattern"):
        print(f"Component 4 emotional pattern: {log['emotional_pattern']}")
    else:
        print("Component 4 emotional pattern: (artifacts unavailable — parent report only)")

    print("\n--- Step 4/4: Component 4 generates the parent-facing report ---")
    print("\n" + log["report"])

    if "Parent-Facing Summary" not in log["report"]:
        print("FAILED: Component 4 report missing Parent-Facing Summary", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
    python integration/end_to_end_pipeline.py --offline     # no model: INTERCEPT uses its fallback line

As each component owner finishes their real implementation, they replace
their stub import below with their real module -- nobody else's code
needs to change, because everyone was built against the same schemas.
"""
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

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
from component3_socratic_educator.src.grammar_decoder import GrammarConstrainedGenerator           # noqa: E402
from component3_socratic_educator.src.schemas import RiskCategory, TriggerPayload                   # noqa: E402
from integration.comp3_to_comp4 import DEVICE_CHILD_ID, to_emotional_pattern_row                   # noqa: E402


def cli_respond(prompt: str) -> str:
    print(f"\n[Socratic Educator]: {prompt}")
    return input("[Child]: ")


def offline_model(*_args, **_kwargs):
    """A model call that always fails, so INTERCEPT takes its fallback line."""
    raise ConnectionError("offline run: no model called")


def build_trigger_from(source: str) -> TriggerPayload:
    raw = comp1_capture() if source == "component1" else comp2_analyze()
    raw["risk_category"] = RiskCategory(raw["risk_category"])
    return TriggerPayload(**raw)


@dataclass
class RecordingGenerator(GrammarConstrainedGenerator):
    """The FSM only takes text from the generator; this keeps where each line
    came from (model, template or fallback, with the reason) for the run log."""
    lines: list = field(default_factory=list)

    def generate(self, state, risk_category, attempt=0, reply_type="other"):
        line = self.generate_with_source(state, risk_category, attempt, reply_type=reply_type)
        self.lines.append((state, line))
        return line.text


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
    return {"trigger": trigger.to_dict(), "turns": turns, "record": record,
            "component4_row": to_emotional_pattern_row(evaluate_output, child_id),
            "report": comp4_report(record)}


def main():
    parser = argparse.ArgumentParser(description="Run the full 4-component pipeline once, end to end.")
    parser.add_argument("--source", choices=["component1", "component2"], default="component1",
                         help="Which upstream detector triggers this session")
    parser.add_argument("--offline", action="store_true", help="Run without the model (fallback INTERCEPT line)")
    args = parser.parse_args()

    print(f"--- Step 1/4: {args.source} detects a risk and emits a trigger ---")
    trigger = build_trigger_from(args.source)
    print(f"Trigger: risk_category={trigger.risk_category.value}, confidence={trigger.confidence_score}")

    print("\n--- Step 2/4: Component 3 runs the Socratic dialogue (FSM + grammar-constrained decoding) ---")
    log = run_session(trigger, respond=cli_respond, model_call=offline_model if args.offline else None)
    intercept = log["turns"][0]
    print(f"\n(INTERCEPT line source: {intercept['source']}"
          f"{', ' + intercept['reason'] if intercept['reason'] else ''})")

    print("\n--- Step 3/4: Component 3 hands a validated structured record to Component 4 ---")
    print(f"EvaluateOutput: {log['record']}")
    print(f"Component 4 input row: {log['component4_row']}")

    print("\n--- Step 4/4: Component 4 generates the parent-facing report ---")
    print("\n" + log["report"])


if __name__ == "__main__":
    main()

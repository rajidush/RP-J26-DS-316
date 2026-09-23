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

As each component owner finishes their real implementation, they replace
their stub import below with their real module -- nobody else's code
needs to change, because everyone was built against the same schemas.
"""
import argparse
import sys
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
from component3_socratic_educator.src.fsm_controller import FSMController                          # noqa: E402
from component3_socratic_educator.src.schemas import RiskCategory, TriggerPayload                   # noqa: E402


def cli_respond(prompt: str) -> str:
    print(f"\n[Socratic Educator]: {prompt}")
    return input("[Child]: ")


def build_trigger_from(source: str) -> TriggerPayload:
    raw = comp1_capture() if source == "component1" else comp2_analyze()
    raw["risk_category"] = RiskCategory(raw["risk_category"])
    return TriggerPayload(**raw)


def main():
    parser = argparse.ArgumentParser(description="Run the full 4-component pipeline once, end to end.")
    parser.add_argument("--source", choices=["component1", "component2"], default="component1",
                         help="Which upstream detector triggers this session")
    args = parser.parse_args()

    print(f"--- Step 1/4: {args.source} detects a risk and emits a trigger ---")
    trigger = build_trigger_from(args.source)
    print(f"Trigger: risk_category={trigger.risk_category.value}, confidence={trigger.confidence_score}")

    print("\n--- Step 2/4: Component 3 runs the Socratic dialogue (FSM + grammar-constrained decoding) ---")
    controller = FSMController()
    evaluate_output, transcript = controller.run(trigger, respond=cli_respond)

    print("\n--- Step 3/4: Component 3 hands a validated structured record to Component 4 ---")
    print(f"EvaluateOutput: {evaluate_output.to_dict()}")

    print("\n--- Step 4/4: Component 4 generates the parent-facing report ---")
    report = comp4_report(evaluate_output.to_dict())
    print("\n" + report)


if __name__ == "__main__":
    main()

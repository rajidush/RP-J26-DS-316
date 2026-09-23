"""
Standalone demo for Component 3 only -- useful while you're building,
before integration/end_to_end_pipeline.py exists or works.

Run from the component3_socratic_educator/ folder:
    python -m src.demo
"""
from pathlib import Path

from .fsm_controller import FSMController
from .schemas import load_mock_trigger

MOCK_INPUTS = Path(__file__).resolve().parents[1] / "mock_inputs"


def cli_respond(prompt: str) -> str:
    print(f"\n[Socratic Educator]: {prompt}")
    return input("[You, playing the child]: ")


def main():
    trigger_file = MOCK_INPUTS / "trigger_from_component1.json"
    trigger = load_mock_trigger(trigger_file)
    print(f"Loaded trigger: {trigger.risk_category.value} (confidence={trigger.confidence_score})\n")

    controller = FSMController()
    result, transcript = controller.run(trigger, respond=cli_respond)

    print("\n--- Session complete ---")
    print(f"risk_level:            {result.risk_level.value}")
    print(f"emotional_state:       {result.emotional_state.value}")
    print(f"self_regulation_shown: {result.self_regulation_shown}")
    print(f"escalation_flag:       {result.escalation_flag}")
    print(f"dialogue_turns:        {result.dialogue_turns}")
    print(f"summary:               {result.dialogue_summary}")


if __name__ == "__main__":
    main()

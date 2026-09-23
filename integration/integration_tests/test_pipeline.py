"""
Automated version of the integration pipeline -- same chain as
end_to_end_pipeline.py, but with scripted child responses instead of
interactive input(), so it can run unattended (CI, or just `pytest`).

This is what you'd actually demo at PP1 if you want a repeatable run
rather than live typing: it proves the same thing, deterministically.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from component1_screen_monitoring.src.component1 import simulate_screen_capture     # noqa: E402
from component4_profiling_xai.src.component4 import generate_report                 # noqa: E402
from component3_socratic_educator.src.fsm_controller import FSMController           # noqa: E402
from component3_socratic_educator.src.schemas import RiskCategory, TriggerPayload   # noqa: E402


def test_full_pipeline_runs_end_to_end_with_scripted_responses():
    raw = simulate_screen_capture()
    raw["risk_category"] = RiskCategory(raw["risk_category"])
    trigger = TriggerPayload(**raw)

    scripted = iter(["I'm not sure", "I guess I was curious", "I should've closed it instead, next time I will"])
    controller = FSMController()
    evaluate_output, transcript = controller.run(trigger, respond=lambda _: next(scripted))

    # Component 3 -> Component 4 handoff
    report = generate_report(evaluate_output.to_dict())

    assert "Parent-Facing Summary" in report
    assert evaluate_output.session_id in report
    assert evaluate_output.dialogue_turns >= 1

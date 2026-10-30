"""
The architecture diagrams (docs/architecture/) must match what is built
(plan, Sat 10 Oct). These tests read the names and numbers from the code, so
a change in the code that the diagrams miss fails here.
Run with: pytest component3_socratic_educator/tests -v
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from component3_socratic_educator.src import config                                 # noqa: E402
from component3_socratic_educator.src.adversarial import POST_CHECK_REASONS, SCHEMA_REASONS  # noqa: E402
from component3_socratic_educator.src.fsm_controller import DialogueState            # noqa: E402

ARCH = ROOT / "docs" / "architecture"
FSM = ARCH / "fsm-state-diagram.md"
GENERATION = ARCH / "intercept-generation-path.md"
DATA_FLOW = ARCH / "system-data-flow.md"


def _text(path):
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", [FSM, GENERATION, DATA_FLOW], ids=lambda p: p.name)
def test_each_diagram_is_a_mermaid_block_github_will_render(path):
    text = _text(path)
    assert "\N{ZERO WIDTH SPACE}" not in text         # one before ``` stops GitHub rendering the diagram
    assert re.search(r"^```mermaid$", text, re.M)
    assert text.count("```") % 2 == 0


def test_the_fsm_diagram_has_every_dialogue_state_and_the_config_values():
    text = _text(FSM)
    for state in DialogueState:
        if state is not DialogueState.DONE:
            assert state.name.capitalize() in text, state
    assert " → ".join(config.SOCRATIC_MOVES) in text
    assert f"questions < {config.MAX_INQUIRE_MOVES}" in text
    assert f"`MAX_INQUIRE_MOVES = {config.MAX_INQUIRE_MOVES}`" in text
    assert f"`AGE_BANDS = {config.AGE_BANDS!r}`".replace("'", '"') in text
    levels = ", ".join(f'"{level}"' for level in sorted(config.ESCALATION_RISK_LEVELS))
    assert f"`ESCALATION_RISK_LEVELS` (`{{{levels}}}`)" in text


def test_the_fsm_diagram_has_the_risk_level_thresholds_the_code_uses():
    source = _text(ROOT / "component3_socratic_educator" / "src" / "fsm_controller.py")
    for threshold in re.findall(r"confidence_score >= ([0-9.]+)", source):
        assert f"≥ {threshold}" in _text(FSM), threshold


def test_the_generation_path_names_every_reason_a_model_line_can_fall_back():
    text = _text(GENERATION)
    for reason in ("model_unreachable", *SCHEMA_REASONS, *POST_CHECK_REASONS):
        assert reason in text, reason


def test_the_data_flow_names_each_schema_and_function_on_the_path():
    # Names only: it catches a renamed function or schema, not a wrong arrow.
    text = _text(DATA_FLOW)
    for name in ("comp1_to_comp3", "comp2_to_comp3", "comp3_to_comp4", "to_emotional_pattern_row",
                 "child_id", "run_emotional_pattern", "generate_report"):
        assert name in text, name

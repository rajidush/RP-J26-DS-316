"""
Component 4 (Behavioral Profiling / XAI Reporting) -- STUB owned by IT23135116.

Consumes the EvaluateOutput-shaped dict produced by Component 3 and turns
it into a parent-facing summary. This stub just formats it; the real
version would accumulate these over time into a profile/dashboard and
add an explainability layer over *why* a session was flagged.
"""
from __future__ import annotations


def generate_report(evaluate_output: dict) -> str:
    lines = [
        "=== Parent-Facing Summary ===",
        f"Session:          {evaluate_output['session_id']}",
        f"Risk level:       {evaluate_output['risk_level']}",
        f"Emotional state:  {evaluate_output['emotional_state']}",
        f"Self-regulation shown: {evaluate_output['self_regulation_shown']}",
        f"Escalated to you: {evaluate_output['escalation_flag']}",
        f"Conversation length: {evaluate_output['dialogue_turns']} turns",
    ]
    if evaluate_output.get("dialogue_summary"):
        lines.append(f"Summary: {evaluate_output['dialogue_summary']}")
    return "\n".join(lines)

"""
Component 4 (Behavioral Profiling / XAI Reporting) -- STUB owned by IT23135116.

Consumes the EvaluateOutput-shaped dict produced by Component 3 and turns
it into a parent-facing summary. This stub just formats it; the real
version would accumulate these over time into a profile/dashboard and
add an explainability layer over *why* a session was flagged.

Function 2 (behavioral anomaly detection) is exposed here through
detect_behavioral_anomaly(); the model logic lives in behavioral_anomaly.py.
"""
from __future__ import annotations

from .behavioral_anomaly import predict_behavioral_anomaly


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


def detect_behavioral_anomaly(behavioral_data: dict) -> dict:
    """Function 2: run behavioral anomaly detection on one behavioral record.

    Delegates to behavioral_anomaly.predict_behavioral_anomaly and returns
    its result unchanged: status, anomaly_flag, decision_score and
    anomaly_score (-decision_score; higher = more anomalous, a relative
    value only -- not a probability, percentage, confidence or clinical
    risk score).
    Raises TypeError / ValueError for invalid input, as that function does.
    """
    return predict_behavioral_anomaly(behavioral_data)

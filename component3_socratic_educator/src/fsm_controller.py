"""
Function 1: FSM Controller.

Deterministic 4-state dialogue engine: INTERCEPT -> INQUIRE -> EVALUATE -> CONTRACT.
This is the piece the panel can see as input -> processing -> output and
you can explain end to end. It is deliberately decoupled from *how* text
gets generated (see grammar_decoder.py) -- the FSM only cares about state
transitions and the structured record it produces at the end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from . import config
from .grammar_decoder import GrammarConstrainedGenerator
from .schemas import EmotionalState, EvaluateOutput, RiskLevel, TriggerPayload


class DialogueState(Enum):
    INTERCEPT = auto()
    INQUIRE = auto()
    EVALUATE = auto()
    CONTRACT = auto()
    DONE = auto()


@dataclass
class Turn:
    state: DialogueState
    system_text: str
    child_response: Optional[str] = None


@dataclass
class SessionTranscript:
    trigger: TriggerPayload
    turns: list[Turn] = field(default_factory=list)

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)


# A child-response provider is injected so this module never depends on
# a live input source. In the CLI demo (integration/end_to_end_pipeline.py)
# this is `input()`; in tests it's a canned list of strings.
ChildResponder = Callable[[str], str]


class FSMController:
    """
    Function 1. Owns the state machine. Call `run(trigger, respond)` to
    execute a full session and get back a validated EvaluateOutput ready
    for Component 4.
    """

    def __init__(self, generator: Optional[GrammarConstrainedGenerator] = None):
        self.generator = generator or GrammarConstrainedGenerator()

    def run(self, trigger: TriggerPayload, respond: ChildResponder) -> tuple[EvaluateOutput, SessionTranscript]:
        trigger.validate()
        transcript = SessionTranscript(trigger=trigger)

        state = DialogueState.INTERCEPT
        last_child_text = ""
        inquire_attempts = 0

        while state != DialogueState.DONE:
            if state == DialogueState.INTERCEPT:
                system_text = self.opening_question(trigger)
                child_text = respond(system_text)
                transcript.add(Turn(state, system_text, child_text))
                last_child_text = child_text
                state = DialogueState.INQUIRE

            elif state == DialogueState.INQUIRE:
                if self._is_complete(last_child_text) or inquire_attempts >= config.MAX_INQUIRE_ATTEMPTS:
                    state = DialogueState.EVALUATE
                    continue
                system_text = self.generator.generate("INQUIRE", trigger.risk_category)
                child_text = respond(system_text)
                transcript.add(Turn(state, system_text, child_text))
                last_child_text = child_text
                inquire_attempts += 1

            elif state == DialogueState.EVALUATE:
                evaluate_output = self._evaluate(trigger, transcript, last_child_text)
                transcript.add(Turn(state, "[internal evaluation, no child-facing text]"))
                state = DialogueState.CONTRACT

            elif state == DialogueState.CONTRACT:
                evaluate_output.escalation_flag = evaluate_output.risk_level.value in config.ESCALATION_RISK_LEVELS
                transcript.add(Turn(
                    state,
                    "Thanks for talking this through with me." if not evaluate_output.escalation_flag
                    else "I'm going to let a parent know we talked about this, so they can support you.",
                ))
                state = DialogueState.DONE

        evaluate_output.dialogue_turns = len([t for t in transcript.turns if t.child_response is not None])
        evaluate_output.validate()
        return evaluate_output, transcript

    # -- helpers -------------------------------------------------------


    def opening_question(self, trigger: TriggerPayload) -> str:
        """Intercept state: open the conversation without naming the flagged content."""
        return self.generator.generate("INTERCEPT", trigger.risk_category)
    
    def _is_complete(self, child_text: str) -> bool:
        return len(child_text.split()) >= config.MIN_RESPONSE_WORDS_FOR_COMPLETENESS

    def _evaluate(self, trigger: TriggerPayload, transcript: SessionTranscript, last_child_text: str) -> EvaluateOutput:
        """
        Placeholder evaluation heuristic (word-count / keyword based) so
        the FSM is fully runnable and testable today. This is the
        function to upgrade first once real dialogue data is available --
        swap the body for a classifier over the transcript, keep the
        EvaluateOutput contract identical.
        """
        self_regulation_shown = any(
            kw in (t.child_response or "").lower()
            for t in transcript.turns
            for kw in ("instead", "next time", "should have", "i could")
        )
        risk_level = RiskLevel.HIGH if trigger.confidence_score >= 0.85 else (
            RiskLevel.MODERATE if trigger.confidence_score >= 0.5 else RiskLevel.LOW
        )
        emotional_state = EmotionalState.UNCLEAR if not last_child_text else (
            EmotionalState.DEFENSIVE if len(last_child_text.split()) <= 2 else EmotionalState.CALM
        )
        return EvaluateOutput(
            session_id=trigger.session_id,
            risk_level=risk_level,
            emotional_state=emotional_state,
            self_regulation_shown=self_regulation_shown,
            escalation_flag=False,  # set properly in CONTRACT state
            dialogue_turns=0,  # filled in by caller
            dialogue_summary=f"{len(transcript.turns)} turns; risk_category={trigger.risk_category.value}",
        )

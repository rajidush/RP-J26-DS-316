"""
Function 1: FSM Controller.

Deterministic 4-state dialogue engine: INTERCEPT -> INQUIRE -> EVALUATE -> CONTRACT.
This is the piece the panel can see as input -> processing -> output and
you can explain end to end. It is deliberately decoupled from *how* text
gets generated (see grammar_decoder.py) -- the FSM only cares about state
transitions and the structured record it produces at the end.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from . import config, emotion_rules
from .grammar_decoder import GrammarConstrainedGenerator
from .schemas import EvaluateOutput, RiskLevel, TriggerPayload


_SELF_REGULATION_PATTERN = re.compile(r"\b(instead|next time|should have|i could)\b")


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
                # A refusal to the opening line skips INQUIRE: no follow-up questions.
                state = DialogueState.EVALUATE if emotion_rules.is_refusal(child_text) else DialogueState.INQUIRE

            elif state == DialogueState.INQUIRE:
                responses = self.inquire_loop(trigger, respond, transcript)
                last_child_text = responses[-1] if responses else last_child_text
                state = DialogueState.EVALUATE

            elif state == DialogueState.EVALUATE:
                evaluate_output = self._evaluate(trigger, transcript, last_child_text)
                transcript.add(Turn(state, "[internal evaluation, no child-facing text]"))
                state = DialogueState.CONTRACT

            elif state == DialogueState.CONTRACT:
                reason = self._escalation_reason(evaluate_output)
                evaluate_output.escalation_flag = reason is not None
                if reason:   # the schema has no reason field (ADR 0005), so the summary says why
                    evaluate_output.dialogue_summary += f"; escalated: {reason}"
                transcript.add(Turn(state, self._closing_line(evaluate_output, last_child_text)))
                state = DialogueState.DONE

        evaluate_output.dialogue_turns = len([t for t in transcript.turns if t.child_response is not None])
        evaluate_output.validate()
        return evaluate_output, transcript

    # -- helpers -------------------------------------------------------


    @staticmethod
    def _escalation_reason(evaluate_output: EvaluateOutput) -> Optional[str]:
        """Why a parent should review this session, or None. Risk level first."""
        if evaluate_output.risk_level.value in config.ESCALATION_RISK_LEVELS:
            return f"risk_level={evaluate_output.risk_level.value}"
        if evaluate_output.emotional_state.value in config.ESCALATION_EMOTIONAL_STATES:
            return f"emotional_state={evaluate_output.emotional_state.value}"
        return None

    def _closing_line(self, evaluate_output: EvaluateOutput, last_child_text: str) -> str:
        if evaluate_output.escalation_flag:
            return "I'm going to let a parent know we talked about this, so they can support you."
        if emotion_rules.is_refusal(last_child_text):
            return "That's okay, you don't have to talk right now. I'm here if you want to talk later."
        return "Thanks for talking this through with me."

    def opening_question(self, trigger: TriggerPayload) -> str:
        """Intercept state: open the conversation without naming the flagged content."""
        return self.generator.generate("INTERCEPT", trigger.risk_category)
    
    def inquire_loop(self, trigger: TriggerPayload, respond, transcript: SessionTranscript) -> list[str]:
        """Inquire state: follow up until completeness or max attempts."""
        responses = []
        attempts = 0
        while attempts < config.MAX_INQUIRE_ATTEMPTS:
            # The line answers the child's latest reply (14 Oct: it used to ignore it).
            latest = next((t.child_response for t in reversed(transcript.turns) if t.child_response is not None), "")
            prompt = self.generator.generate("INQUIRE", trigger.risk_category, attempt=attempts,
                                             reply_type=emotion_rules.reply_type(latest))
            answer = respond(prompt)
            transcript.add(Turn(DialogueState.INQUIRE, prompt, answer))
            responses.append(answer)
            attempts += 1
            if emotion_rules.is_refusal(answer) or self._is_complete(answer):
                break
        return responses
    
    def _is_complete(self, child_text: str) -> bool:
        """A complete reply: long enough, or it already states a feeling, a denial or
        a plan (14 Oct: "I feel sick" was asked again for being under 4 words)."""
        return (len(child_text.split()) >= config.MIN_RESPONSE_WORDS_FOR_COMPLETENESS
                or emotion_rules.states_something(child_text))

    def _evaluate(self, trigger: TriggerPayload, transcript: SessionTranscript, last_child_text: str) -> EvaluateOutput:
        """
        Rule-based evaluation so the FSM is fully runnable and testable
        today: risk from detector confidence, emotional_state from
        emotion_rules.classify(), self-regulation from keywords. Upgrade
        path: swap in a trained classifier over the transcript, keeping
        the EvaluateOutput contract identical.
        """
        # Whole-word match: "i could" must not fire inside "i couldn't".
        self_regulation_shown = any(
            _SELF_REGULATION_PATTERN.search((t.child_response or "").lower())
            for t in transcript.turns
        )
        risk_level = RiskLevel.HIGH if trigger.confidence_score >= 0.85 else (
            RiskLevel.MODERATE if trigger.confidence_score >= 0.5 else RiskLevel.LOW
        )
        # Rule-based classifier over every reply the child gave (src/emotion_rules.py).
        child_replies = [t.child_response for t in transcript.turns if t.child_response is not None]
        emotional_state = emotion_rules.classify(child_replies).state
        return EvaluateOutput(
            session_id=trigger.session_id,
            risk_level=risk_level,
            emotional_state=emotional_state,
            self_regulation_shown=self_regulation_shown,
            escalation_flag=False,  # set properly in CONTRACT state
            dialogue_turns=0,  # filled in by caller
            dialogue_summary=f"{len(transcript.turns)} turns; risk_category={trigger.risk_category.value}",
        )

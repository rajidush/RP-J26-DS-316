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

from . import config, emotion_rules
from .assessor import CHANGE_TALK, STANCE, RulesAssessor, SessionAssessor
from .grammar_decoder import GeneratedLine, GrammarConstrainedGenerator, resolve_age_band
from .schemas import EvaluateOutput, RiskLevel, TriggerPayload


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
    move: Optional[str] = None             # INQUIRE's Socratic move (config.SOCRATIC_MOVES)
    easier: bool = False                   # the move's easier form, asked after "I don't know"
    age_band: Optional[str] = None         # the band the session's lines were worded for
    source: Optional[str] = None           # the line's source: "model", "template" or "fallback"
    fallback_reason: Optional[str] = None  # why the model's line was not used

    @classmethod
    def of(cls, state: "DialogueState", line: GeneratedLine, child_response: Optional[str], age_band: str,
           move: Optional[str] = None, easier: bool = False) -> "Turn":
        return cls(state, line.text, child_response, move, easier, age_band, line.source, line.reason)


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

    def __init__(self, generator: Optional[GrammarConstrainedGenerator] = None,
                 assessor: Optional[SessionAssessor] = None, age_band: Optional[str] = None):
        self.generator = generator or GrammarConstrainedGenerator()
        # EVALUATE's classifier arm (src/assessor.py): the PP1 rules until the comparison picks one.
        self.assessor = assessor or RulesAssessor()
        self.age_band = age_band

    def run(self, trigger: TriggerPayload, respond: ChildResponder,
            age_band: Optional[str] = None) -> tuple[EvaluateOutput, SessionTranscript]:
        """`age_band` (one of config.AGE_BANDS) overrides the controller's; a missing
        or invalid band words every line for 8-10 (ADR 0007)."""
        trigger.validate()
        transcript = SessionTranscript(trigger=trigger)
        band = resolve_age_band(age_band if age_band is not None else self.age_band)

        state = DialogueState.INTERCEPT
        last_child_text = ""
        inquire_attempts = 0

        while state != DialogueState.DONE:
            if state == DialogueState.INTERCEPT:
                line = self.generator.generate_with_source("INTERCEPT", trigger.risk_category)
                child_text = respond(line.text)
                transcript.add(Turn.of(state, line, child_text, band))
                last_child_text = child_text
                # A refusal to the opening line skips INQUIRE: no follow-up questions.
                state = DialogueState.EVALUATE if emotion_rules.is_refusal(child_text) else DialogueState.INQUIRE

            elif state == DialogueState.INQUIRE:
                responses = self.inquire_loop(trigger, respond, transcript, band)
                last_child_text = responses[-1] if responses else last_child_text
                state = DialogueState.EVALUATE

            elif state == DialogueState.EVALUATE:
                evaluate_output = self._evaluate(trigger, transcript, last_child_text)
                transcript.add(Turn(state, "[internal evaluation, no child-facing text]", age_band=band))
                state = DialogueState.CONTRACT

            elif state == DialogueState.CONTRACT:
                reason = self._escalation_reason(evaluate_output)
                evaluate_output.escalation_flag = reason is not None
                if reason:   # the schema has no reason field (ADR 0005), so the summary says why
                    evaluate_output.dialogue_summary += f"; escalated: {reason}"
                transcript.add(Turn(state, self._closing_line(evaluate_output, last_child_text), age_band=band,
                                    source="template"))
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
    
    def inquire_loop(self, trigger: TriggerPayload, respond, transcript: SessionTranscript,
                     age_band: str = config.DEFAULT_AGE_BAND) -> list[str]:
        """Inquire state: the Socratic moves in order (clarify, consequences,
        alternatives), at most MAX_INQUIRE_MOVES questions. A refusal ends INQUIRE
        at once, and so does change talk (the assessor's stance for a safer step in
        the child's own words) once the move it answers is done. "I don't know" asks the
        same move again in its easier form, once, and the repeat counts toward the cap."""
        responses: list[str] = []
        moves = iter(config.SOCRATIC_MOVES)
        move: Optional[str] = next(moves)
        easier = False
        while move is not None and len(responses) < config.MAX_INQUIRE_MOVES:
            line = self.generator.generate_with_source("INQUIRE", trigger.risk_category, move=move,
                                                       age_band=age_band, easier=easier)
            answer = respond(line.text)
            transcript.add(Turn.of(DialogueState.INQUIRE, line, answer, age_band, move, easier))
            responses.append(answer)
            if emotion_rules.is_refusal(answer) or self._is_change_talk(transcript):
                break
            if emotion_rules.is_dont_know(answer) and not easier:
                easier = True
            else:
                move, easier = next(moves, None), False
        return responses

    def _is_change_talk(self, transcript: SessionTranscript) -> bool:
        """The child's latest reply is change talk, by the injected assessor."""
        latest = self.assessor.assess(transcript.turns).replies[-1]
        return latest.labels.get(STANCE) == CHANGE_TALK

    def _evaluate(self, trigger: TriggerPayload, transcript: SessionTranscript, last_child_text: str) -> EvaluateOutput:
        """
        Risk from detector confidence; emotional_state and self-regulation
        from the injected session assessor (src/assessor.py), so a trained
        classifier replaces the rules without changing the EvaluateOutput
        contract.
        """
        assessment = self.assessor.assess(transcript.turns)
        risk_level = RiskLevel.HIGH if trigger.confidence_score >= 0.85 else (
            RiskLevel.MODERATE if trigger.confidence_score >= 0.5 else RiskLevel.LOW
        )
        return EvaluateOutput(
            session_id=trigger.session_id,
            risk_level=risk_level,
            emotional_state=assessment.emotional_state,
            self_regulation_shown=assessment.constructed_plan,
            escalation_flag=False,  # set properly in CONTRACT state
            dialogue_turns=0,  # filled in by caller
            dialogue_summary=f"{len(transcript.turns)} turns; risk_category={trigger.risk_category.value}",
        )

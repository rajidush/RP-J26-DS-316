"""
Per-state latency (plan, Tue 13 Oct): how long the child waits for each line.

A wait runs from the moment the child's last reply comes in (or the session
starts) to the moment the next line is ready. The child's own reply time is
left out. INTERCEPT's wait also has its model call timed on its own, so the
log shows how much of the step is the HTTP call and how much is our code.
The FSM code is not changed: the clock is read in the reply callback, and the
generator's model call is wrapped for the session and restored afterwards.
"""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import grammar_decoder
from .fsm_controller import FSMController
from .schemas import EvaluateOutput, TriggerPayload


@dataclass(frozen=True)
class Wait:
    state: str                    # INTERCEPT, INQUIRE, or EVALUATE+CONTRACT (the closing line)
    seconds: float
    source: Optional[str] = None  # INTERCEPT only: "model" or "fallback"
    reason: Optional[str] = None  # INTERCEPT only: the fallback reason


@dataclass
class TimedSession:
    record: EvaluateOutput
    waits: list[Wait]
    model_call_seconds: list[float] = field(default_factory=list)   # every model call, failed ones included


def time_session(controller: FSMController, trigger: TriggerPayload, respond: Callable[[str], str],
                 clock: Callable[[], float] = time.perf_counter) -> TimedSession:
    generator = controller.generator
    saved_call = generator.model_call
    saved_generate = vars(generator).get("generate_with_source")   # an instance override, if any
    model_seconds: list[float] = []
    intercept_lines = []

    def timed_call(prompt, **options):
        call = saved_call or grammar_decoder.call_local_model   # looked up per call, as the generator does
        start = clock()
        try:
            return call(prompt, **options)
        finally:
            model_seconds.append(clock() - start)

    generate = generator.generate_with_source

    def recording_generate(state, *args, **kwargs):
        line = generate(state, *args, **kwargs)
        if state == "INTERCEPT":
            intercept_lines.append(line)
        return line

    marks = []   # (line shown, reply received) per child turn

    def timed_respond(line):
        shown = clock()
        reply = respond(line)
        marks.append((shown, clock()))
        return reply

    generator.model_call, generator.generate_with_source = timed_call, recording_generate
    try:
        start = clock()
        record, transcript = controller.run(trigger, timed_respond)
        end = clock()
    finally:
        generator.model_call = saved_call
        if saved_generate is None:
            del generator.generate_with_source   # back to the class method
        else:
            generator.generate_with_source = saved_generate

    child_turns = [t for t in transcript.turns if t.child_response is not None]
    waits, ready_from = [], start
    for turn, (shown, answered) in zip(child_turns, marks):
        waits.append(Wait(turn.state.name, shown - ready_from))
        ready_from = answered
    waits.append(Wait("EVALUATE+CONTRACT", end - ready_from))
    if intercept_lines:
        waits[0] = Wait(waits[0].state, waits[0].seconds, intercept_lines[0].source, intercept_lines[0].reason)
    return TimedSession(record, waits, model_seconds)


def summarise(seconds: list[float], target: float) -> dict:
    """min/median/p95/max in seconds to the millisecond (p95 by nearest rank) and
    how many are over the target."""
    if not seconds:
        return {"n": 0}
    ordered = sorted(seconds)
    return {"n": len(ordered), "min": round(ordered[0], 3), "median": round(statistics.median(ordered), 3),
            "p95": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 3), "max": round(ordered[-1], 3),
            "over_target": sum(s > target for s in ordered)}

"""
response.py — Function 2 (Post-Detection Response): overlay state machine,
end-to-end latency log, and the scripted mock demo.

ResponseController
------------------
Driven by Function 1's output, one call per captured frame:

    on_payload(payload, frame)   flagged frame → TriggerPayload (schema-valid)
    on_clear_frame()             frame that produced no payload (safe)
    on_dismiss()                 user pressed Dismiss / Esc (from the Tk thread)

Behaviour (thresholds in config.py):
* The action comes from response_policy.decide() — blur or block.
* Within one visible incident the action only escalates (blur → block).
* Every flagged payload EXTENDS the hold. The overlay hides only when
  ≥ MIN_HOLD_S has passed since the last flagged payload AND the last
  CLEAR_FRAMES consecutive frames were safe — no on/off flicker.
* After a dismiss, payloads from the same incident (session_id) do not
  re-show the overlay for DISMISS_GRACE_S unless confidence ≥ BLOCK_THRESHOLD.

Detection keeps running while the overlay is up when the overlay is excluded
from screen capture (capture_exclusion.py). Where it cannot be excluded,
CaptureGate applies config.CAPTURE_EXCLUSION_FALLBACK = "pause": no
classification while the overlay is up; after MIN_HOLD_S one "peek" frame is
captured with the overlay briefly hidden and decides whether to clear.

The overlay is anything with show_blur(image, caption, on_visible=None),
show_block(message, on_visible=None) and hide() — a fake works for tests.
"""
from __future__ import annotations

import json
import logging
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from .config import (BLOCK_THRESHOLD, CAPTURE_EXCLUSION_FALLBACK, CLEAR_FRAMES,
                         DISMISS_GRACE_S, E2E_LATENCY_JSON, MIN_HOLD_S)
    from .response_policy import decide, stronger
except ImportError:
    from config import (BLOCK_THRESHOLD, CAPTURE_EXCLUSION_FALLBACK, CLEAR_FRAMES,
                        DISMISS_GRACE_S, E2E_LATENCY_JSON, MIN_HOLD_S)
    from response_policy import decide, stronger

log = logging.getLogger(__name__)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="milliseconds")


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _caption(payload: dict) -> str:
    return (f"Potentially harmful content detected\n"
            f"({payload['risk_category']}, confidence {payload['confidence_score']:.2f})")


# ---------------------------------------------------------------------------
# End-to-end latency
# ---------------------------------------------------------------------------

class E2ELatencyLog:
    """
    One record per overlay appearance (incident start or escalation):
    frame captured → payload emitted → overlay visible.
    """

    def __init__(self, path: Path = E2E_LATENCY_JSON, source: str = "live",
                 strategy: str | None = None) -> None:
        self.path, self.source, self.strategy = Path(path), source, strategy
        self.records: list[dict] = []
        self._lock = threading.Lock()

    def start(self, payload: dict, action: str, captured_at: str, emitted_at: float) -> dict:
        rec = {"session_id": payload["session_id"], "action": action,
               "risk_category": payload["risk_category"],
               "confidence_score": payload["confidence_score"],
               "source": self.source, "strategy": self.strategy,
               "frame_captured_at": captured_at, "payload_emitted_at": _iso(emitted_at),
               "overlay_visible_at": None,
               "capture_to_payload_ms": round((emitted_at - _epoch(captured_at)) * 1000, 1),
               "payload_to_visible_ms": None, "capture_to_visible_ms": None}
        with self._lock:
            self.records.append(rec)
        return rec

    def visible(self, rec: dict, visible_at: float) -> None:
        with self._lock:
            rec["overlay_visible_at"] = _iso(visible_at)
            rec["payload_to_visible_ms"] = round(
                (visible_at - _epoch(rec["payload_emitted_at"])) * 1000, 1)
            rec["capture_to_visible_ms"] = round(
                (visible_at - _epoch(rec["frame_captured_at"])) * 1000, 1)

    def save(self) -> int:
        """Append this run's completed records to the JSON list on disk."""
        with self._lock:
            done = [r for r in self.records if r["overlay_visible_at"]]
        if not done:
            return 0
        existing = json.loads(self.path.read_text()) if self.path.exists() else []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(existing + done, indent=2))
        return len(done)

    def summary(self) -> str:
        ms = [r["capture_to_visible_ms"] for r in self.records if r["capture_to_visible_ms"] is not None]
        if not ms:
            return "e2e latency: no overlay shown this run"
        return (f"e2e latency (frame captured → overlay visible), {len(ms)} appearance(s): "
                f"mean {statistics.mean(ms):.0f} ms, median {statistics.median(ms):.0f} ms, "
                f"max {max(ms):.0f} ms")


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class ResponseController:
    def __init__(
        self,
        overlay,
        override: str = "auto",
        min_hold_s: float = MIN_HOLD_S,
        clear_frames: int = CLEAR_FRAMES,
        dismiss_grace_s: float = DISMISS_GRACE_S,
        block_threshold: float = BLOCK_THRESHOLD,
        latency: E2ELatencyLog | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.overlay = overlay
        self.override = override
        self.min_hold_s, self.clear_frames = min_hold_s, clear_frames
        self.dismiss_grace_s, self.block_threshold = dismiss_grace_s, block_threshold
        self.latency = latency
        self._clock, self._wall = clock, wall_clock
        self._lock = threading.Lock()

        self.action: str | None = None           # what is on screen now (None = hidden)
        self.session_id: str | None = None
        self._last_flag_at: float | None = None
        self._clear = 0
        self._dismissed_session: str | None = None
        self._dismissed_at: float | None = None

    @property
    def visible(self) -> bool:
        return self.action is not None

    # -- events --------------------------------------------------------------

    def on_payload(self, payload: dict, frame: Any = None, emitted_at: float | None = None) -> str | None:
        """A flagged frame's TriggerPayload. Returns the action now on screen (or None)."""
        emitted_at = self._wall() if emitted_at is None else emitted_at
        action = decide(payload, self.override, self.block_threshold)
        if action is None:
            return self.action                    # invalid payload — logged and ignored
        with self._lock:
            now = self._clock()
            sid = payload["session_id"]
            if self._suppressed(sid, payload["confidence_score"], now):
                log.info("dismissed by user — not re-showing incident %s (grace %.0fs)",
                         sid[:8], self.dismiss_grace_s)
                return self.action

            self._last_flag_at, self._clear = now, 0
            self.session_id = sid
            target = stronger(self.action, action)
            if target != self.action:
                self._show(target, payload, frame, emitted_at)
            elif target == "blur" and frame is not None:
                self.overlay.show_blur(frame.image, _caption(payload))    # refresh blurred view
            return self.action

    def on_clear_frame(self) -> bool:
        """A safe frame. Returns True if this frame hid the overlay."""
        with self._lock:
            if not self.visible:
                return False
            self._clear += 1
            since_flag = self._clock() - self._last_flag_at
            if self._clear >= self.clear_frames and since_flag >= self.min_hold_s:
                log.info("overlay hidden — %d clear frames, %.1fs since last flag",
                         self._clear, since_flag)
                self.overlay.hide()
                self._reset_incident()
                return True
            return False

    def on_dismiss(self) -> None:
        """User dismissed the overlay (Dismiss button / Esc)."""
        with self._lock:
            if not self.visible:
                return
            log.info("overlay dismissed by user (incident %s)", (self.session_id or "?")[:8])
            self._dismissed_session, self._dismissed_at = self.session_id, self._clock()
            self.overlay.hide()
            self._reset_incident()

    # -- capture-exclusion fallback ("pause") ---------------------------------

    def frame_policy(self, capture_excluded: bool,
                     fallback: str = CAPTURE_EXCLUSION_FALLBACK) -> str:
        """
        What to do with the next frame:
          "classify" — normal (overlay hidden, or excluded from capture, or fallback off)
          "skip"     — overlay is visible and would be in the capture; hold not over yet
          "peek"     — hold is over: capture one frame with the overlay briefly hidden
        """
        if fallback not in ("pause", "off"):
            raise ValueError(f"CAPTURE_EXCLUSION_FALLBACK must be 'pause' or 'off', got {fallback!r}")
        with self._lock:
            if not self.visible or capture_excluded or fallback == "off":
                return "classify"
            return "skip" if self._clock() - self._last_flag_at < self.min_hold_s else "peek"

    def on_peek_result(self, payload: dict | None, frame: Any = None,
                       emitted_at: float | None = None) -> bool:
        """
        Result of the single "peek" frame. Flagged → keep the overlay and extend the
        hold (escalating if needed). Safe → hide and end the incident (one clean
        frame decides, because further frames would show the overlay again).
        Returns True if the overlay was hidden.
        """
        if payload is not None:
            if decide(payload, self.override, self.block_threshold) is None:
                return False                      # invalid payload — keep overlay, re-peek later
            self.on_payload(payload, frame, emitted_at)
            return False
        with self._lock:
            if not self.visible:
                return False
            log.info("overlay hidden — peek frame was safe (fallback 'pause')")
            self.overlay.hide()
            self._reset_incident()
            return True

    # -- internals -----------------------------------------------------------

    def _suppressed(self, sid: str, confidence: float, now: float) -> bool:
        return (sid == self._dismissed_session and self._dismissed_at is not None
                and now - self._dismissed_at < self.dismiss_grace_s
                and confidence < self.block_threshold)

    def _show(self, action: str, payload: dict, frame: Any, emitted_at: float) -> None:
        rec = None
        if self.latency is not None:
            captured = getattr(frame, "timestamp", None) or payload["timestamp"]
            rec = self.latency.start(payload, action, captured, emitted_at)
        on_visible = (lambda t, r=rec: self.latency.visible(r, t)) if rec else None
        if action == "blur" and frame is not None:
            self.overlay.show_blur(frame.image, _caption(payload), on_visible=on_visible)
        else:                                     # block (or blur without an image)
            self.overlay.show_block(_caption(payload) + "\n\nThis screen has been blocked.",
                                    on_visible=on_visible)
        log.info("overlay %s → %s (%s %.2f)", "shown" if self.action is None else "escalated",
                 action, payload["risk_category"], payload["confidence_score"])
        self.action = action

    def _reset_incident(self) -> None:
        self.action, self.session_id = None, None
        self._last_flag_at, self._clear = None, 0


# ---------------------------------------------------------------------------
# Wiring the fallback into run_monitoring_loop
# ---------------------------------------------------------------------------

class CaptureGate:
    """
    Connects controller.frame_policy() to the monitoring loop:

        run_monitoring_loop(..., frame_gate=gate.allow,
                            capture_fn=gate.wrap(capture_frame_fn),
                            on_result=gate.on_result)

    Where the overlay is excluded from capture this is a pass-through.
    """

    def __init__(self, controller: ResponseController, overlay,
                 fallback: str = CAPTURE_EXCLUSION_FALLBACK) -> None:
        self.controller, self.overlay, self.fallback = controller, overlay, fallback
        self._decision = "classify"
        self.peeks = self.skipped = 0

    def _excluded(self) -> bool:
        # Before the first show nothing is visible, so the answer does not matter yet.
        return bool(getattr(self.overlay, "capture_excluded", False))

    def allow(self) -> bool:
        """frame_gate: False = skip this iteration (no capture, no classification)."""
        self._decision = self.controller.frame_policy(self._excluded(), self.fallback)
        if self._decision == "skip":
            self.skipped += 1
        return self._decision != "skip"

    def wrap(self, capture_fn: Callable[[], Any]) -> Callable[[], Any]:
        def capture():
            if self._decision != "peek":
                return capture_fn()
            self.peeks += 1
            self.overlay.hide_and_wait()          # off screen before the capture
            try:
                return capture_fn()
            finally:
                self.overlay.restore()            # back up immediately; classify while covered
        return capture

    def on_result(self, frame: Any, detection: Any, payload: dict | None) -> None:
        if self._decision == "peek":
            self.controller.on_peek_result(payload, frame)
        elif payload is not None:
            self.controller.on_payload(payload, frame)
        else:
            self.controller.on_clear_frame()


# ---------------------------------------------------------------------------
# Mock demo (no model, no violent content)
# ---------------------------------------------------------------------------

#: (risk_category, confidence) per step; None = a safe frame. One step per second.
MOCK_SCRIPT: list[tuple[str, float] | None] = [
    ("violence", 0.70),         # blur appears
    ("violence", 0.70),         # hold extended
    None, None, None, None,     # 2 clear frames reached, but hidden only once 5 s since last flag
    None, None,
    ("violence", 0.92),         # ≥ BLOCK_THRESHOLD → block
    ("explicit_visual", 0.88),  # always block; hold extended
    None, None, None, None,     # hides ≥ 5 s after the last flag
    None, None, None,
]


def mock_payload(category: str, confidence: float, session_id: str,
                 timestamp: str | None = None) -> dict:
    """A schema-valid TriggerPayload for the mock demo."""
    return {
        "session_id": session_id,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "source_component": "component1_screen_monitoring",
        "content_type": "image",
        "risk_category": category,
        "confidence_score": confidence,
        "context_metadata": {"duration_visible_ms": 0},
    }


def play_mock(controller: ResponseController, frame_fn: Callable[[], Any],
              script=MOCK_SCRIPT, step_s: float = 1.0,
              sleep_fn: Callable[[float], None] = time.sleep,
              should_stop: Callable[[], bool] = lambda: False) -> None:
    """Feed *script* through *controller* exactly like the live loop would."""
    session = None
    for i, step in enumerate(script):
        if should_stop():
            return
        frame = frame_fn()
        if step is None:
            session = None                       # a safe frame ends the incident
            hid = controller.on_clear_frame()
            log.info("[mock %2d] safe frame%s", i, " → overlay hidden" if hid else "")
        else:
            session = session or str(uuid.uuid4())
            category, conf = step
            action = controller.on_payload(mock_payload(category, conf, session, frame.timestamp), frame)
            log.info("[mock %2d] %s %.2f → %s", i, category, conf, action or "suppressed")
        sleep_fn(step_s)

"""
response.py — Trigger logic for Component 1, Function 2 (Post-Detection Response).

Responsibilities
----------------
* Consume Function 1's per-frame detection result
  (flagged / category / confidence / timestamp / frame_reference).
* Decide flag → action: show the overlay in "blur" or "block" mode when a
  frame is flagged, and hide it again afterwards.

Why the overlay is held for a fixed time
----------------------------------------
While the overlay is visible, every new screenshot captures the *overlay*,
not the content underneath. Classifying those frames would always return
"safe" and immediately hide the overlay again (on/off flicker). So once
triggered, the controller:

  1. holds the overlay for ``hold_seconds`` and skips detection meanwhile;
  2. hides it, skipping the frame captured in the same tick (it still
     contains the overlay);
  3. resumes detection — if the content is still harmful, it re-triggers.
"""
from __future__ import annotations

import time
from typing import Callable

RESPONSE_MODES = ("blur", "block")


def _caption(result: dict) -> str:
    return (f"Potentially harmful content detected\n"
            f"({result['category']}, confidence {result['confidence']:.2f})")


class ResponseController:
    """
    Turns per-frame detection results into overlay show/hide actions.

    Parameters
    ----------
    overlay : ResponseOverlay
        Anything with ``show_blur(image, caption)``, ``show_block(message)``
        and ``hide()`` — a fake works for tests.
    detector : callable
        ``detector(frame) -> dict`` — e.g. from ``capture.make_detector_callback()``.
    mode : str
        ``"blur"`` or ``"block"``.
    hold_seconds : float
        Minimum time the overlay stays up once triggered.
    clock : callable
        Monotonic time source (injectable for tests).
    """

    def __init__(
        self,
        overlay,
        detector: Callable[[object], dict],
        mode: str = "blur",
        hold_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if mode not in RESPONSE_MODES:
            raise ValueError(f"mode must be one of {RESPONSE_MODES}, got {mode!r}")
        self.overlay = overlay
        self.detector = detector
        self.mode = mode
        self.hold_seconds = hold_seconds
        self._clock = clock
        self._shown_at: float | None = None
        self.last_result: dict | None = None

    @property
    def active(self) -> bool:
        """True while the overlay is being held up."""
        return self._shown_at is not None

    def on_frame(self, frame) -> dict | None:
        """
        ``capture_loop`` callback. Returns the detection result, or ``None``
        when the frame was skipped because it contains the overlay.
        """
        now = self._clock()

        if self._shown_at is not None:
            if now - self._shown_at >= self.hold_seconds:
                self.overlay.hide()
                self._shown_at = None
                print("  ↩️  overlay hidden — resuming detection")
            return None

        result = self.detector(frame)
        self.last_result = result
        if result["flagged"]:
            self._trigger(frame, result)
            self._shown_at = now
        return result

    def _trigger(self, frame, result: dict) -> None:
        if self.mode == "blur":
            self.overlay.show_blur(frame.image, caption=_caption(result))
        else:
            self.overlay.show_block(_caption(result) + "\n\nThis screen has been blocked.")
        print(f"  🛡️  overlay shown ({self.mode}) — holding {self.hold_seconds}s")

"""
capture.py — Screen frame capture module for Component 1 (Screen Monitoring).

Responsibilities
----------------
* Periodically grab a screenshot (or receive a frame from an external hook).
* Return a raw PIL Image and associated metadata for downstream processing.

Status: scaffold — replace `capture_frame()` body with real OS-level capture
        (e.g. mss, pyautogui, or a platform API) when ready.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class RawFrame:
    """Container for one captured screen frame."""

    image: Any                          # PIL.Image.Image once real capture is wired up
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    frame_reference: str = ""           # e.g. file path or unique frame ID
    metadata: dict = field(default_factory=dict)


def capture_frame(source: str = "screen") -> RawFrame:
    """
    Capture a single screen frame.

    Parameters
    ----------
    source : str
        Capture source identifier ('screen', or a window title / process name).

    Returns
    -------
    RawFrame
        A raw frame ready for `build_dataset.py` or direct classification.

    Notes
    -----
    **Stub implementation** — returns a placeholder RawFrame with `image=None`.
    Replace the body with a real capture call, e.g.::

        import mss, PIL.Image, io
        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[0])
            img  = PIL.Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    frame_ref = f"frame_{int(time.time() * 1000)}"

    # --- real capture goes here ---
    image = None  # replace with actual PIL.Image

    return RawFrame(
        image=image,
        timestamp=timestamp,
        frame_reference=frame_ref,
        metadata={"source": source},
    )

"""
capture.py — Screen frame capture module for Component 1 (Screen Monitoring).

Responsibilities
----------------
* Grab the primary display screenshot via mss (fastest cross-platform backend).
* Return a RawFrame containing a PIL Image and associated metadata.
* Optionally save the frame to disk (used by the CLI entry-point below).

Dependencies
------------
    mss>=9.0.1
    Pillow>=10.3.0
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class RawFrame:
    """Container for one captured screen frame."""

    image: Any                          # PIL.Image.Image
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    frame_reference: str = ""           # e.g. file path or unique frame ID
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture_frame(
    monitor_index: int = 0,
    source: str = "screen",
) -> RawFrame:
    """
    Capture a single screen frame using mss.

    Parameters
    ----------
    monitor_index : int
        Which monitor to capture. 0 = full virtual desktop (all monitors);
        1 = primary monitor; 2+ = additional monitors.
    source : str
        Logical source label stored in metadata.

    Returns
    -------
    RawFrame
        A frame whose `.image` is a ``PIL.Image.Image`` in RGB mode.

    Raises
    ------
    RuntimeError
        If both mss and the pyautogui fallback are unavailable.
    """
    try:
        import mss
        import mss.tools
        from PIL import Image

        with mss.MSS() as sct:
            # monitor 0 = combined virtual desktop; 1 = primary screen
            monitor = sct.monitors[monitor_index]
            shot = sct.grab(monitor)
            # Convert BGRA raw bytes → PIL RGB image
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    except ImportError:
        # ---------- fallback: pyautogui ----------
        try:
            import pyautogui
            img = pyautogui.screenshot()          # already returns PIL.Image in RGB
        except ImportError as exc:
            raise RuntimeError(
                "Neither mss nor pyautogui is installed. "
                "Run: pip install mss>=9.0.1"
            ) from exc

    timestamp = datetime.now(timezone.utc).isoformat()
    frame_ref = f"frame_{int(time.time() * 1000)}"

    return RawFrame(
        image=img,
        timestamp=timestamp,
        frame_reference=frame_ref,
        metadata={
            "source": source,
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
        },
    )


def save_frame(frame: RawFrame, output_path: Path) -> Path:
    """
    Save a RawFrame's image to *output_path* (PNG).

    Parameters
    ----------
    frame : RawFrame
    output_path : Path
        Destination file.  Parent directories are created automatically.

    Returns
    -------
    Path
        The absolute path of the saved file.
    """
    if frame.image is None:
        raise ValueError("RawFrame.image is None — nothing to save.")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.image.save(output_path, format="PNG")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry-point: python capture.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    OUT = REPO_ROOT / "mock_inputs" / "screenshot.png"

    print("Capturing screen …")
    frame = capture_frame(monitor_index=1)          # primary monitor
    path  = save_frame(frame, OUT)

    print(f"✅ Screenshot saved → {path}")
    print(f"   Size       : {frame.metadata['width']} × {frame.metadata['height']} px")
    print(f"   Mode       : {frame.metadata['mode']}")
    print(f"   Timestamp  : {frame.timestamp}")
    print(f"   Reference  : {frame.frame_reference}")

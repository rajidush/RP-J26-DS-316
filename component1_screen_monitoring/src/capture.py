"""
capture.py — Screen frame capture module for Component 1 (Screen Monitoring).

Responsibilities
----------------
* Grab the primary display screenshot via mss (fastest cross-platform backend).
* Return a RawFrame containing a PIL Image and associated metadata.
* Optionally save the frame to disk (used by the CLI entry-point below).
* Run a continuous capture loop, dispatching each frame to an on_frame callback.
* Provide make_detector_callback() — loads jaranohaal/vit-base-violence-detection
  once (not per frame) and returns a callback that classifies every frame and
  prints flagged / category / confidence to the console.

Dependencies
------------
    mss>=9.0.1
    Pillow>=10.3.0
    timm>=0.9.0          (for make_detector_callback)
    torch>=2.2.0         (for make_detector_callback)
    safetensors>=0.4.0   (for make_detector_callback)
    huggingface_hub      (for make_detector_callback)
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .config import THRESHOLD                     # imported as package
except ImportError:
    from config import THRESHOLD                      # run as script from src/


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
# Capture loop
# ---------------------------------------------------------------------------

def capture_loop(
    on_frame=None,
    interval: float = 1.0,
    max_frames: int | None = None,
    monitor_index: int = 1,
    source: str = "screen",
) -> None:
    """
    Continuously capture screen frames and dispatch each one to *on_frame*.

    Parameters
    ----------
    on_frame : callable | None
        ``on_frame(frame: RawFrame) -> None`` called after every successful
        capture.  If ``None``, frames are captured but silently discarded
        (useful for benchmarking throughput).
    interval : float
        Seconds to wait between captures.  Set to 0 for maximum throughput.
    max_frames : int | None
        Stop after this many frames.  ``None`` runs indefinitely until
        interrupted with Ctrl-C.
    monitor_index : int
        Monitor to capture (1 = primary monitor).
    source : str
        Logical source label stored in each frame's metadata.

    Notes
    -----
    The loop catches per-frame exceptions and prints a warning rather than
    aborting, so transient OS-level failures don't kill a long-running session.
    """
    count = 0
    print(f"[capture_loop] Starting — interval={interval}s  "
          f"max_frames={max_frames if max_frames is not None else '∞'}")
    try:
        while max_frames is None or count < max_frames:
            t_start = time.time()
            try:
                frame = capture_frame(monitor_index=monitor_index, source=source)
                if on_frame is not None:
                    on_frame(frame)
                count += 1
            except Exception as exc:
                print(f"[capture_loop] ⚠️  Frame {count} error: {exc}")

            elapsed = time.time() - t_start
            sleep_for = max(0.0, interval - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        print(f"\n[capture_loop] Interrupted after {count} frame(s).")

    print(f"[capture_loop] Done — {count} frame(s) captured.")


# ---------------------------------------------------------------------------
# Detector callback (loads model once, classifies every frame)
# ---------------------------------------------------------------------------

#: Maps model output index → human-readable category.
#: Discovered empirically: index 0 = Non Violence, index 1 = Violence.
_IDX_TO_CATEGORY: dict[int, str] = {0: "non_violence", 1: "violence"}

#: Confidence threshold above which a frame is considered flagged (shared config).
CONFIDENCE_THRESHOLD: float = THRESHOLD


def make_detector_callback(
    checkpoint: str = "jaranohaal/vit-base-violence-detection",
    timm_arch: str = "vit_base_patch16_224",
    threshold: float = CONFIDENCE_THRESHOLD,
    device: str | None = None,
):
    """
    Build and return an ``on_frame`` callback with the ViT model loaded once.

    The returned callable classifies every ``RawFrame`` it receives and prints
    a structured result line to stdout.

    Parameters
    ----------
    checkpoint : str
        HuggingFace repo ID for the violence-detection model.
    timm_arch : str
        timm architecture string matching the checkpoint's weight keys.
    threshold : float
        Confidence above which ``flagged`` is set to ``True``.
    device : str | None
        ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).

    Returns
    -------
    callable
        ``detect(frame: RawFrame) -> dict``
        The dict contains: flagged, category, confidence, timestamp,
        frame_reference.  It is also printed to stdout.

    Example
    -------
    >>> callback = make_detector_callback()
    >>> capture_loop(on_frame=callback, max_frames=10)
    """
    import torch
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as load_safetensors

    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[detector] Loading {checkpoint} ({timm_arch}) on {_device} …")
    weights_path = hf_hub_download(checkpoint, "model.safetensors")
    _model = timm.create_model(timm_arch, num_classes=2, pretrained=False)
    state_dict = load_safetensors(weights_path, device="cpu")
    missing, unexpected = _model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[detector] ⚠️  missing={len(missing)}  unexpected={len(unexpected)}")
    else:
        print("[detector] ✅ All weights loaded (strict match).")
    _model.to(_device).eval()

    # timm preprocessing pipeline (ImageNet normalisation, 224×224 centre-crop)
    data_config = timm.data.resolve_model_data_config(_model)
    _transform   = timm.data.create_transform(**data_config, is_training=False)

    print(f"[detector] Ready — threshold={threshold}\n")

    def detect(frame: RawFrame) -> dict:
        """Classify *frame* and print + return the result."""
        if frame.image is None:
            raise ValueError("RawFrame.image is None — cannot classify.")

        x = _transform(frame.image).unsqueeze(0).to(_device)
        with torch.no_grad():
            logits = _model(x)[0]                         # shape (2,)
            probs  = torch.softmax(logits, dim=-1)

        pred_idx   = int(logits.argmax().item())
        confidence = float(probs[pred_idx].item())
        category   = _IDX_TO_CATEGORY.get(pred_idx, f"LABEL_{pred_idx}")
        flagged    = (category == "violence") and (confidence >= threshold)

        result = {
            "flagged":         flagged,
            "category":        category,
            "confidence":      round(confidence, 4),
            "timestamp":       frame.timestamp,
            "frame_reference": frame.frame_reference,
        }

        flag_icon = "🚨 FLAGGED" if flagged else "✅ safe   "
        print(
            f"  {flag_icon}  |  category={category:<15s}  "
            f"conf={confidence:.4f}  |  ref={frame.frame_reference}"
        )
        return result

    return detect


# ---------------------------------------------------------------------------
# CLI entry-point: python capture.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Single-shot screenshot save ──────────────────────────────────────
    REPO_ROOT = Path(__file__).resolve().parents[1]
    OUT = REPO_ROOT / "mock_inputs" / "screenshot.png"

    print("Capturing single screenshot …")
    _frame = capture_frame(monitor_index=1)
    _path  = save_frame(_frame, OUT)
    print(f"✅ Screenshot saved → {_path}")
    print(f"   Size      : {_frame.metadata['width']} × {_frame.metadata['height']} px")
    print(f"   Mode      : {_frame.metadata['mode']}")
    print(f"   Timestamp : {_frame.timestamp}")
    print(f"   Reference : {_frame.frame_reference}\n")

    # ── End-to-end loop test: 10 frames, detect on each ──────────────────
    print("=" * 60)
    print("End-to-end test: capture + detect (max_frames=10, interval=0.5s)")
    print("=" * 60)
    detector = make_detector_callback()
    capture_loop(on_frame=detector, max_frames=10, interval=0.5, monitor_index=1)

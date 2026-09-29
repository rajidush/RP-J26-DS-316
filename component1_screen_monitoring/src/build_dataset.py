"""
build_dataset.py — Dataset builder for Component 1 (Screen Monitoring).

Responsibilities
----------------
* Accept captured RawFrames from capture.py.
* Save frames to disk with a structured filename convention.
* Write a JSONL manifest so each saved frame can be re-loaded and labelled.

Status: scaffold — storage path and manifest format are wired up; real
        pre-processing (resize, normalise, dedup) should be added inside
        `process_and_store()`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from capture import RawFrame

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_frames"
MANIFEST_NAME = "manifest.jsonl"


def process_and_store(
    frame: RawFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """
    Persist a captured frame and append a record to the JSONL manifest.

    Parameters
    ----------
    frame : RawFrame
        A frame produced by :func:`capture.capture_frame`.
    output_dir : Path
        Root directory for saved frames.  Created automatically if absent.

    Returns
    -------
    dict
        The manifest record that was written (mirrors the output schema fields
        minus `flagged`, `category`, `confidence` which are set by the
        classifier).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME

    # Build a deterministic file name from the frame reference
    fname = f"{frame.frame_reference}.png"
    frame_path = output_dir / fname

    # --- save image ---
    if frame.image is not None:
        frame.image.save(frame_path)
    else:
        # Stub: write an empty placeholder so the manifest path is real
        frame_path.touch()

    record = {
        "frame_reference": frame.frame_reference,
        "timestamp": frame.timestamp,
        "file_path": str(frame_path),
        "metadata": frame.metadata,
    }

    with manifest_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    return record


def load_manifest(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[dict]:
    """Return all records from the JSONL manifest as a list of dicts."""
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return []
    with manifest_path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]

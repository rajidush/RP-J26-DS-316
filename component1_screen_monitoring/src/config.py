"""
config.py — Single source of tunable settings for Component 1.

Every module (detector, capture, component1, run_monitor) reads these values
instead of hard-coding its own, so the threshold is the same everywhere.
"""
from __future__ import annotations

from pathlib import Path

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPONENT_ROOT.parent

#: P(violent) at or above which a frame is flagged.
THRESHOLD: float = 0.5

#: Seconds between screen samples in the live monitoring loop.
CAPTURE_INTERVAL_S: float = 2.0

#: Fine-tuned ViT checkpoint (not in git — see the component README).
MODEL_DIR: Path = COMPONENT_ROOT / "models" / "violence-vit-screen-ft-v1"

#: Shared contract every emitted payload is validated against.
SCHEMA_PATH: Path = REPO_ROOT / "docs" / "interface-contracts" / "comp1_to_comp3.schema.json"

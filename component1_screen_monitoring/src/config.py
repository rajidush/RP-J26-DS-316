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

#: Which crops of each screen frame are classified (see regions.py):
#: "full", "tiles", "motion" or "tiles+motion".
REGION_STRATEGY: str = "tiles+motion"

#: Seconds between screen samples in the live monitoring loop.
CAPTURE_INTERVAL_S: float = 2.0

#: Fine-tuned ViT checkpoint (not in git — see the component README).
MODEL_DIR: Path = COMPONENT_ROOT / "models" / "violence-vit-screen-ft-v1"

#: Screen-style dataset (scraped pages, composites, recordings) — see data/screens/README.md.
SCREENS_DIR: Path = COMPONENT_ROOT / "data" / "screens"

#: Shared contract every emitted payload is validated against.
SCHEMA_PATH: Path = REPO_ROOT / "docs" / "interface-contracts" / "comp1_to_comp3.schema.json"

# ---------------------------------------------------------------------------
# Function 2 — post-detection response (response_policy.py / response.py)
# ---------------------------------------------------------------------------

#: risk_category values that are always blocked (never just blurred).
BLOCK_CATEGORIES: frozenset[str] = frozenset({"explicit_visual", "self_harm_imagery"})

#: "violence" payloads at or above this confidence are blocked, below it blurred.
#: Also the confidence that overrides a user's dismissal (see DISMISS_GRACE_S).
BLOCK_THRESHOLD: float = 0.85

#: The overlay stays up at least this long after the LAST flagged payload.
MIN_HOLD_S: float = 5.0

#: …and only hides once this many consecutive frames were safe.
CLEAR_FRAMES: int = 2

#: After the user dismisses the overlay, the same incident (session_id) does not
#: re-show it for this long — unless a payload reaches BLOCK_THRESHOLD.
DISMISS_GRACE_S: float = 10.0

#: End-to-end latency log (frame captured → payload → overlay visible).
E2E_LATENCY_JSON: Path = COMPONENT_ROOT / "results" / "e2e_latency.json"

#: What the monitor does while the overlay is visible but could NOT be excluded
#: from screen capture (capture_exclusion.exclude_from_capture() returned False —
#: Linux, Windows < 10 2004). Captures would then contain the overlay, so:
#:   "pause" — skip classification while the overlay is up; after MIN_HOLD_S,
#:             briefly hide it, capture ONE frame, re-show it at once, and clear
#:             the incident if that frame is safe (content may show for ~0.1–0.3 s).
#:   "off"   — classify anyway (frames will contain the overlay; not recommended).
#: Has no effect where exclusion works (verified on macOS).
CAPTURE_EXCLUSION_FALLBACK: str = "pause"

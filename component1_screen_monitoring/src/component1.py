"""
Component 1 (Screen Monitoring) — owned by IT23377844.

Turns screen frames into TriggerPayloads that match
docs/interface-contracts/comp1_to_comp3.schema.json exactly. Component 3 only
depends on that schema, so it needs no changes.

Two modes
---------
* Mock — ``simulate_screen_capture()`` returns a fixed payload from
  mock_inputs/sample_trigger.json (no model, no screen access).
* Real — ``capture_and_classify()`` (one frame) and ``run_monitoring_loop()``
  (continuous) capture the screen, run the fine-tuned ViolenceDetector, and
  emit a schema-validated payload for every flagged frame.

A frame that is not flagged produces NO payload: the schema describes a
risk trigger and has no "safe" category. Raw frames are never written to
disk or included in a payload.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

try:
    from .config import CAPTURE_INTERVAL_S, REGION_STRATEGY, SCHEMA_PATH  # imported as package
    from .regions import motion_reference
except ImportError:
    from config import CAPTURE_INTERVAL_S, REGION_STRATEGY, SCHEMA_PATH   # run as script from src/
    from regions import motion_reference

log = logging.getLogger(__name__)

MOCK_INPUTS = Path(__file__).resolve().parents[1] / "mock_inputs"

SOURCE_COMPONENT = "component1_screen_monitoring"
CONTENT_TYPE = "image"

#: Detector category → schema risk_category.
_RISK_CATEGORY = {"violence": "violence"}


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------

def simulate_screen_capture(mock_file: str = "sample_trigger.json") -> dict:
    with open(MOCK_INPUTS / mock_file) as f:
        payload = json.load(f)
    # fresh session id + timestamp each run, so integration demos don't collide
    payload["session_id"] = str(uuid.uuid4())
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    return payload


# ---------------------------------------------------------------------------
# Payload building + validation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _validator():
    from jsonschema import Draft7Validator

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    Draft7Validator.check_schema(schema)
    # FORMAT_CHECKER makes "format": "date-time" an actual check
    # (needs rfc3339-validator installed; otherwise it is silently skipped).
    return Draft7Validator(schema, format_checker=Draft7Validator.FORMAT_CHECKER)


def validate_payload(payload: dict) -> dict:
    """
    Validate *payload* against comp1_to_comp3.schema.json.

    Returns the payload unchanged; raises ``jsonschema.ValidationError`` if invalid.
    """
    _validator().validate(payload)
    return payload


def build_payload(
    frame: Any,
    detection: Any,
    session_id: str | None = None,
    duration_visible_ms: int = 0,
) -> dict | None:
    """
    Build a schema-valid TriggerPayload from a ``capture.RawFrame`` and a
    ``detector.Detection``. Returns ``None`` if the frame is not flagged.
    """
    if not detection.flagged:
        return None

    payload = {
        "session_id": session_id or str(uuid.uuid4()),
        "timestamp": frame.timestamp,
        "source_component": SOURCE_COMPONENT,
        "content_type": CONTENT_TYPE,
        "risk_category": _RISK_CATEGORY.get(detection.category, "unknown_flagged"),
        "confidence_score": float(detection.confidence),
        "context_metadata": {"duration_visible_ms": int(duration_visible_ms)},
    }
    return validate_payload(payload)


class IncidentTracker:
    """
    Groups consecutive flagged frames into one incident.

    Every payload in an incident shares a ``session_id``;
    ``duration_visible_ms`` is the time since the incident's first flagged
    frame. A frame that is not flagged ends the incident.
    """

    def __init__(self) -> None:
        self.session_id: str | None = None
        self._started: datetime | None = None

    def update(self, frame: Any, flagged: bool) -> tuple[str, int] | None:
        if not flagged:
            self.session_id = self._started = None
            return None
        ts = datetime.fromisoformat(frame.timestamp)
        if self.session_id is None:
            self.session_id, self._started = str(uuid.uuid4()), ts
        return self.session_id, int((ts - self._started).total_seconds() * 1000)


# ---------------------------------------------------------------------------
# Real mode
# ---------------------------------------------------------------------------

def _default_capture():
    try:
        from .capture import capture_frame
    except ImportError:
        from capture import capture_frame
    return capture_frame(monitor_index=1)


def capture_and_classify(
    detector: Any,
    capture_fn: Callable[[], Any] | None = None,
    strategy: str = REGION_STRATEGY,
) -> dict | None:
    """
    Capture one real frame, classify it, and return a payload (or ``None`` if not flagged).

    A single frame has no previous frame, so the "motion" region is skipped.
    """
    frame = (capture_fn or _default_capture)()
    detection = detector.predict_regions(frame.image, None, strategy)
    log.info("frame %s → %s (p_violent=%.4f in %s, %d crops, %.0f ms)", frame.frame_reference,
             detection.category, detection.confidence, detection.region, detection.n_crops,
             detection.inference_ms)
    return build_payload(frame, detection)


def run_monitoring_loop(
    detector: Any,
    interval: float = CAPTURE_INTERVAL_S,
    max_iterations: int | None = None,
    on_payload: Callable[[dict], None] | None = None,
    capture_fn: Callable[[], Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    strategy: str = REGION_STRATEGY,
) -> dict:
    """
    Sample the screen every *interval* seconds and emit a payload per flagged frame.

    Each frame is classified with ``detector.predict_regions`` (*strategy*).
    Only the latest frame plus a small grayscale copy of the previous one (for
    motion) are held in memory, and nothing is saved to disk.
    Not-flagged frames are logged at DEBUG. Ctrl+C stops the loop cleanly.

    Parameters
    ----------
    detector : object with ``predict(image) -> Detection`` (e.g. ViolenceDetector)
    interval : seconds between samples (default: config.CAPTURE_INTERVAL_S)
    max_iterations : stop after this many frames; ``None`` runs until Ctrl+C
    on_payload : called with each validated payload (default: log it)
    capture_fn, sleep_fn : injectable for tests
    strategy : region strategy (default: config.REGION_STRATEGY)

    Returns
    -------
    dict
        ``{"frames": n, "payloads": n, "errors": n}``
    """
    capture_fn = capture_fn or _default_capture
    on_payload = on_payload or (lambda p: log.info("payload %s", json.dumps(p)))
    tracker = IncidentTracker()
    stats = {"frames": 0, "payloads": 0, "errors": 0}
    prev = None                                   # small grayscale copy of the last frame

    log.info("monitoring started — interval=%ss, max_iterations=%s, threshold=%s, strategy=%s",
             interval, max_iterations, getattr(detector, "threshold", "?"), strategy)
    try:
        while max_iterations is None or stats["frames"] < max_iterations:
            t0 = time.monotonic()
            try:
                frame = capture_fn()                  # replaces the previous frame
                detection = detector.predict_regions(frame.image, prev, strategy)
                prev = motion_reference(frame.image)
                incident = tracker.update(frame, detection.flagged)
                if incident:
                    payload = build_payload(frame, detection, *incident)
                    on_payload(payload)
                    stats["payloads"] += 1
                else:
                    log.debug("frame %s not flagged (p_violent=%.4f in %s, %d crops, %.0f ms)",
                              frame.frame_reference, detection.confidence, detection.region,
                              detection.n_crops, detection.inference_ms)
            except Exception:
                stats["errors"] += 1
                log.exception("frame %d failed", stats["frames"] + 1)
            stats["frames"] += 1

            if max_iterations is not None and stats["frames"] >= max_iterations:
                break
            sleep_fn(max(0.0, interval - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        log.info("interrupted by user")

    log.info("monitoring stopped — %(frames)d frames, %(payloads)d payloads, "
             "%(errors)d errors", stats)
    return stats

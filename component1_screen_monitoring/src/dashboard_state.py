"""
dashboard_state.py — Streamlit-free helpers for the Function 1 demo dashboard.

Kept separate from dashboard.py so they can be unit-tested without Streamlit.
All detection / payload logic is reused from detector.py and component1.py —
nothing here re-implements it.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFilter

try:
    from .component1 import IncidentTracker, build_payload   # imported as package
    from .regions import motion_reference
except ImportError:
    from component1 import IncidentTracker, build_payload    # run from src/
    from regions import motion_reference

#: Readings kept in memory for the chart / counters.
MAX_READINGS = 200

THUMBNAIL_SIZE = (480, 300)


@dataclass
class Reading:
    """One processed frame (no image data — thumbnails are kept separately)."""

    timestamp: str
    frame_reference: str
    p_violent: float
    flagged: bool
    inference_ms: float
    total_ms: float
    payload: dict | None
    region: str = "full"
    region_box: tuple[int, int, int, int] | None = None
    n_crops: int = 1


@dataclass
class ReadingHistory:
    """Bounded in-memory history of readings and emitted payloads."""

    maxlen: int = MAX_READINGS
    readings: deque = field(init=False)
    payloads: deque = field(init=False)
    frames_processed: int = 0
    payloads_emitted: int = 0
    _total_ms_sum: float = 0.0
    _crops_sum: int = 0
    _started: datetime | None = None

    def __post_init__(self) -> None:
        self.readings = deque(maxlen=self.maxlen)
        self.payloads = deque(maxlen=self.maxlen)

    def add(self, reading: Reading) -> None:
        self.readings.append(reading)
        self.frames_processed += 1
        self._total_ms_sum += reading.total_ms
        self._crops_sum += reading.n_crops
        if self._started is None:
            self._started = datetime.fromisoformat(reading.timestamp)
        if reading.payload is not None:
            self.payloads.append(reading.payload)
            self.payloads_emitted += 1

    @property
    def latest(self) -> Reading | None:
        return self.readings[-1] if self.readings else None

    @property
    def mean_latency_ms(self) -> float:
        """Mean total (capture + inference) ms over every frame processed."""
        return self._total_ms_sum / self.frames_processed if self.frames_processed else 0.0

    @property
    def mean_crops(self) -> float:
        """Mean number of crops classified per frame."""
        return self._crops_sum / self.frames_processed if self.frames_processed else 0.0

    def payloads_newest_first(self) -> list[dict]:
        return list(reversed(self.payloads))

    def chart_rows(self, threshold: float) -> list[dict]:
        """
        Rows for the p(violent)-over-time chart: x = seconds since monitoring
        started, plus the threshold as a flat line.
        """
        return [{"seconds": round((datetime.fromisoformat(r.timestamp) - self._started)
                                  .total_seconds(), 1),
                 "p_violent": r.p_violent, "threshold": threshold}
                for r in self.readings]


def make_thumbnail(
    image: Image.Image,
    blur: bool,
    box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """
    Small in-memory preview; heavily blurred when the frame is flagged.
    *box* (frame coordinates) is drawn as a red rectangle — the winning region.
    """
    thumb = image.convert("RGB")
    full_w = thumb.width
    thumb.thumbnail(THUMBNAIL_SIZE)
    if blur:
        thumb = thumb.filter(ImageFilter.GaussianBlur(radius=18))
    if box is not None:
        s = thumb.width / full_w
        ImageDraw.Draw(thumb).rectangle([round(c * s) for c in box], outline="#ff2d2d", width=3)
    return thumb


def process_frame(
    detector: Any,
    capture_fn: Callable[[], Any],
    tracker: IncidentTracker,
    strategy: str,
    prev: Image.Image | None = None,
) -> tuple[Reading, Image.Image, Image.Image]:
    """
    Capture one frame, classify its regions, and build the payload if flagged.

    Mirrors one iteration of ``component1.run_monitoring_loop`` (same
    ``predict_regions``, tracker and ``build_payload``), plus timings and a
    thumbnail for display.

    Returns ``(reading, thumbnail, motion_reference)`` — pass the last one back
    as *prev* on the next call (it is a small grayscale copy, not the frame).
    """
    t0 = time.perf_counter()
    frame = capture_fn()
    capture_ms = (time.perf_counter() - t0) * 1000
    detection = detector.predict_regions(frame.image, prev, strategy)

    incident = tracker.update(frame, detection.flagged)
    payload = build_payload(frame, detection, *incident) if incident else None

    reading = Reading(
        timestamp=frame.timestamp,
        frame_reference=frame.frame_reference,
        p_violent=detection.confidence,
        flagged=detection.flagged,
        inference_ms=detection.inference_ms,
        total_ms=round(capture_ms + detection.inference_ms, 1),
        payload=payload,
        region=detection.region,
        region_box=detection.region_box,
        n_crops=detection.n_crops,
    )
    box = detection.region_box if detection.region != "full" else None   # full = no box
    thumb = make_thumbnail(frame.image, blur=detection.flagged, box=box)
    return reading, thumb, motion_reference(frame.image)

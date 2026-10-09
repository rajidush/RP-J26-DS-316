"""Tests for src/dashboard_state.py (no Streamlit, no real model or screen)."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.capture import RawFrame
from component1_screen_monitoring.src.component1 import IncidentTracker, validate_payload
from component1_screen_monitoring.src.dashboard_state import (
    Reading,
    ReadingHistory,
    make_thumbnail,
    process_frame,
)

T0 = datetime(2026, 10, 7, 10, 0, 0, tzinfo=timezone.utc)


def _reading(i, total_ms=100.0, payload=None, n_crops=1):
    return Reading(timestamp=(T0 + timedelta(seconds=2 * i)).isoformat(), frame_reference=f"f{i}",
                   p_violent=0.1, flagged=payload is not None, inference_ms=50.0,
                   total_ms=total_ms, payload=payload, n_crops=n_crops)


def test_history_is_bounded_but_counters_are_not():
    h = ReadingHistory(maxlen=3)
    for i in range(5):
        h.add(_reading(i, payload={"n": i} if i % 2 == 0 else None))
    assert [r.frame_reference for r in h.readings] == ["f2", "f3", "f4"]
    assert h.latest.frame_reference == "f4"
    assert h.frames_processed == 5
    assert h.payloads_emitted == 3
    assert h.payloads_newest_first() == [{"n": 4}, {"n": 2}, {"n": 0}]


def test_mean_latency_and_empty_history():
    h = ReadingHistory()
    assert h.latest is None and h.mean_latency_ms == 0.0
    h.add(_reading(0, total_ms=100))
    h.add(_reading(1, total_ms=300))
    assert h.mean_latency_ms == 200.0


def test_chart_rows_use_seconds_since_start_and_threshold_line():
    h = ReadingHistory(maxlen=2)
    for i in range(3):                         # first reading falls out of the buffer…
        h.add(_reading(i, n_crops=6 if i else 1))
    assert h.chart_rows(0.5) == [              # …but the clock still starts at it
        {"seconds": 2.0, "p_violent": 0.1, "threshold": 0.5},
        {"seconds": 4.0, "p_violent": 0.1, "threshold": 0.5},
    ]
    assert round(h.mean_crops, 2) == round((1 + 6 + 6) / 3, 2)


def test_thumbnail_draws_region_box():
    img = Image.new("RGB", (1440, 900), "black")
    thumb = make_thumbnail(img, blur=False, box=(720, 450, 1440, 900))   # bottom-right quarter
    assert thumb.getpixel((240, 150))[0] > 200                          # box corner → red
    assert thumb.getpixel((100, 50)) == (0, 0, 0)                       # outside box untouched


def test_thumbnail_is_small_and_blurred_only_when_flagged():
    img = Image.new("L", (1440, 900))
    img.putdata([255 * ((x // 20 + y // 20) % 2) for y in range(900) for x in range(1440)])
    sharp, blurred = make_thumbnail(img, blur=False), make_thumbnail(img, blur=True)
    assert max(sharp.size) <= 480
    lo, hi = blurred.convert("L").getextrema()
    assert hi - lo < sharp.convert("L").getextrema()[1] - sharp.convert("L").getextrema()[0]


class FakeDetector:
    def __init__(self, probs, threshold=0.5):
        self._probs, self.threshold = iter(probs), threshold

    def predict_regions(self, image, prev_image=None, strategy="tiles+motion"):
        p = next(self._probs)
        flagged = p >= self.threshold
        return SimpleNamespace(category="violence" if flagged else "safe",
                               confidence=p, flagged=flagged, inference_ms=40.0,
                               region="centre", region_box=(10, 10, 50, 50), n_crops=6)


def _capture():
    n = {"i": 0}

    def capture():
        ts = (T0 + timedelta(seconds=2 * n["i"])).isoformat()
        n["i"] += 1
        return RawFrame(image=Image.new("RGB", (64, 64)), timestamp=ts,
                        frame_reference=f"frame_{n['i']}")
    return capture


def test_process_frame_safe_then_flagged_incident():
    det, cap, tracker = FakeDetector([0.1, 0.9, 0.95]), _capture(), IncidentTracker()

    safe, _, prev = process_frame(det, cap, tracker, "tiles+motion")
    assert safe.payload is None and not safe.flagged
    assert safe.total_ms >= safe.inference_ms
    assert (safe.region, safe.n_crops) == ("centre", 6)
    assert prev.mode == "L"                    # small grayscale copy, not the frame

    first, thumb, prev = process_frame(det, cap, tracker, "tiles+motion", prev)
    second, _, _ = process_frame(det, cap, tracker, "tiles+motion", prev)
    for r in (first, second):
        validate_payload(r.payload)
    assert first.payload["session_id"] == second.payload["session_id"]
    assert second.payload["context_metadata"]["duration_visible_ms"] == 2000
    assert isinstance(thumb, Image.Image)

# Component 1 — Screen Monitoring

**Owner:** IT23377844
**Status:** Function 1 complete — real screen capture → fine-tuned violence detector → schema-valid payload.

## What it does

Samples the screen every few seconds, classifies each frame with a fine-tuned
ViT violence classifier, and emits a `TriggerPayload` for every **flagged**
frame. Frames that are not flagged produce no payload (the contract describes
a risk trigger and has no "safe" category). Raw frames are never written to
disk or sent downstream — only classification metadata.

Every payload is validated with `jsonschema` against
[`docs/interface-contracts/comp1_to_comp3.schema.json`](../docs/interface-contracts/comp1_to_comp3.schema.json)
before it is emitted, so Component 3 needs no changes.

| Payload field | Source |
|---|---|
| `session_id` | UUID per incident (a run of consecutive flagged frames) |
| `timestamp` | capture time of the frame (UTC, ISO 8601) |
| `source_component` | `"component1_screen_monitoring"` |
| `content_type` | `"image"` |
| `risk_category` | `"violence"` |
| `confidence_score` | detector's P(violent), 0–1 |
| `context_metadata.duration_visible_ms` | time since the incident's first flagged frame |

## Setup

```bash
cd component1_screen_monitoring
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The fine-tuned checkpoint (~330 MB) is **not in git**. Unzip
`violence-vit-screen-ft-v1.zip` so that `config.json` sits directly in:

```
component1_screen_monitoring/models/violence-vit-screen-ft-v1/
```

It needs `transformers<5` (saved with 4.57). On macOS, the terminal running
the monitor needs **Screen Recording** permission (System Settings → Privacy &
Security), otherwise captures come back black.

## Run it

All commands from `component1_screen_monitoring/`:

```bash
# Mock mode — fixed payload from mock_inputs/sample_trigger.json (no model, no capture)
python src/run_monitor.py --mock

# One real frame — capture, classify, print the payload (or "No payload — frame not flagged.")
python src/run_monitor.py --once

# Live loop — sample every 2 s until Ctrl+C
python src/run_monitor.py --live

# Live loop, 5 frames, show every not-flagged frame (DEBUG)
python src/run_monitor.py --live --max-iterations 5 -v
```

Options: `--interval SECONDS` (live sampling period), `--max-iterations N`,
`--threshold P` (flag when P(violent) ≥ P), `--strategy {full,tiles,motion,tiles+motion}`
(which crops are classified — see below), `-v` (debug logging).

## Region-based inference

Squashing the whole 1440×900 screen into the model's 224×224 input makes a
video in a normal-size window (e.g. a YouTube Short in a browser) a tiny part
of the image, and the model — trained on full-frame video stills — misses it.
So each frame is classified as a **batch of crops in one forward pass**
(`ViolenceDetector.predict_regions`, crops chosen in `src/regions.py`):

| Strategy | Crops |
|---|---|
| `full` | whole frame (old behaviour) |
| `tiles` | whole frame + 2×2 overlapping grid (60% × 60% each) + centre crop |
| `motion` | whole frame + the largest region that changed since the previous frame (padded 10%) |
| `tiles+motion` (default) | both |

p(violent) is the **maximum over the crops**; the winning region's name and box
are reported (and drawn on the dashboard thumbnail). The payload schema is
unchanged. Motion only needs a 160-px grayscale copy of the previous frame,
which is all that is kept in memory.

Benchmark every strategy on live frames → `results/latency_regions.json`:

```bash
python src/run_monitor.py --benchmark 20
```

The motion crop is only added when part of the screen is changing, so run the
benchmark with a video playing to measure the 7-crop case.

From Python:

```python
from component1_screen_monitoring.src.component1 import (
    simulate_screen_capture, capture_and_classify, run_monitoring_loop)
from component1_screen_monitoring.src.detector import ViolenceDetector

payload = simulate_screen_capture()                     # mock
det = ViolenceDetector()
payload = capture_and_classify(det)                     # one frame → dict or None
run_monitoring_loop(det, max_iterations=10, on_payload=print)
```

## Dashboard

A local Streamlit dashboard for demoing Function 1:

```bash
streamlit run component1_screen_monitoring/src/dashboard.py
```

- **Live monitoring** — Start/Stop, capture interval, threshold slider,
  region strategy and monitor picker in the sidebar; latest frame thumbnail
  (blurred when flagged, red box = highest-scoring region), big SAFE /
  FLAGGED status, p(violent), inference and total ms, a p(violent) chart over
  seconds since start with the threshold line, counters (frames, payloads,
  mean latency, crops per frame), and the schema-valid TriggerPayloads
  emitted (newest first).
- **Test an image** — upload an image to see its p(violent), status, and the
  payload it would produce (or "no payload — safe").

The detector is loaded once; at most the last 200 readings are kept in memory
and frames are never written to disk.

> **macOS:** the app that launches Streamlit (Terminal, iTerm or VS Code) needs
> **Screen Recording** permission — System Settings → Privacy & Security →
> Screen Recording. Restart that app after granting it. Without it the
> thumbnails come back black (the dashboard warns you).

## Configuration

All tunables live in [`src/config.py`](src/config.py) and are used everywhere
(detector, capture, monitor, CLI defaults):

| Setting | Default | Meaning |
|---|---|---|
| `THRESHOLD` | `0.5` | flag a frame when P(violent) ≥ this |
| `REGION_STRATEGY` | `tiles+motion` | which crops of each frame are classified |
| `CAPTURE_INTERVAL_S` | `2.0` | seconds between samples in the live loop |
| `MODEL_DIR` | `models/violence-vit-screen-ft-v1` | fine-tuned checkpoint |
| `SCHEMA_PATH` | `docs/interface-contracts/comp1_to_comp3.schema.json` | contract payloads are validated against |

## Other tools

```bash
python src/detector.py path/to/image.png    # classify one image file
python src/detector.py --benchmark 20       # capture + inference latency → see results/latency.json
```

## Tests

```bash
pytest component1_screen_monitoring/tests -v
```

`tests/test_pipeline.py` covers schema validation (flagged / not-flagged /
invalid payloads) and the monitoring loop using fake detections, so it does not
need the model or screen access. `tests/test_detector.py` uses a tiny random
ViT, plus one test against the real checkpoint if it is present.
`tests/test_regions.py` checks crop counts per strategy, motion detection on
synthetic frames, that boxes stay inside the frame, and that `predict_regions`
returns the max of the per-crop confidences (tiny random ViT).

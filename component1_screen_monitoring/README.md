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
`--threshold P` (flag when P(violent) ≥ P), `-v` (debug logging).

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

## Configuration

All tunables live in [`src/config.py`](src/config.py) and are used everywhere
(detector, capture, monitor, CLI defaults):

| Setting | Default | Meaning |
|---|---|---|
| `THRESHOLD` | `0.5` | flag a frame when P(violent) ≥ this |
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

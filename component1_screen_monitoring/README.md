# Component 1 — Screen Monitoring

**Owner:** IT23377844
**Status:** Function 1 complete — real screen capture → fine-tuned violence detector → schema-valid payload.
Function 2 complete — the payload drives an automatic blur / block overlay (see [Function 2 / full demo](#function-2--full-demo)).

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

Works on macOS, Windows and Linux with Python 3.10+; the same
`requirements.txt` installs everywhere (no platform-specific packages). See
[Running on macOS](#running-on-macos) / [Running on Windows](#running-on-windows)
for venv activation and OS notes.

The fine-tuned checkpoint (~330 MB) is **not in git**. Unzip
`violence-vit-screen-ft-v1.zip` so that `config.json` sits directly in:

```
component1_screen_monitoring/models/violence-vit-screen-ft-v1/
```

It needs `transformers<5` (saved with 4.57).

### Running on macOS

```bash
cd component1_screen_monitoring
python3 -m venv .venv
source .venv/bin/activate          # prompt shows (.venv); `deactivate` to leave
pip install -r requirements.txt
python src/run_monitor.py --mock --respond
```

- Give the app you run it from (Terminal, iTerm or VS Code) **Screen Recording**
  permission — System Settings → Privacy & Security → Screen Recording — then
  restart that app. Without it captures are black.
- Inference runs on the Apple-silicon GPU (`mps`) automatically.
- The overlay is excluded from screen capture (verified on macOS 15.3.1), so
  detection keeps running while it is visible.

### Running on Windows

PowerShell:

```powershell
cd component1_screen_monitoring
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1      # prompt shows (.venv); `deactivate` to leave
pip install -r requirements.txt
python src\run_monitor.py --mock --respond
```

Command Prompt (cmd.exe): activate with `.venv\Scripts\activate.bat` instead.

- If PowerShell refuses to run `Activate.ps1` ("running scripts is disabled"),
  allow it for your user once:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use cmd.exe.
- Use `\` or `/` in paths (`python src\run_monitor.py` or `python src/run_monitor.py`).
- No Screen Recording permission is needed. Inference runs on CPU unless you
  installed a CUDA build of PyTorch (see `requirements.txt`).
- **Capture exclusion** uses `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`,
  available on **Windows 10 version 2004 or later** (and Windows 11). On older
  Windows the overlay cannot be excluded, and the monitor automatically uses the
  `"pause"` fallback (see [Function 2](#function-2--full-demo)). The log line
  `capture exclusion on win32: …` says which applies. *Not yet tested on a
  Windows machine — please report the log line from a teammate's run.*

**Linux:** set up like macOS (`source .venv/bin/activate`). Screen-capture
exclusion is not available on Linux, so the `"pause"` fallback is always used.

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

## Function 2 / full demo

Function 2 turns each schema-valid `TriggerPayload` into an on-screen response:
a full-screen, always-on-top overlay that either **blurs** the screen or
**blocks** it with a message. Add `--respond` to `--live` or `--mock`
(without `--respond`, `run_monitor.py` is Function 1 only, unchanged).

All commands from `component1_screen_monitoring/`:

```bash
# Scripted demo — no model, no violent content: blur → hold extended → hidden → block → hidden (~17 s)
python src/run_monitor.py --mock --respond

# Live: real capture → fine-tuned detector → payload → overlay, rules from config.py
python src/run_monitor.py --live --respond            # same as --respond auto

# Live, forced mode (demo / strict setting)
python src/run_monitor.py --live --respond blur
python src/run_monitor.py --live --respond block
```

**Decision rules** (`src/response_policy.py`, thresholds in `config.py`); every
payload is validated against the schema first — invalid payloads are logged and ignored:

| risk_category | confidence | action |
|---|---|---|
| `explicit_visual`, `self_harm_imagery` | any | block |
| `violence` | ≥ `BLOCK_THRESHOLD` (0.85) | block |
| `violence` | < 0.85 | blur |
| `unknown_flagged` | any | blur |

Within one incident the response only escalates (blur → block), never back.

**Hold / no flicker** (`src/response.py`). Detection keeps running while the
overlay is up. Every flagged payload extends the hold; the overlay hides only
when ≥ `MIN_HOLD_S` (5 s) have passed since the last flagged payload **and** the
last `CLEAR_FRAMES` (2) frames were safe.

This works because the overlay window is **excluded from screen capture**
(`src/capture_exclusion.py`, ctypes — no extra dependency):

| Platform | Method | Status |
|---|---|---|
| macOS | `NSWindow.sharingType = none` | **verified** on macOS 15.3.1 |
| Windows 10 2004+ / 11 | `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` | implemented, not yet tested on Windows |
| older Windows | `WDA_MONITOR` (overlay captured as black) → treated as *not* excluded | fallback |
| Linux | not available | fallback |

macOS verification: with the overlay visible, a capture taken *without* the
exclusion was 90.6 % overlay-dark pixels (mean difference from the
hidden-overlay baseline 80.5 / 255); *with* the exclusion it was 3.2 % —
identical to the baseline — with a mean difference of 1.5 / 255, i.e. the
capture shows the real screen underneath.

**Fallback when the overlay cannot be excluded** (`CAPTURE_EXCLUSION_FALLBACK =
"pause"`): frames would contain the overlay, so while it is visible the monitor
**does not classify**. After `MIN_HOLD_S` it hides the overlay for one capture,
re-shows it immediately, classifies that frame while covered, and clears the
incident if it is safe (a flagged frame extends the hold instead). The content
can show through for one peek — measured at **≈ 0.2 s** on macOS with exclusion
deliberately off. Where exclusion works (macOS) this fallback is never used.

**Dismiss.** Click **Dismiss (Esc)** or press **Esc** — both verified on macOS
(the borderless overlay is made able to become the key window, so it receives
Esc; a real Esc key event closed it). After a dismiss, the same incident does
not re-show the overlay for `DISMISS_GRACE_S` (10 s) unless a payload reaches
`BLOCK_THRESHOLD`; a new incident shows it immediately. Ctrl+C in the terminal
stops everything.

**End-to-end latency.** For every overlay appearance (incident start or
escalation) the run appends frame captured → payload emitted → overlay visible
to `results/e2e_latency.json` and prints a one-line summary on exit. The model
is warmed up first so the first measurement is not a cold-start outlier.

> **macOS:** the app you run this from (Terminal, iTerm or VS Code) needs
> **Screen Recording** permission — System Settings → Privacy & Security →
> Screen Recording — then restart that app. Without it captures are black,
> nothing is detected, and the mock demo falls back to a plain demo picture.

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
| `BLOCK_CATEGORIES` | `explicit_visual`, `self_harm_imagery` | always blocked |
| `BLOCK_THRESHOLD` | `0.85` | violence ≥ this → block (below → blur); also overrides a dismissal |
| `MIN_HOLD_S` | `5.0` | overlay stays up at least this long after the last flagged payload |
| `CLEAR_FRAMES` | `2` | consecutive safe frames needed before hiding |
| `DISMISS_GRACE_S` | `10.0` | after Dismiss, the same incident does not re-show for this long |
| `E2E_LATENCY_JSON` | `results/e2e_latency.json` | end-to-end latency log |
| `CAPTURE_EXCLUSION_FALLBACK` | `"pause"` | what to do when the overlay can't be excluded from capture (`"pause"` or `"off"`) |

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
`tests/test_response.py` covers Function 2 without a GUI (fake overlay + fake
clock): decision rules per category/confidence, the `--respond` override,
invalid payloads, hold extension, the `MIN_HOLD_S` + `CLEAR_FRAMES` hide rule,
escalation, dismiss grace, the latency log and the mock script.
`tests/test_cross_platform.py` covers Windows/Linux behaviour on any OS: the
`SetWindowDisplayAffinity` logic with a fake `user32`, platform dispatch, that no
module imports platform libraries at import time, and the `"pause"` fallback
through the real monitoring loop.
`tests/test_overlay.py` checks the Tk overlay (blur/block, Dismiss button, Esc
binding, visibility callback) and is skipped when no display is available.

"""
run_monitor.py — Command-line runner for Component 1 (Function 1 + Function 2).

Function 1 only (no GUI)
------------------------
    python component1_screen_monitoring/src/run_monitor.py --mock
    python component1_screen_monitoring/src/run_monitor.py --once
    python component1_screen_monitoring/src/run_monitor.py --live [--interval 2] [--max-iterations N]
    python component1_screen_monitoring/src/run_monitor.py --benchmark 20

Function 2 — add --respond to show the blur/block overlay
---------------------------------------------------------
    python component1_screen_monitoring/src/run_monitor.py --mock --respond          # scripted demo, no model
    python component1_screen_monitoring/src/run_monitor.py --live --respond          # auto: rules in config.py
    python component1_screen_monitoring/src/run_monitor.py --live --respond blur     # always blur
    python component1_screen_monitoring/src/run_monitor.py --live --respond block    # always block
    python component1_screen_monitoring/src/run_monitor.py --live --mode strict --respond

With --respond the overlay runs on the main thread and the capture/detection
loop (or the mock script) in a worker thread. Dismiss with the button or Esc;
stop with Ctrl+C. End-to-end latency is appended to
results/function2/e2e_latency.json. --max-iterations N also limits the mock
script to its first N steps.

Common options:
    --mode default|strict   default = whole screen only ("full"); strict = whole
                            screen + 2×2 tiles + centre ("tiles") — more recall,
                            more false alarms (see config.MODES). Default: default.
    --model-dir DIR         checkpoint (default: config.MODEL_DIR, ft-v2); for comparison
                            models/violence-vit-screen-ft-v1 (V1, config.RLVS_FT_V1_MODEL_DIR)
                            or models/violence-vit-pretrained-hf (P, config.PRETRAINED_MODEL_DIR)
    --strategy S            advanced: any region strategy, overrides --mode
    --threshold T           default: config.THRESHOLD
    --respond [auto|blur|block]   Function 2 overlay (with --live or --mock)
    -v                      DEBUG logs, including every not-flagged frame

--benchmark N captures N live frames once, then times every region strategy
on those same frames and saves results/03_screen_eval_pretrained/latency_regions.json.

Every printed payload has been validated against comp1_to_comp3.schema.json.
Stop --live with Ctrl+C.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.component1 import (
    capture_and_classify,
    run_monitoring_loop,
    simulate_screen_capture,
    validate_payload,
)
from component1_screen_monitoring.src.config import (
    CAPTURE_INTERVAL_S,
    COMPONENT_ROOT,
    DEFAULT_MODE,
    MODEL_DIR,
    MODES,
    RESULTS_SCREEN_EVAL,
    THRESHOLD,
)
from component1_screen_monitoring.src.regions import STRATEGIES, motion_reference

LATENCY_REGIONS_JSON = RESULTS_SCREEN_EVAL / "latency_regions.json"
#: Gap between benchmark captures — short, but long enough for video to move.
BENCHMARK_GAP_S = 0.5


def _print_payload(payload: dict) -> None:
    print(json.dumps(payload, indent=2), flush=True)


def _machine_name() -> str:
    """CPU name on macOS (sysctl); platform.processor()/machine() elsewhere (no sysctl on Windows)."""
    if sys.platform == "darwin":
        try:
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
            if name:
                return name
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def _stats(xs: list[float]) -> dict:
    return {"mean": round(statistics.mean(xs), 1), "median": round(statistics.median(xs), 1),
            "max": round(max(xs), 1)}


def benchmark_strategies(detector, n: int) -> dict:
    """
    Capture *n* live frames (kept in memory only), then time predict_regions
    for every strategy on the same frames. Returns the results dict.
    """
    from component1_screen_monitoring.src.capture import capture_frame

    frames, capture_ms = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        frames.append(capture_frame(monitor_index=1).image)
        capture_ms.append((time.perf_counter() - t0) * 1000)
        time.sleep(BENCHMARK_GAP_S)

    results = {}
    for strategy in STRATEGIES:
        base_crops = detector.predict_regions(frames[0], frames[0], strategy).n_crops  # warm-up;
        inf_ms, crops, motion_frames = [], [], 0      # identical frames → no motion crop
        prev = None
        for img in frames:
            d = detector.predict_regions(img, prev, strategy)
            prev = motion_reference(img)
            inf_ms.append(d.inference_ms)
            crops.append(d.n_crops)
            motion_frames += d.n_crops > base_crops
        total = [c + i for c, i in zip(capture_ms, inf_ms)]
        results[strategy] = {
            "inference_ms": _stats(inf_ms),
            "total_ms": _stats(total),
            "mean_crops": round(statistics.mean(crops), 2),
            "frames_with_motion_crop": motion_frames if "motion" in strategy else None,
        }
        print(f"  {strategy:<13} crops {statistics.mean(crops):4.1f}   inference mean "
              f"{statistics.mean(inf_ms):6.1f} ms   total mean {statistics.mean(total):6.1f} ms",
              flush=True)

    return {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "command": f"python component1_screen_monitoring/src/run_monitor.py --benchmark {n}",
        "device": detector.device,
        "machine": _machine_name(),
        "frames": n,
        "frame_size": {"width": frames[0].width, "height": frames[0].height},
        "gap_between_captures_s": BENCHMARK_GAP_S,
        "note": "Same captured frames for every strategy; the motion crop is only added "
                "when part of the screen changed between consecutive frames.",
        "capture_ms": _stats(capture_ms),
        "strategies": results,
    }


def _mock_frame_fn():
    """Frames for the mock demo: one real screen capture (if allowed), re-stamped each step."""
    from component1_screen_monitoring.src.capture import RawFrame, capture_frame
    from PIL import Image, ImageDraw

    try:
        base = capture_frame(monitor_index=1).image
    except Exception:                                     # no Screen Recording permission etc.
        base = Image.new("RGB", (1440, 900), (40, 90, 160))
        ImageDraw.Draw(base).text((40, 40), "MOCK DEMO SCREEN", fill="white")
    return lambda: RawFrame(image=base, timestamp=datetime.now(timezone.utc).isoformat(),
                            frame_reference="mock")


def run_with_overlay(args, detector=None) -> None:
    """Function 2: overlay on the main thread, live loop / mock script in a worker."""
    from component1_screen_monitoring.src.overlay import ResponseOverlay
    from component1_screen_monitoring.src.capture import capture_frame
    from component1_screen_monitoring.src.response import (MOCK_SCRIPT, CaptureGate, E2ELatencyLog,
                                                           ResponseController, play_mock)
    log = logging.getLogger(__name__)
    latency = E2ELatencyLog(source="mock" if args.mock else "live",
                            strategy=None if args.mock else args.strategy)
    overlay = ResponseOverlay()
    controller = ResponseController(overlay, override=args.respond, latency=latency)
    overlay.on_dismiss = controller.on_dismiss
    gate = CaptureGate(controller, overlay)       # pass-through where capture exclusion works
    stop = threading.Event()

    def worker():
        try:
            if args.mock:
                script = MOCK_SCRIPT[:args.max_iterations] if args.max_iterations else MOCK_SCRIPT
                play_mock(controller, _mock_frame_fn(), script=script, should_stop=stop.is_set)
            else:
                run_monitoring_loop(detector, interval=args.interval,
                                    max_iterations=args.max_iterations, on_payload=_print_payload,
                                    strategy=args.strategy, on_result=gate.on_result,
                                    should_stop=stop.is_set, frame_gate=gate.allow,
                                    capture_fn=gate.wrap(lambda: capture_frame(monitor_index=1)))
        except Exception:
            log.exception("worker failed")
        finally:
            overlay.stop()

    def _stop(*_):
        stop.set()
        overlay.stop()

    signal.signal(signal.SIGINT, _stop)                   # Ctrl+C lands on the Tk main thread
    log.info("response mode: %s · overlay excluded from capture: %s", args.respond,
             "checked when first shown")
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    overlay.run()
    stop.set()
    thread.join(timeout=5)
    from component1_screen_monitoring.src import capture_exclusion
    log.info("overlay excluded from capture: %s (method: %s)%s", overlay.capture_excluded,
             capture_exclusion.last_method or "not attempted — overlay never shown",
             "" if overlay.capture_excluded or not overlay.exclusion_checked else
             f" · fallback '{gate.fallback}': {gate.skipped} frames skipped, {gate.peeks} peeks")
    saved = latency.save()
    print(latency.summary() + (f" → appended {saved} record(s) to "
                               f"{latency.path.relative_to(COMPONENT_ROOT.parent)}" if saved else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description="Component 1 screen monitoring runner")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="emit the mock payload (no model, no capture)")
    mode.add_argument("--once", action="store_true", help="capture + classify one real frame")
    mode.add_argument("--live", action="store_true", help="continuous capture loop")
    mode.add_argument("--benchmark", type=int, metavar="N",
                      help="time every region strategy on N live frames → "
                           "results/03_screen_eval_pretrained/latency_regions.json")
    ap.add_argument("--interval", type=float, default=CAPTURE_INTERVAL_S,
                    help=f"seconds between samples in --live (default: {CAPTURE_INTERVAL_S})")
    ap.add_argument("--max-iterations", type=int, default=None,
                    help="stop --live after N frames (default: run until Ctrl+C)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help=f"flag when P(violent) >= this (default: {THRESHOLD})")
    ap.add_argument("--mode", choices=list(MODES), default=DEFAULT_MODE,
                    help="default = whole screen only; strict = whole screen + tiles "
                         f"(default: {DEFAULT_MODE}; {', '.join(f'{m}={s}' for m, s in MODES.items())})")
    ap.add_argument("--strategy", choices=STRATEGIES, default=None,
                    help="advanced: region strategy, overrides --mode")
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                    help=f"checkpoint folder (default: {MODEL_DIR.relative_to(COMPONENT_ROOT)})")
    ap.add_argument("--respond", nargs="?", const="auto", choices=("auto", "blur", "block"),
                    default=None,
                    help="Function 2: show the blur/block overlay (with --live or --mock). "
                         "Bare --respond = auto (rules in config.py); blur/block force a mode. "
                         "Omit for Function 1 only.")
    ap.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = ap.parse_args()
    if args.respond and not (args.live or args.mock):
        ap.error("--respond works with --live or --mock")
    custom_strategy = args.strategy is not None
    args.strategy = args.strategy or MODES[args.mode]     # one strategy for every path below

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Third-party DEBUG noise would drown out the per-frame lines.
    for noisy in ("PIL", "urllib3", "filelock", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.mock:
        if args.respond:
            run_with_overlay(args)
        else:
            _print_payload(validate_payload(simulate_screen_capture()))
        return

    from component1_screen_monitoring.src.detector import ViolenceDetector

    try:
        detector = ViolenceDetector(args.model_dir, threshold=args.threshold)
    except FileNotFoundError as exc:
        sys.exit(f"ERROR: {exc}")
    logging.getLogger(__name__).info(
        "model %s loaded on %s — mode %s (strategy %s), threshold %s", detector.model_version,
        detector.device, "custom" if custom_strategy else args.mode, args.strategy, args.threshold)
    if args.respond:
        # Warm-up: the first MPS/CUDA inference is ~1 s slower; keep it out of the e2e latency log.
        from PIL import Image
        detector.predict_regions(Image.new("RGB", (1440, 900)), None, args.strategy)

    if args.benchmark:
        print(f"Capturing {args.benchmark} frames, then timing each strategy …", flush=True)
        results = benchmark_strategies(detector, args.benchmark)
        LATENCY_REGIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
        LATENCY_REGIONS_JSON.write_text(json.dumps(results, indent=2))
        print(f"Saved → {LATENCY_REGIONS_JSON.relative_to(COMPONENT_ROOT.parent)}")
    elif args.once:
        payload = capture_and_classify(detector, strategy=args.strategy)
        if payload is None:
            print("No payload — frame not flagged.")
        else:
            _print_payload(payload)
    elif args.respond:
        run_with_overlay(args, detector)
    else:
        run_monitoring_loop(detector, interval=args.interval, max_iterations=args.max_iterations,
                            on_payload=_print_payload, strategy=args.strategy)


if __name__ == "__main__":
    main()

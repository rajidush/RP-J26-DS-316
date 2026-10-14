"""
run_monitor.py — Command-line runner for Component 1 (Function 1: Screen Monitoring).

Modes
-----
    python component1_screen_monitoring/src/run_monitor.py --mock
    python component1_screen_monitoring/src/run_monitor.py --once
    python component1_screen_monitoring/src/run_monitor.py --live [--interval 2] [--max-iterations N]
    python component1_screen_monitoring/src/run_monitor.py --benchmark 20

Common options: --threshold (default: config.THRESHOLD), --strategy (region
strategy, default: config.REGION_STRATEGY), -v (show DEBUG logs, including
every not-flagged frame).

--benchmark N captures N live frames once, then times every region strategy
on those same frames and saves results/latency_regions.json.

Every printed payload has been validated against comp1_to_comp3.schema.json.
Stop --live with Ctrl+C.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import subprocess
import sys
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
    REGION_STRATEGY,
    THRESHOLD,
)
from component1_screen_monitoring.src.regions import STRATEGIES, motion_reference

LATENCY_REGIONS_JSON = COMPONENT_ROOT / "results" / "latency_regions.json"
#: Gap between benchmark captures — short, but long enough for video to move.
BENCHMARK_GAP_S = 0.5


def _print_payload(payload: dict) -> None:
    print(json.dumps(payload, indent=2), flush=True)


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
        "machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip() or platform.machine(),
        "frames": n,
        "frame_size": {"width": frames[0].width, "height": frames[0].height},
        "gap_between_captures_s": BENCHMARK_GAP_S,
        "note": "Same captured frames for every strategy; the motion crop is only added "
                "when part of the screen changed between consecutive frames.",
        "capture_ms": _stats(capture_ms),
        "strategies": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Component 1 screen monitoring runner")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="emit the mock payload (no model, no capture)")
    mode.add_argument("--once", action="store_true", help="capture + classify one real frame")
    mode.add_argument("--live", action="store_true", help="continuous capture loop")
    mode.add_argument("--benchmark", type=int, metavar="N",
                      help="time every region strategy on N live frames → results/latency_regions.json")
    ap.add_argument("--interval", type=float, default=CAPTURE_INTERVAL_S,
                    help=f"seconds between samples in --live (default: {CAPTURE_INTERVAL_S})")
    ap.add_argument("--max-iterations", type=int, default=None,
                    help="stop --live after N frames (default: run until Ctrl+C)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help=f"flag when P(violent) >= this (default: {THRESHOLD})")
    ap.add_argument("--strategy", choices=STRATEGIES, default=REGION_STRATEGY,
                    help=f"region strategy (default: {REGION_STRATEGY})")
    ap.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Third-party DEBUG noise would drown out the per-frame lines.
    for noisy in ("PIL", "urllib3", "filelock", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.mock:
        _print_payload(validate_payload(simulate_screen_capture()))
        return

    from component1_screen_monitoring.src.detector import ViolenceDetector

    detector = ViolenceDetector(threshold=args.threshold)
    logging.getLogger(__name__).info("model loaded on %s", detector.device)

    if args.benchmark:
        print(f"Capturing {args.benchmark} frames, then timing each strategy …", flush=True)
        results = benchmark_strategies(detector, args.benchmark)
        LATENCY_REGIONS_JSON.write_text(json.dumps(results, indent=2))
        print(f"Saved → {LATENCY_REGIONS_JSON.relative_to(COMPONENT_ROOT.parent)}")
    elif args.once:
        payload = capture_and_classify(detector, strategy=args.strategy)
        if payload is None:
            print("No payload — frame not flagged.")
        else:
            _print_payload(payload)
    else:
        run_monitoring_loop(detector, interval=args.interval, max_iterations=args.max_iterations,
                            on_payload=_print_payload, strategy=args.strategy)


if __name__ == "__main__":
    main()

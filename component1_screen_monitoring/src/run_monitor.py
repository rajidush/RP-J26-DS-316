"""
run_monitor.py — Command-line runner for Component 1 (Function 1: Screen Monitoring).

Modes
-----
    python component1_screen_monitoring/src/run_monitor.py --mock
    python component1_screen_monitoring/src/run_monitor.py --once
    python component1_screen_monitoring/src/run_monitor.py --live [--interval 2] [--max-iterations N]

Common options: --threshold (default: config.THRESHOLD), -v (show DEBUG logs,
including every not-flagged frame).

Every printed payload has been validated against comp1_to_comp3.schema.json.
Stop --live with Ctrl+C.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.component1 import (
    capture_and_classify,
    run_monitoring_loop,
    simulate_screen_capture,
    validate_payload,
)
from component1_screen_monitoring.src.config import CAPTURE_INTERVAL_S, THRESHOLD


def _print_payload(payload: dict) -> None:
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Component 1 screen monitoring runner")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="emit the mock payload (no model, no capture)")
    mode.add_argument("--once", action="store_true", help="capture + classify one real frame")
    mode.add_argument("--live", action="store_true", help="continuous capture loop")
    ap.add_argument("--interval", type=float, default=CAPTURE_INTERVAL_S,
                    help=f"seconds between samples in --live (default: {CAPTURE_INTERVAL_S})")
    ap.add_argument("--max-iterations", type=int, default=None,
                    help="stop --live after N frames (default: run until Ctrl+C)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help=f"flag when P(violent) >= this (default: {THRESHOLD})")
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

    if args.once:
        payload = capture_and_classify(detector)
        if payload is None:
            print("No payload — frame not flagged.")
        else:
            _print_payload(payload)
    else:
        run_monitoring_loop(detector, interval=args.interval,
                            max_iterations=args.max_iterations, on_payload=_print_payload)


if __name__ == "__main__":
    main()

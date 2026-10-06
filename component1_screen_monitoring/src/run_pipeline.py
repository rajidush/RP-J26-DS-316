"""
run_pipeline.py — End-to-end Component 1 pipeline: capture → detect → respond.

Function 1 (capture_loop + detector) runs in a worker thread; Function 2's
overlay runs the Tk main loop on the main thread.

Usage
-----
    python component1_screen_monitoring/src/run_pipeline.py --mode blur
    python component1_screen_monitoring/src/run_pipeline.py --mode block --hold 3 --max-frames 30

Stop with Ctrl-C in the terminal, or let --max-frames run out.
Esc hides a visible overlay (it re-triggers if the content is still flagged).
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.capture import (
    CONFIDENCE_THRESHOLD,
    capture_loop,
    make_detector_callback,
)
from component1_screen_monitoring.src.overlay import ResponseOverlay
from component1_screen_monitoring.src.response import RESPONSE_MODES, ResponseController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--mode", choices=RESPONSE_MODES, default="blur")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between captures")
    parser.add_argument("--hold", type=float, default=5.0, help="seconds the overlay stays up")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    overlay = ResponseOverlay()
    detector = make_detector_callback(threshold=args.threshold)
    controller = ResponseController(overlay, detector, mode=args.mode, hold_seconds=args.hold)

    def _worker() -> None:
        capture_loop(on_frame=controller.on_frame, interval=args.interval,
                     max_frames=args.max_frames, monitor_index=1)
        overlay.stop()

    # Ctrl-C lands on the main thread, which is inside the Tk main loop.
    signal.signal(signal.SIGINT, lambda *_: overlay.stop())

    threading.Thread(target=_worker, daemon=True).start()
    overlay.run()
    print("[pipeline] stopped.")


if __name__ == "__main__":
    main()

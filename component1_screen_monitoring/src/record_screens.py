"""
record_screens.py — Record labelled screen captures for the screen-style dataset.

!! This is the ONLY Component 1 tool that writes screen frames to disk. !!
run_monitor.py, the monitoring loop and the dashboard never save frames.

Use it to capture what the screen really looks like while you play RLVS clips
(from data/raw or the original RLVS videos) in a browser or media player at
different window sizes / positions — and, for the safe class, while doing
ordinary things (reading, shopping, watching non-violent clips).

    data/screens/recorded/<label>/<session>_<n>.png
    data/screens/recorded.csv            (filename, label, session, timestamp, monitor, size)

Usage
-----
    python component1_screen_monitoring/src/record_screens.py --label violence --interval 1 --max 60
    python component1_screen_monitoring/src/record_screens.py --label safe --max 60 --session browsing1

A 5-second countdown gives you time to switch to the window you want to record.
Stop early with Ctrl+C. Make sure nothing private is on screen while recording.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.capture import capture_frame, save_frame
from component1_screen_monitoring.src.config import SCREENS_DIR

LABELS = ("violence", "safe")
RECORDED_DIR = SCREENS_DIR / "recorded"
RECORDED_CSV = SCREENS_DIR / "recorded.csv"
CSV_FIELDS = ["filename", "label", "session", "timestamp", "monitor", "width", "height"]


def record(label: str, interval: float, max_frames: int, session: str, monitor: int,
           countdown: int = 5, out_dir: Path = RECORDED_DIR, csv_path: Path = RECORDED_CSV,
           capture_fn=capture_frame, sleep_fn=time.sleep) -> int:
    """Capture up to *max_frames* frames to disk. Returns how many were saved."""
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    if "_" in session:
        raise ValueError("session must not contain '_' (it separates <session>_<n>)")
    dest = out_dir / label
    dest.mkdir(parents=True, exist_ok=True)

    for s in range(countdown, 0, -1):
        print(f"  recording '{label}' in {s} …", flush=True)
        sleep_fn(1)

    new_csv = not csv_path.exists()
    saved = 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_csv:
            w.writeheader()
        try:
            while saved < max_frames:
                t0 = time.monotonic()
                frame = capture_fn(monitor_index=monitor)
                name = f"{session}_{saved}.png"
                if (dest / name).exists():
                    raise FileExistsError(f"{dest / name} exists — use a new --session")
                save_frame(frame, dest / name)
                w.writerow({"filename": f"{label}/{name}", "label": label, "session": session,
                            "timestamp": frame.timestamp, "monitor": monitor,
                            "width": frame.image.width, "height": frame.image.height})
                f.flush()
                saved += 1
                print(f"  saved {label}/{name}", flush=True)
                if saved < max_frames:
                    sleep_fn(max(0.0, interval - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            print("\n  stopped by user")
    print(f"Saved {saved} frame(s) → {dest}")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Record labelled screen captures to data/screens/recorded/. "
                    "WARNING: this is the only Component 1 tool that WRITES SCREEN FRAMES TO DISK "
                    "— make sure nothing private is on screen.")
    ap.add_argument("--label", required=True, choices=LABELS, help="class of what will be on screen")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between captures (default 1)")
    ap.add_argument("--max", type=int, default=60, dest="max_frames", help="frames to save (default 60)")
    ap.add_argument("--session", default=datetime.now().strftime("%Y%m%d-%H%M%S"),
                    help="name for this recording (default: timestamp); no underscores")
    ap.add_argument("--monitor", type=int, default=1, help="mss monitor index (1 = primary)")
    args = ap.parse_args()
    record(args.label, args.interval, args.max_frames, args.session, args.monitor)


if __name__ == "__main__":
    main()

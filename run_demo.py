#!/usr/bin/env python3
"""
run_demo.py -- THE single entry point for the whole system.

This is what you show the panel: one command, on one laptop, no setup
beyond `pip install -r requirements.txt`, no internet required.

    python run_demo.py

It's a thin wrapper around integration/end_to_end_pipeline.py so that
running the project doesn't require anyone to know the folder structure.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    pipeline = ROOT / "integration" / "end_to_end_pipeline.py"
    sys.exit(subprocess.call([sys.executable, str(pipeline), *sys.argv[1:]]))

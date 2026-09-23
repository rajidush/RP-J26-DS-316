#!/usr/bin/env bash
# One-command environment setup for the whole team.
# Usage: bash scripts/setup_env.sh
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Done. Activate with:  source .venv/bin/activate"
echo "Run the demo with:    python run_demo.py"
echo "Run all tests with:   pytest -v"

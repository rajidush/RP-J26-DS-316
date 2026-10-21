"""
config.py — Single source of tunable settings for Component 1.

Every module (detector, capture, component1, run_monitor) reads these values
instead of hard-coding its own, so the threshold is the same everywhere.
"""
from __future__ import annotations

from pathlib import Path

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPONENT_ROOT.parent

#: P(violent) at or above which a frame is flagged.
THRESHOLD: float = 0.5

#: Which crops of each screen frame are classified (see regions.py):
#: "full", "tiles", "motion" or "tiles+motion".
REGION_STRATEGY: str = "tiles+motion"

#: Seconds between screen samples in the live monitoring loop.
CAPTURE_INTERVAL_S: float = 2.0

#: Fine-tuned ViT checkpoint (not in git — see the component README).
MODEL_DIR: Path = COMPONENT_ROOT / "models" / "violence-vit-screen-ft-v1"

#: Screen-style dataset (scraped pages, composites, recordings) — see data/screens/README.md.
SCREENS_DIR: Path = COMPONENT_ROOT / "data" / "screens"

#: Experiment results, one folder per experiment — see results/README.md.
RESULTS_DIR: Path = COMPONENT_ROOT / "results"
RESULTS_WEEK1_BASELINE: Path = RESULTS_DIR / "00_week1_baseline"
RESULTS_RLVS_FINETUNE: Path = RESULTS_DIR / "01_rlvs_finetune"
RESULTS_PARAM_SWEEP: Path = RESULTS_DIR / "02_param_sweep"
RESULTS_SCREEN_EVAL: Path = RESULTS_DIR / "03_screen_eval_pretrained"
RESULTS_SCREEN_FINETUNE: Path = RESULTS_DIR / "04_screen_finetune"
RESULTS_MODEL_COMPARISON: Path = RESULTS_DIR / "05_model_comparison"
RESULTS_FUNCTION2: Path = RESULTS_DIR / "function2"

#: RLVS train/val/test split the checkpoints were trained with (notebook section 2).
SPLITS_CSV: Path = RESULTS_RLVS_FINETUNE / "splits.csv"

#: Shared contract every emitted payload is validated against.
SCHEMA_PATH: Path = REPO_ROOT / "docs" / "interface-contracts" / "comp1_to_comp3.schema.json"

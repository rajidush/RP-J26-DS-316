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

#: Detection modes — same model and threshold, different crops of each frame
#: (see regions.py). Numbers are model B at 0.5 on the screen test set,
#: results/05_model_comparison/model_comparison.md:
#:   "default" → "full":  whole screen only        — recall 0.548, false alarms 9.3 %,  ~51 ms/frame
#:   "strict"  → "tiles": whole screen + 2×2 tiles + centre crop
#:                                                  — recall 0.880, false alarms 18.9 %, ~228 ms/frame
MODES: dict[str, str] = {"default": "full", "strict": "tiles"}
DEFAULT_MODE: str = "default"

#: Which crops of each screen frame are classified: "full", "tiles", "motion" or
#: "tiles+motion". Follows DEFAULT_MODE; pass a mode / strategy to override.
REGION_STRATEGY: str = MODES[DEFAULT_MODE]

#: Seconds between screen samples in the live monitoring loop.
CAPTURE_INTERVAL_S: float = 2.0

#: Fine-tuned ViT checkpoint (not in git — see the component README): the screen
#: fine-tune (notebook section 15), chosen in results/05_model_comparison/.
MODEL_DIR: Path = COMPONENT_ROOT / "models" / "violence-vit-screen-ft-v2"

#: The TRUE pretrained model — jaranohaal/vit-base-violence-detection, converted to
#: Hugging Face format by src/convert_pretrained.py (no fine-tuning of our own).
PRETRAINED_MODEL_DIR: Path = COMPONENT_ROOT / "models" / "violence-vit-pretrained-hf"

#: V1 — the pretrained model fine-tuned on full-frame RLVS images (notebook §7; the
#: deployed model until the switch to ft-v2). Select either with --model-dir.
RLVS_FT_V1_MODEL_DIR: Path = COMPONENT_ROOT / "models" / "violence-vit-screen-ft-v1"

#: Where checkpoints come from: Google Drive, pp1_component1/checkpoints/<folder name>/.
MODEL_DOWNLOAD_HINT: str = "Google Drive: pp1_component1/checkpoints/{name}"


def model_version(model_dir: Path | str) -> str:
    """The name reported in payloads: the checkpoint folder name, e.g. 'violence-vit-screen-ft-v2'."""
    return Path(model_dir).name


#: model_version of the default model.
MODEL_VERSION: str = model_version(MODEL_DIR)

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

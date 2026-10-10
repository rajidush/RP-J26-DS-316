"""ONNX heads (Guardian GS-010). Skipped until `tools/export_onnx.py` has been run."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from component2_hate_speech_detection.src.engine.heads import DEFAULT_HEADS, hate_score_from_labels
from component2_hate_speech_detection.src.engine.onnx_backend import onnx_model_dir

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "component2_hate_speech_detection/evaluation/results/onnx_parity.json"
exported = all((onnx_model_dir(m) / "model.onnx").exists() for m in DEFAULT_HEADS)
pytestmark = pytest.mark.skipif(not exported, reason="run tools/export_onnx.py first")


def test_onnx_heads_rank_abuse_above_ordinary_chat():
    from component2_hate_speech_detection.src.engine.onnx_backend import OnnxBackend

    for model_id in DEFAULT_HEADS:
        # Dynabench is trained on attacks against a named group, so the harmful text names one.
        rows = OnnxBackend(model_id)(["muslims are vermin and should be wiped out", "see you at football practice"])
        harmful, benign = (hate_score_from_labels(r) for r in rows)
        assert harmful > 0.5 and benign < 0.2, (model_id, harmful, benign)


def test_engine_runs_without_importing_torch():
    code = (
        "import sys; from component2_hate_speech_detection.src.engine.analyzer import Analyzer; "
        "a = Analyzer(); a.scorer.warm_up(); v = a.analyze_text('they breed like vermin and should be caged', 9); "
        "print(v.rung, 'torch' in sys.modules, a.scorer.name)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    rung, torch_loaded, name = out.stdout.split()
    assert rung != "L0" and torch_loaded == "False" and "onnx" in name


def test_shipped_precision_changed_no_decisions():
    report = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert report["shipped_engine_changes_vs_fp32"] == 0
    for head in report["heads"].values():
        assert head["fp32_vs_torch_max_diff"] <= 1e-4

"""
Tests for src/detector.py.

Uses a tiny randomly-initialised ViT saved to a temp folder, so the tests run
in seconds and do NOT need the 330 MB fine-tuned checkpoint. One extra test
runs against the real checkpoint only if it has been downloaded.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detector import DEFAULT_MODEL_DIR, Detection, ViolenceDetector  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

    d = tmp_path_factory.mktemp("tiny_vit")
    cfg = ViTConfig(
        image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=64, num_labels=2,
        id2label={0: "safe", 1: "violent"}, label2id={"safe": 0, "violent": 1},
    )
    ViTForImageClassification(cfg).save_pretrained(d)
    ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(d)
    return d


def _img(color=(120, 30, 30), size=(640, 400), mode="RGB"):
    return Image.new(mode, size, color)


def test_predict_returns_valid_detection(tiny_model_dir):
    det = ViolenceDetector(tiny_model_dir, device="cpu")
    out = det.predict(_img())
    assert isinstance(out, Detection)
    assert 0.0 <= out.confidence <= 1.0
    assert out.category in {"violence", "safe"}
    assert out.flagged == (out.confidence >= det.threshold)
    assert (out.category == "violence") == out.flagged
    assert out.inference_ms > 0


def test_accepts_non_rgb_screenshots(tiny_model_dir):
    # mss / macOS screenshots can be RGBA; must not crash
    det = ViolenceDetector(tiny_model_dir, device="cpu")
    assert 0.0 <= det.predict(_img((10, 10, 10, 255), mode="RGBA")).confidence <= 1.0


def test_deterministic(tiny_model_dir):
    det = ViolenceDetector(tiny_model_dir, device="cpu")
    img = _img()
    assert det.predict(img).confidence == det.predict(img).confidence


@pytest.mark.parametrize("thr, expected", [(0.0, True), (1.0, False)])
def test_threshold_controls_flag(tiny_model_dir, thr, expected):
    det = ViolenceDetector(tiny_model_dir, threshold=thr, device="cpu")
    out = det.predict(_img())
    if thr == 1.0 and out.confidence == 1.0:
        pytest.skip("degenerate random model")
    assert out.flagged is expected


def test_missing_model_dir_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model found"):
        ViolenceDetector(tmp_path / "nope")


def test_bad_threshold_rejected(tiny_model_dir):
    with pytest.raises(ValueError):
        ViolenceDetector(tiny_model_dir, threshold=1.5)


@pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "config.json").exists(),
                    reason="fine-tuned checkpoint not downloaded")
def test_real_checkpoint_loads_and_predicts():
    det = ViolenceDetector()
    assert det.model.config.label2id.get("violent") == 1
    assert 0.0 <= det.predict(_img()).confidence <= 1.0

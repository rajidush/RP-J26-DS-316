"""
Tests for region-based inference: src/regions.py and ViolenceDetector.predict_regions.

Synthetic frames + a tiny randomly-initialised ViT — no real checkpoint needed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from regions import (  # noqa: E402
    STRATEGIES,
    motion_box,
    motion_reference,
    region_boxes,
    tile_boxes,
)

W, H = 1440, 900


def _background(seed=0):
    """Static, textured 'desktop' so motion detection has something to ignore."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(90, 160, (H, W, 3), dtype=np.uint8))


def _with_square(bg, x, y, size=240, color=(250, 20, 20)):
    img = bg.copy()
    img.paste(color, (x, y, x + size, y + size))
    return img


def _inside(box, size=(W, H)):
    x0, y0, x1, y1 = box
    return 0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]


# ---------------------------------------------------------------------------
# Crop counts + geometry
# ---------------------------------------------------------------------------

def test_crop_count_per_strategy():
    bg = _background()
    moved = _with_square(bg, 600, 300)
    assert len(region_boxes(bg, None, "full")) == 1
    assert len(region_boxes(bg, None, "tiles")) == 6            # full + 2x2 + centre
    assert len(region_boxes(bg, None, "motion")) == 1           # no previous frame → no motion crop
    assert len(region_boxes(moved, bg, "motion")) == 2
    assert len(region_boxes(bg, bg, "tiles+motion")) == 6       # static → no motion crop
    assert len(region_boxes(moved, bg, "tiles+motion")) == 7


def test_full_frame_is_always_first():
    for strategy in STRATEGIES:
        name, box = region_boxes(_background(), None, strategy)[0]
        assert (name, box) == ("full", (0, 0, W, H))


def test_tiles_overlap_and_cover_frame():
    tiles = dict(tile_boxes((W, H)))
    assert set(tiles) == {"top_left", "top_right", "bottom_left", "bottom_right", "centre"}
    tl, br = tiles["top_left"], tiles["bottom_right"]
    assert tl[:2] == (0, 0) and br[2:] == (W, H)                # corners reach the edges
    assert tl[2] > br[0] and tl[3] > br[1]                      # neighbouring tiles overlap
    assert tl[2] - tl[0] == round(0.6 * W)


@pytest.mark.parametrize("size", [(1440, 900), (2560, 1600), (801, 333), (64, 64)])
def test_boxes_stay_inside_frame(size):
    bg = Image.new("RGB", size, (100, 100, 100))
    edge = bg.copy()
    edge.paste((255, 255, 255), (size[0] * 3 // 4, size[1] * 3 // 4, size[0], size[1]))
    for strategy in STRATEGIES:
        for name, box in region_boxes(edge, bg, strategy):
            assert _inside(box, size), (strategy, name, box)


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        region_boxes(_background(), None, "everything")


# ---------------------------------------------------------------------------
# Motion region
# ---------------------------------------------------------------------------

def test_motion_finds_changing_square_on_static_background():
    bg = _background()
    box = motion_box(_with_square(bg, 600, 300), bg)
    assert box is not None and _inside(box)
    x0, y0, x1, y1 = box
    # contains the square (600..840, 300..540) and is not much bigger than it + ~10% padding
    assert x0 <= 600 and y0 <= 300 and x1 >= 840 and y1 >= 540
    assert (x1 - x0) < 240 * 1.6 and (y1 - y0) < 240 * 1.6


def test_motion_box_tracks_a_moving_square_and_picks_the_largest_change():
    bg = _background()
    prev = _with_square(bg, 100, 100)
    cur = _with_square(_with_square(bg, 100, 100), 900, 500, size=320)   # new, bigger change
    cur.paste((0, 255, 0), (50, 800, 80, 830))                            # small extra change
    x0, y0, x1, y1 = motion_box(cur, prev)
    assert x0 <= 900 <= x1 and y0 <= 500 <= y1


def test_no_motion_for_static_frame_or_tiny_change():
    bg = _background()
    assert motion_box(bg, bg) is None
    assert motion_box(bg, None) is None
    speck = bg.copy()
    speck.paste((255, 255, 255), (700, 400, 704, 404))                   # cursor-sized
    assert motion_box(speck, bg) is None


def test_motion_works_from_small_reference_copy():
    bg = _background()
    cur = _with_square(bg, 600, 300)
    ref = motion_reference(bg)
    assert ref.mode == "L" and ref.width == 160                           # what the loop keeps
    assert motion_box(cur, ref) == motion_box(cur, bg)


# ---------------------------------------------------------------------------
# ViolenceDetector.predict_regions (tiny random model)
# ---------------------------------------------------------------------------

pytest.importorskip("torch")
pytest.importorskip("transformers")
from detector import ViolenceDetector  # noqa: E402


@pytest.fixture(scope="module")
def detector(tmp_path_factory):
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

    d = tmp_path_factory.mktemp("tiny_vit_regions")
    cfg = ViTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=64, num_labels=2,
                    id2label={0: "safe", 1: "violent"}, label2id={"safe": 0, "violent": 1})
    ViTForImageClassification(cfg).save_pretrained(d)
    ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(d)
    return ViolenceDetector(d, device="cpu")


@pytest.mark.parametrize("strategy, with_motion, expected", [
    ("full", False, 1), ("tiles", False, 6), ("motion", True, 2),
    ("tiles+motion", False, 6), ("tiles+motion", True, 7),
])
def test_predict_regions_crop_count(detector, strategy, with_motion, expected):
    bg = _background()
    cur = _with_square(bg, 600, 300) if with_motion else bg
    assert detector.predict_regions(cur, bg, strategy).n_crops == expected


def test_predict_regions_confidence_is_max_of_crops(detector):
    bg = _background()
    cur = _with_square(bg, 600, 300)
    det = detector.predict_regions(cur, bg, "tiles+motion")

    per_crop = {name: detector.predict(cur.crop(box)).confidence
                for name, box in region_boxes(cur, bg, "tiles+motion")}
    assert det.confidence == pytest.approx(max(per_crop.values()), abs=2e-4)
    assert per_crop[det.region] == pytest.approx(det.confidence, abs=2e-4)
    assert det.region_box == dict(region_boxes(cur, bg, "tiles+motion"))[det.region]
    assert det.flagged == (det.confidence >= detector.threshold)
    assert det.inference_ms > 0


def test_predict_regions_full_matches_predict(detector):
    img = _background(seed=3)
    assert detector.predict_regions(img, None, "full").confidence == pytest.approx(
        detector.predict(img).confidence, abs=2e-4)

"""Tests for src/eval_screens.py on a tiny fake dataset with a fake detector."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.eval_screens import (
    collect_dataset,
    compute_metrics,
    evaluate,
    to_markdown,
)

RED, GREY = (220, 0, 0), (128, 128, 128)


@pytest.fixture
def fake_screens(tmp_path):
    """3 scraped (safe), composites 2 violent + 2 safe, recorded 1 violent + 1 safe."""
    def save(sub, name, color):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 40), color).save(tmp_path / sub / name)

    for i in range(3):
        save("scraped_pages", f"site-{i}_0.png", GREY)
    save("composite/violence", "site-0_0.png", RED)
    save("composite/violence", "site-1_0.png", GREY)       # the 'missed' violent one
    save("composite/non_violence", "site-0_0.png", GREY)
    save("composite/non_violence", "site-1_0.png", RED)    # a false positive
    save("recorded/violence", "s1_0.png", RED)
    save("recorded/safe", "s1_0.png", GREY)
    (tmp_path / "scraped_pages" / "notes.txt").write_text("not an image")
    return tmp_path


class ColorDetector:
    """Flags images whose top-left pixel is red; reports 6 crops for 'tiles'."""

    threshold = 0.5

    def predict_regions(self, image, prev_image=None, strategy="tiles"):
        p = 0.9 if image.convert("RGB").getpixel((0, 0)) == RED else 0.1
        return SimpleNamespace(confidence=p, flagged=p >= self.threshold, region="full",
                               n_crops=1 if strategy == "full" else 6, inference_ms=1.0)


def test_collect_dataset_labels_and_sources(fake_screens):
    items = collect_dataset(fake_screens)
    assert len(items) == 9                                   # .txt ignored
    by = {(i["source"], i["label"]) for i in items}
    assert by == {("scraped", 0), ("composite", 1), ("composite", 0), ("recorded", 1), ("recorded", 0)}
    assert sum(i["label"] for i in items) == 3


def test_collect_dataset_missing_folders(tmp_path):
    assert collect_dataset(tmp_path) == []


def test_compute_metrics_known_values():
    m = compute_metrics([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])
    assert m["confusion"] == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}
    assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["false_positive_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_compute_metrics_undefined_cases():
    only_safe = compute_metrics([0, 0], [0, 1])               # like the scraped pages
    assert only_safe["precision"] == 0.0 and only_safe["recall"] is None and only_safe["f1"] is None
    assert only_safe["false_positive_rate"] == 0.5
    none_found = compute_metrics([1, 1], [0, 0])
    assert none_found["precision"] is None and none_found["recall"] == 0.0 and none_found["f1"] == 0.0


def test_evaluate_per_source_and_overall(fake_screens):
    metrics, rows = evaluate(ColorDetector(), collect_dataset(fake_screens))
    assert set(metrics) == {"full", "tiles"}
    assert len(rows) == 9 * 2
    full = metrics["full"]
    assert full["scraped"]["accuracy"] == 1.0 and full["scraped"]["recall"] is None
    assert full["composite"]["confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert full["recorded"]["f1"] == 1.0
    assert full["overall"]["n"] == 9
    assert full["overall"]["confusion"] == {"tp": 2, "fp": 1, "tn": 5, "fn": 1}
    assert {r["n_crops"] for r in rows if r["strategy"] == "tiles"} == {6}


def test_markdown_report_has_every_source_and_strategy(fake_screens):
    metrics, _ = evaluate(ColorDetector(), collect_dataset(fake_screens))
    md = to_markdown({"model_dir": "m", "threshold": 0.5, "device": "cpu",
                      "evaluated_at": "now", "counts": {"scraped": 3}, "metrics": metrics})
    for token in ("| scraped | full", "| composite | tiles", "| overall | full", "—"):
        assert token in md

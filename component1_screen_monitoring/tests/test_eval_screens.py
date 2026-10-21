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


def test_pick_examples_mixes_missed_and_false_alarms(fake_screens):
    from component1_screen_monitoring.src.eval_screens import pick_examples
    _, rows = evaluate(ColorDetector(), collect_dataset(fake_screens))
    ex = pick_examples(rows, n=8)
    assert ex and all(r["pred"] != r["label"] for r in ex)
    assert {r["label"] for r in ex} == {0, 1}                 # both kinds of error
    assert len({r["file"] for r in ex}) == len(ex)            # one entry per image


def test_save_examples_blurs_violent_images(fake_screens, tmp_path):
    from component1_screen_monitoring.src.eval_screens import save_examples
    sharp = tmp_path / "violent.png"
    img = Image.new("RGB", (400, 250), "white")
    img.paste((0, 0, 0), (0, 0, 200, 250))                    # hard black/white edge
    img.save(sharp)
    out = tmp_path / "grid.png"
    save_examples([{"file": str(sharp), "label": 1, "pred": 0, "source": "composite",
                    "strategy": "full", "p_violent": 0.01}], out, cols=1)
    grid = Image.open(out).convert("L")
    row = [grid.getpixel((x, 120)) for x in range(185, 216)]
    assert max(row) - min(row) < 250 and any(30 < v < 225 for v in row)   # edge smeared


# ---------------------------------------------------------------------------
# --subset clean: only composites whose RLVS frame is from the val/test split
# ---------------------------------------------------------------------------

@pytest.fixture
def screens_with_provenance(fake_screens):
    """composites.csv + an RLVS splits.csv for the 4 fake composites."""
    import csv
    with open(fake_screens / "composites.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "class", "background", "source_frame"])
        w.writerow(["violence/site-0_0.png", "violence", "site-0_0.png", "raw/violence/1.jpg"])          # train
        w.writerow(["violence/site-1_0.png", "violence", "site-1_0.png", "raw/violence/2.jpg"])          # test
        w.writerow(["non_violence/site-0_0.png", "non_violence", "site-0_0.png", "raw/non_violence/1.jpg"])  # val
        w.writerow(["non_violence/site-1_0.png", "non_violence", "site-1_0.png", "raw/non_violence/2.jpg"])  # train
    splits = fake_screens / "splits.csv"
    splits.write_text("path,label,split\nviolence /1.jpg,1,train\nviolence /2.jpg,1,test\n"
                      "non_violence /1.jpg,0,val\nnon_violence /2.jpg,0,train\n")
    return fake_screens, splits


def test_subset_all_is_unchanged(screens_with_provenance):
    from component1_screen_monitoring.src.eval_screens import filter_subset
    root, splits = screens_with_provenance
    items = collect_dataset(root)
    assert filter_subset(items, "all", root, splits) == items


def test_subset_clean_keeps_val_test_composites_and_pages(screens_with_provenance):
    from component1_screen_monitoring.src.eval_screens import filter_subset
    root, splits = screens_with_provenance
    clean = filter_subset(collect_dataset(root), "clean", root, splits)
    kept = {(i["source"], Path(i["path"]).relative_to(root).as_posix()) for i in clean}
    assert kept == {("scraped", f"scraped_pages/site-{i}_0.png") for i in range(3)} | {
        ("composite", "composite/violence/site-1_0.png"),        # RLVS test frame
        ("composite", "composite/non_violence/site-0_0.png"),    # RLVS val frame
    }                                                            # train-split frames and recorded/ dropped


def test_subset_clean_fails_on_unknown_composite(screens_with_provenance):
    from component1_screen_monitoring.src.eval_screens import filter_subset
    root, splits = screens_with_provenance
    Image.new("RGB", (64, 40), RED).save(root / "composite" / "violence" / "unlogged_0.png")
    with pytest.raises(ValueError, match="not in composites.csv"):
        filter_subset(collect_dataset(root), "clean", root, splits)


def test_subset_unknown_name(screens_with_provenance):
    from component1_screen_monitoring.src.eval_screens import filter_subset
    root, splits = screens_with_provenance
    with pytest.raises(ValueError, match="unknown subset"):
        filter_subset(collect_dataset(root), "dirty", root, splits)


def test_clean_markdown_says_subset(fake_screens):
    metrics, _ = evaluate(ColorDetector(), collect_dataset(fake_screens))
    md = to_markdown({"model_dir": "m", "threshold": 0.5, "device": "cpu", "subset": "clean",
                      "evaluated_at": "now", "counts": {}, "metrics": metrics})
    assert "Subset: clean" in md


# ---------------------------------------------------------------------------
# Output locations: one experiment folder per mode, never the results/ root
# ---------------------------------------------------------------------------

def test_default_output_per_mode():
    from component1_screen_monitoring.src.config import (RESULTS_DIR, RESULTS_MODEL_COMPARISON,
                                                        RESULTS_SCREEN_EVAL)
    from component1_screen_monitoring.src.eval_screens import default_output
    assert default_output(None, 0.5, "all") == (RESULTS_SCREEN_EVAL, "screen_eval")
    assert default_output(None, 0.5, "clean") == (RESULTS_SCREEN_EVAL, "screen_eval_clean")
    d, stem = default_output([Path("models/violence-vit-screen-ft-v2")], 0.25, "clean")
    assert d == RESULTS_MODEL_COMPARISON / "runs" and stem == "violence-vit-screen-ft-v2_t0.25_clean"
    d, stem = default_output([Path("m/a"), Path("m/b")], 0.5, "all")
    assert stem == "a+b_t0.50_full"
    assert RESULTS_DIR not in {default_output(None, 0.5, "all")[0], d}


def test_results_paths_point_into_experiment_folders():
    from component1_screen_monitoring.src import config
    assert config.SPLITS_CSV == config.RESULTS_DIR / "01_rlvs_finetune" / "splits.csv"
    assert config.SPLITS_CSV.exists()
    for name in ("RESULTS_WEEK1_BASELINE", "RESULTS_RLVS_FINETUNE", "RESULTS_PARAM_SWEEP",
                 "RESULTS_SCREEN_EVAL", "RESULTS_SCREEN_FINETUNE", "RESULTS_MODEL_COMPARISON",
                 "RESULTS_FUNCTION2"):
        assert getattr(config, name).parent == config.RESULTS_DIR

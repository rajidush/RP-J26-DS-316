"""Tests for src/make_composites.py and the offline helpers of src/scrape_screenshots.py."""
import csv
import hashlib
import random
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from make_composites import LAYOUTS, compose, make_composites, source_id_of  # noqa: E402
from scrape_screenshots import read_urls, source_id  # noqa: E402

PAGE = (1280, 800)


def _page(color=(245, 245, 245), size=PAGE):
    return Image.new("RGB", size, color)


def _frame(color=(200, 0, 0)):
    return Image.new("RGB", (416, 416), color)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("seed", range(15))
def test_output_size_and_box_inside(layout, seed):
    img, info = compose(_page(), _frame(), random.Random(seed), layout)
    assert img.size == PAGE
    x0, y0, x1, y1 = info["box"]
    assert 0 <= x0 < x1 <= PAGE[0] and 0 <= y0 < y1 <= PAGE[1]
    assert info["layout"] == layout


@pytest.mark.parametrize("seed", range(10))
def test_box_inside_on_small_background(seed):
    small = (400, 260)
    img, info = compose(_page(size=small), _frame(), random.Random(seed))
    x0, y0, x1, y1 = info["box"]
    assert img.size == small and 0 <= x0 < x1 <= small[0] and 0 <= y0 < y1 <= small[1]


def test_frame_content_is_inside_box():
    img, info = compose(_page(), _frame((200, 0, 0)), random.Random(3), "article_image")
    x0, y0, x1, y1 = info["box"]
    assert img.getpixel(((x0 + x1) // 2, (y0 + y1) // 2)) == (200, 0, 0)
    assert img.getpixel((2, 2)) == (245, 245, 245)            # page left alone outside


def test_video_box_size_range():
    for seed in range(30):
        _, info = compose(_page(), _frame(), random.Random(seed), "video_player")
        w = info["box"][2] - info["box"][0]
        assert 0.20 * PAGE[0] - 1 <= w <= 0.60 * PAGE[0] + 1


@pytest.fixture
def tiny_dataset(tmp_path):
    bg, raw = tmp_path / "scraped_pages", tmp_path / "raw"
    bg.mkdir()
    for sid in ("example-com-aaaaaa", "news-org-bbbbbb"):
        for n in range(2):
            _page((240 - 20 * n, 240, 240)).save(bg / f"{sid}_{n}.png")
    for cls, color in (("violence", (200, 0, 0)), ("non_violence", (0, 0, 200))):
        (raw / cls).mkdir(parents=True)
        for i in range(3):
            _frame(tuple(min(255, c + 10 * i) for c in color)).save(raw / cls / f"{i}.jpg")
    return tmp_path


def _run(root, seed, out="composite"):
    return make_composites(root / "scraped_pages", root / "raw", root / out,
                           root / f"{out}.csv", per_background=2, seed=seed)


def test_batch_counts_prefixes_and_log(tiny_dataset):
    rows = _run(tiny_dataset, seed=1)
    assert len(rows) == 4 * 2 * 2                              # 4 pages × 2 per page × 2 classes
    for cls in ("violence", "non_violence"):
        files = sorted((tiny_dataset / "composite" / cls).glob("*.png"))
        assert len(files) == 8
        assert {source_id_of(f) for f in files} == {"example-com-aaaaaa", "news-org-bbbbbb"}
    with open(tiny_dataset / "composite.csv") as f:
        logged = list(csv.DictReader(f))
    assert len(logged) == 16
    for r in logged:
        assert source_id_of(Path(r["filename"])) == source_id_of(Path(r["background"]))
        assert r["source_frame"].startswith(f"raw/{r['class']}/")


def _digest(folder):
    return {p.relative_to(folder).as_posix(): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*.png"))}


def test_deterministic_with_seed(tiny_dataset):
    rows_a = _run(tiny_dataset, seed=7, out="a")
    rows_b = _run(tiny_dataset, seed=7, out="b")
    rows_c = _run(tiny_dataset, seed=8, out="c")
    assert rows_a == rows_b
    assert _digest(tiny_dataset / "a") == _digest(tiny_dataset / "b")
    assert _digest(tiny_dataset / "a") != _digest(tiny_dataset / "c")


def test_rerun_replaces_previous_output(tiny_dataset):
    _run(tiny_dataset, seed=1)
    make_composites(tiny_dataset / "scraped_pages", tiny_dataset / "raw",
                    tiny_dataset / "composite", tiny_dataset / "composite.csv",
                    per_background=1, seed=1)
    assert len(list((tiny_dataset / "composite" / "violence").glob("*.png"))) == 4


# --- scraper helpers (no network) ------------------------------------------

def test_read_urls_skips_comments_blanks_and_duplicates(tmp_path):
    f = tmp_path / "urls.txt"
    f.write_text("# header\nhttps://a.org/x\n\nhttps://b.org/  # trailing comment\nhttps://a.org/x\n")
    assert read_urls(f) == ["https://a.org/x", "https://b.org/"]


def test_source_id_stable_and_has_no_underscore():
    sid = source_id("https://www.en_test.example.org/wiki/Page_1")
    assert sid == source_id("https://www.en_test.example.org/wiki/Page_1")
    assert "_" not in sid and sid.startswith("en-test-example-org-")
    assert source_id("https://a.org/1") != source_id("https://a.org/2")
    assert source_id_of(Path(f"{sid}_2.png")) == sid

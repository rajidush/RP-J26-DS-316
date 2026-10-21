"""
Tests for the screen TRAINING set: URL overlap check, group split, tile labels,
RLVS frame grouping, train/val integrity, and an end-to-end build on fake data.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_leakage import rlvs_key_of, split_integrity  # noqa: E402
from dataset_checks import (assert_disjoint_url_lists, domain_overlap, normalize_url,  # noqa: E402
                            read_url_list, site_of, url_overlap)
from make_composites import (group_split, make_train_set, overlap_share, size_bucket,  # noqa: E402
                             tile_label)
from rlvs_groups import group_frames  # noqa: E402

SCREENS = Path(__file__).resolve().parents[1] / "data" / "screens"


# ---------------------------------------------------------------------------
# URL overlap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", [
    ("https://www.Example.com/a/", "http://example.com/a"),
    ("https://example.com", "https://example.com/"),
    ("https://example.com/a#section", "https://example.com/a"),
    ("https://example.com:443/a", "https://example.com/a"),
])
def test_normalize_url_treats_variants_as_same(a, b):
    assert normalize_url(a) == normalize_url(b)


def test_normalize_url_keeps_different_pages_apart():
    assert normalize_url("https://a.com/x") != normalize_url("https://a.com/y")
    assert normalize_url("https://a.com/x?p=1") != normalize_url("https://a.com/x?p=2")


@pytest.mark.parametrize("url, site", [
    ("https://docs.djangoproject.com/en/stable/", "djangoproject.com"),
    ("https://www.bbc.co.uk/news", "bbc.co.uk"),
    ("https://www.metoffice.gov.uk/", "metoffice.gov.uk"),
    ("https://kids.nationalgeographic.com/", "nationalgeographic.com"),
])
def test_site_of(url, site):
    assert site_of(url) == site


def test_overlap_detection():
    train = ["https://a.com/x", "https://www.b.com/y/", "https://c.org/z"]
    test = ["http://b.com/y", "https://sub.c.org/other"]
    assert url_overlap(train, test) == {"b.com/y"}
    assert domain_overlap(train, test) == {"b.com", "c.org"}


def _write(path, urls):
    path.write_text("# comment\n" + "\n".join(urls) + "\n")
    return path


def test_assert_disjoint_fails_on_shared_url(tmp_path):
    tr = _write(tmp_path / "train.txt", ["https://new.org/", "https://www.shared.com/page/"])
    te = _write(tmp_path / "test.txt", ["https://shared.com/page"])
    with pytest.raises(ValueError, match="1 shared URL"):
        assert_disjoint_url_lists(tr, te)


def test_assert_disjoint_fails_on_shared_website(tmp_path):
    tr = _write(tmp_path / "train.txt", ["https://docs.shared.com/a"])
    te = _write(tmp_path / "test.txt", ["https://shared.com/b"])
    with pytest.raises(ValueError, match="1 shared website"):
        assert_disjoint_url_lists(tr, te)


def test_assert_disjoint_passes(tmp_path):
    assert_disjoint_url_lists(_write(tmp_path / "a.txt", ["https://a.com/"]),
                              _write(tmp_path / "b.txt", ["https://b.com/"]))


@pytest.mark.skipif(not (SCREENS / "urls_train.txt").exists(), reason="train URL list not present")
def test_real_train_and_test_url_lists_are_disjoint():
    assert_disjoint_url_lists(SCREENS / "urls_train.txt", SCREENS / "urls.txt")
    n = len(read_url_list(SCREENS / "urls_train.txt"))
    assert 150 <= n <= 200


# ---------------------------------------------------------------------------
# Group split
# ---------------------------------------------------------------------------

def _groups(n_groups=200, seed=0):
    rng = np.random.default_rng(seed)
    return [f"g{i}" for i in range(n_groups) for _ in range(int(rng.integers(1, 8)))]


def test_group_split_no_group_in_both_and_about_15_percent():
    items = _groups()
    split = group_split(items, 0.15, seed=3)
    assert set(split) == set(items) and set(split.values()) == {"train", "val"}
    val_share = sum(split[g] == "val" for g in items) / len(items)
    assert 0.13 <= val_share <= 0.18


def test_group_split_is_deterministic_and_seed_dependent():
    items = _groups()
    assert group_split(items, 0.15, 1) == group_split(items, 0.15, 1)
    assert group_split(items, 0.15, 1) != group_split(items, 0.15, 2)


def test_split_integrity_detects_leak():
    ok = [{"split": "train", "page_id": "p1", "site": "a", "video_group": "v1"},
          {"split": "val", "page_id": "p2", "site": "b", "video_group": "v2"}]
    assert split_integrity(ok) == {"page_id": [], "site": [], "video_group": []}
    leak = ok + [{"split": "val", "page_id": "p3", "site": "c", "video_group": "v1"}]
    assert split_integrity(leak)["video_group"] == ["v1"]


# ---------------------------------------------------------------------------
# Tiles, size buckets, frame grouping
# ---------------------------------------------------------------------------

def test_size_bucket():
    assert size_bucket((0, 0, 380, 10), 1280) == "under30"
    assert size_bucket((0, 0, 384, 10), 1280) == "30-45"
    assert size_bucket((0, 0, 576, 10), 1280) == "45plus"


def test_tile_label_by_overlap():
    box = (100, 100, 300, 200)                                   # pasted frame
    assert overlap_share(box, (0, 0, 400, 400)) == 1.0
    assert tile_label(1, box, (0, 0, 400, 400)) == 1             # contains the frame
    assert tile_label(1, box, (500, 0, 900, 400)) == 0           # no overlap → safe view
    assert tile_label(1, box, (200, 0, 600, 400)) is None        # half the frame → ambiguous, skipped
    assert tile_label(0, box, (0, 0, 400, 400)) == 0             # safe composite → always safe


def test_group_frames_same_class_and_transitive():
    a = np.array([1.0, 0.0]); b = np.array([0.99, 0.141]); c = np.array([0.96, 0.28])
    feats = np.stack([a, b, c, a]) / np.linalg.norm(np.stack([a, b, c, a]), axis=1, keepdims=True)
    classes = np.array(["v", "v", "v", "n"])
    g = group_frames([feats], [0.985], classes)
    assert g[0] == g[1] == g[2]                                   # a~b, b~c → one group
    assert g[3] != g[0]                                           # identical, but other class


def test_rlvs_key_of():
    assert rlvs_key_of("raw/violence/1045.jpg") == "violence /1045.jpg"


# ---------------------------------------------------------------------------
# End-to-end on fake data
# ---------------------------------------------------------------------------

@pytest.fixture
def fake(tmp_path):
    pages = tmp_path / "train_pages"; pages.mkdir()
    log = tmp_path / "train_sources.csv"
    with open(log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "url", "timestamp", "status"]); w.writeheader()
        for i in range(12):
            sid = f"site{i}-com-{i:06x}"
            for n in range(2):
                Image.new("RGB", (640, 400), (230, 230, 230)).save(pages / f"{sid}_{n}.png")
                w.writerow({"filename": f"{sid}_{n}.png", "url": f"https://site{i}.com/p",
                            "timestamp": "t", "status": "ok"})
    root = tmp_path / "rlvs"
    rows = []
    for cls, color in (("violence", (200, 0, 0)), ("non_violence", (0, 0, 200))):
        (root / cls).mkdir(parents=True)
        for i in range(60):
            Image.new("RGB", (64, 64), color).save(root / cls / f"{i}.jpg")
            rows.append({"path": f"{cls} /{i}.jpg", "class": cls,
                         "rlvs_split": "test" if i >= 50 else "train", "video_group": f"{cls[0]}{i // 3}",
                         "in_test_composites": 0, "near_test": 1 if i in (0, 1, 2) else 0,
                         "max_cls_to_test": 0, "max_thumb_to_test": 0})
    frames_csv = tmp_path / "frames.csv"
    with open(frames_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return pages, log, frames_csv, root, tmp_path


def test_make_train_set_end_to_end(fake):
    pages, log, frames_csv, root, tmp = fake
    out = make_train_set(pages, log, frames_csv, root, tmp / "train_composites",
                         csv_path=tmp / "screen_train.csv", n_per_class=40, seed=5)
    comps = [r for r in out if not r["is_tile"]]
    assert sum(r["label"] == 1 for r in comps) == sum(r["label"] == 0 for r in comps) == 40
    assert {r["split"] for r in comps} == {"train", "val"}
    assert split_integrity(out) == {"page_id": [], "site": [], "video_group": []}
    frames = {r["path"]: r for r in csv.DictReader(open(frames_csv))}
    for r in out:
        assert frames[r["source_frame"]]["rlvs_split"] == "train"      # TRAIN split only
        assert frames[r["source_frame"]]["near_test"] == "0"           # never near the test set
        assert (tmp / r["path"]).exists() and r["group"] == f"{r['page_id']}|{r['video_group']}"
    tiles = [r for r in out if r["is_tile"]]
    assert tiles and all(r["label"] == 0 for r in tiles if "non_violence" in r["path"])
    with open(tmp / "screen_train.csv") as f:
        assert next(csv.reader(f))[:5] == ["path", "label", "group", "size_bucket", "is_tile"]


def test_make_train_set_refuses_test_folders(fake):
    pages, log, frames_csv, root, tmp = fake
    from make_composites import OUT_ROOT
    with pytest.raises(ValueError, match="TEST set"):
        make_train_set(pages, log, frames_csv, root, OUT_ROOT / "violence", csv_path=tmp / "x.csv")

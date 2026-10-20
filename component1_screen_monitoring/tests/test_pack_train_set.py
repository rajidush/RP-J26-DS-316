"""Tests for src/pack_train_set.py: the Colab zip holds the training set and nothing else."""
import csv
import json
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pack_train_set import check_arcname, pack  # noqa: E402

FIELDS = ["path", "label", "group", "size_bucket", "is_tile", "split"]


@pytest.fixture
def screens(tmp_path):
    """A fake data/screens/ with a training set AND the things that must never be packed."""
    rows = []
    for cls, label in (("violence", 1), ("non_violence", 0)):
        for i in range(3):
            for kind, is_tile in (("composites", 0), ("tiles", 1)):
                p = f"train_composites/{kind}/{cls}/p{i}_0{'_tl' if is_tile else ''}.png"
                (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 20), (label * 200, 0, 0)).save(tmp_path / p)
                rows.append({"path": p, "label": label, "group": f"p{i}|v{i}", "size_bucket": "30-45",
                             "is_tile": is_tile, "split": "val" if i == 2 else "train"})
    with open(tmp_path / "screen_train.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    (tmp_path / "rlvs_frames.csv").write_text("path,class,rlvs_split,near_test\nviolence /1.jpg,violence,train,0\n")
    # never packed: TEST set, raw pages, weights, stray files not in screen_train.csv
    for p in ("composite/violence/t_0.png", "composite/non_violence/t_0.png", "scraped_pages/t_0.png",
              "train_pages/p0_0.png"):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(tmp_path / p)
    for p in ("composites.csv", "sources.csv", "urls.txt", "train_composites/model.safetensors",
              "train_composites/composites/violence/stray.png"):
        (tmp_path / p).write_text("x")
    return tmp_path


def test_pack_contains_exactly_the_training_set(screens, tmp_path):
    out = tmp_path / "dist" / "train.zip"
    m = pack(screens, out)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
        assert zf.testzip() is None
    listed = {r["path"] for r in csv.DictReader(open(screens / "screen_train.csv"))}
    assert names == listed | {"screen_train.csv", "rlvs_frames.csv", "manifest.json"}
    assert m["images"] == manifest["images"] == 12 and m["composites"] == m["tiles"] == 6
    assert m["zip_bytes"] == out.stat().st_size
    assert not out.with_suffix(".zip.part").exists()


def test_pack_excludes_test_data_pages_and_weights(screens, tmp_path):
    out = tmp_path / "train.zip"
    pack(screens, out)
    names = zipfile.ZipFile(out).namelist()
    for bad in ("composite/", "scraped_pages/", "train_pages/", "composites.csv", "sources.csv",
                "urls.txt", ".safetensors", "stray.png"):
        assert not any(n.startswith(bad) or n.endswith(bad) for n in names), bad


def test_pack_paths_resolve_after_unzip(screens, tmp_path):
    out = tmp_path / "train.zip"
    pack(screens, out)
    dest = tmp_path / "unzipped"
    zipfile.ZipFile(out).extractall(dest)
    for r in csv.DictReader(open(dest / "screen_train.csv")):
        assert (dest / r["path"]).is_file()


def test_pack_refuses_listed_test_image(screens, tmp_path):
    with open(screens / "screen_train.csv", "a", newline="") as f:
        csv.writer(f).writerow(["composite/violence/t_0.png", 1, "g", "30-45", 0, "train"])
    with pytest.raises(ValueError, match="TEST set"):
        pack(screens, tmp_path / "x.zip")
    assert not (tmp_path / "x.zip").exists()


def test_pack_fails_on_missing_image(screens, tmp_path):
    (screens / "train_composites/tiles/violence/p0_0_tl.png").unlink()
    with pytest.raises(FileNotFoundError, match="not on disk"):
        pack(screens, tmp_path / "x.zip")


@pytest.mark.parametrize("name", ["composite/violence/a.png", "scraped_pages/a.png", "train_pages/a.png",
                                  "train_composites/m.safetensors", "../x.png", "/abs.png", "other/a.png",
                                  "sources.csv"])
def test_check_arcname_rejects(name):
    with pytest.raises(ValueError):
        check_arcname(name)


@pytest.mark.parametrize("name", ["screen_train.csv", "rlvs_frames.csv",
                                  "train_composites/tiles/violence/a_0_tl.png"])
def test_check_arcname_accepts(name):
    check_arcname(name)

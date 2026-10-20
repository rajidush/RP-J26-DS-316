"""
check_leakage.py — Confirm the screen TRAINING set shares nothing with the TEST set.

TEST  = data/screens/urls.txt, scraped_pages/, composite/ (composites.csv)
TRAIN = data/screens/urls_train.txt, train_pages/, train_composites/ (screen_train.csv)

Checks (any failure → exit code 1):
  1. URLs       no shared URL and no shared website — in the lists AND in what was
                actually scraped (sources.csv vs train_sources.csv)
  2. Pages      no shared page id and no byte-identical page screenshot; every
                training composite's background is a training page
  3. RLVS       no shared RLVS frame (by path and by content), no shared
                approximate video group (rlvs_groups.py), no training frame
                flagged near_test, and every training frame is from the RLVS
                TRAIN split
  4. Train/val  no page, website or video group in both train and val
Also reports which RLVS split the existing TEST composites' frames come from.

Writes results/screen_train_leakage.json.

Usage
-----
    python component1_screen_monitoring/src/check_leakage.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

try:
    from .config import COMPONENT_ROOT, SCREENS_DIR
    from .dataset_checks import domain_overlap, read_url_list, url_overlap
except ImportError:
    from config import COMPONENT_ROOT, SCREENS_DIR
    from dataset_checks import domain_overlap, read_url_list, url_overlap

REPORT = COMPONENT_ROOT / "results" / "screen_train_leakage.json"


def _rows(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def _md5s(folder: Path) -> dict[str, str]:
    return {hashlib.md5(p.read_bytes()).hexdigest(): p.name for p in sorted(Path(folder).glob("*.png"))}


def rlvs_key_of(source_frame: str) -> str:
    """composites.csv 'raw/violence/1045.jpg' → splits.csv key 'violence /1045.jpg'."""
    _, cls, name = source_frame.split("/")
    return f"{cls} /{name}"


def split_integrity(rows: list[dict], keys=("page_id", "site", "video_group")) -> dict[str, list]:
    """For each key: values that appear in BOTH the train and val split (should be empty)."""
    out = {}
    for k in keys:
        tr = {r[k] for r in rows if r["split"] == "train"}
        va = {r[k] for r in rows if r["split"] == "val"}
        out[k] = sorted(tr & va)
    return out


def run(screens: Path = SCREENS_DIR) -> dict:
    checks: dict[str, dict] = {}

    def check(name, shared, **info):
        checks[name] = {"ok": not shared, "shared": sorted(shared)[:20], "n_shared": len(shared), **info}

    # 1. URLs ---------------------------------------------------------------
    tr_urls, te_urls = read_url_list(screens / "urls_train.txt"), read_url_list(screens / "urls.txt")
    check("url_lists_shared_urls", url_overlap(tr_urls, te_urls), n_train=len(tr_urls), n_test=len(te_urls))
    check("url_lists_shared_websites", domain_overlap(tr_urls, te_urls))
    tr_log, te_log = _rows(screens / "train_sources.csv"), _rows(screens / "sources.csv")
    tr_scraped = {r["url"] for r in tr_log if r["status"].startswith("ok")}
    te_scraped = {r["url"] for r in te_log if r["status"].startswith("ok")}
    check("scraped_shared_urls", url_overlap(tr_scraped, te_scraped))
    check("scraped_shared_websites", domain_overlap(tr_scraped, te_scraped))

    # 2. Pages --------------------------------------------------------------
    tr_pages = {p.stem.split("_", 1)[0] for p in (screens / "train_pages").glob("*.png")}
    te_pages = {p.stem.split("_", 1)[0] for p in (screens / "scraped_pages").glob("*.png")}
    check("shared_page_ids", tr_pages & te_pages, n_train_pages=len(tr_pages), n_test_pages=len(te_pages))
    tr_md5, te_md5 = _md5s(screens / "train_pages"), _md5s(screens / "scraped_pages")
    check("identical_page_screenshots", {tr_md5[h] for h in set(tr_md5) & set(te_md5)})
    train_rows = _rows(screens / "screen_train.csv")
    used_pages = {r["page_id"] for r in train_rows}
    check("training_backgrounds_not_training_pages", used_pages - tr_pages)

    # 3. RLVS frames --------------------------------------------------------
    frames = {r["path"]: r for r in _rows(screens / "rlvs_frames.csv")}
    test_comp = _rows(screens / "composites.csv")
    te_keys = {rlvs_key_of(r["source_frame"]) for r in test_comp}
    tr_keys = {r["source_frame"] for r in train_rows}
    check("shared_rlvs_frames", tr_keys & te_keys, n_train_frames=len(tr_keys), n_test_frames=len(te_keys))
    rlvs_root = Path.home() / "Desktop" / "new_violence"
    def md5(key):
        return hashlib.md5((rlvs_root / key.replace(" /", "/")).read_bytes()).hexdigest()
    check("identical_rlvs_frames", {md5(k) for k in tr_keys} & {md5(k) for k in te_keys})
    tr_groups = {frames[k]["video_group"] for k in tr_keys}
    te_groups = {frames[k]["video_group"] for k in te_keys}
    check("shared_video_groups", tr_groups & te_groups,
          note="video groups are APPROXIMATE (near-identical frames), see rlvs_groups.py")
    check("training_frames_near_test", {k for k in tr_keys if frames[k]["near_test"] != "0"})
    check("training_frames_not_rlvs_train", {k for k in tr_keys if frames[k]["rlvs_split"] != "train"})

    # 4. Train / val ----------------------------------------------------------
    for k, shared in split_integrity(train_rows).items():
        check(f"train_val_shared_{k}", set(shared))

    # Report: RLVS split of the TEST composites -------------------------------
    test_split = {
        "distinct_frames_by_rlvs_split": dict(Counter(frames[k]["rlvs_split"] for k in te_keys)),
        "composites_by_rlvs_split": dict(Counter(frames[rlvs_key_of(r["source_frame"])]["rlvs_split"]
                                                for r in test_comp)),
    }
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks,
            "test_composites_rlvs_split": test_split}


def main() -> int:
    result = run()
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2))
    width = max(map(len, result["checks"]))
    for name, c in result["checks"].items():
        extra = {k: v for k, v in c.items() if k not in ("ok", "shared", "n_shared")}
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {name:<{width}}  shared: {c['n_shared']}"
              + (f"  {extra}" if extra else "") + (f"  e.g. {c['shared'][:3]}" if c["shared"] else ""))
    ts = result["test_composites_rlvs_split"]
    print(f"\nTEST composites — RLVS split of their frames: distinct frames {ts['distinct_frames_by_rlvs_split']}, "
          f"composites {ts['composites_by_rlvs_split']}")
    print(f"\n{'NO LEAKAGE' if result['ok'] else 'LEAKAGE FOUND'} → {REPORT.relative_to(COMPONENT_ROOT.parent)}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

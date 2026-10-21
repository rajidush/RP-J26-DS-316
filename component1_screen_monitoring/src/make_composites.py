"""
make_composites.py — Paste RLVS frames into scraped page screenshots.

Simulates what a screen looks like when a video or image is shown inside an
ordinary page: a frame from data/raw/<class>/ is placed on a page screenshot
from data/screens/scraped_pages/ in one of three layouts:

    video_player   16:9 box, 20–60% of the page width, letterboxed or filled,
                   sometimes with a fake player control bar
    article_image  4:3 / 3:2 image, 25–45% of the width, in the content column
    feed_post      social-feed card (avatar + name lines + 1:1 / 4:5 image)

Output: data/screens/composite/<class>/<sourceid>_<n>.png (same size as the
background). <sourceid> is the background page's id, so every composite made
from one page shares a prefix → group-aware train/test splits later.
Every composite is logged to data/screens/composites.csv. Fully reproducible
for a given --seed.

Usage
-----
    # TEST set (unchanged behaviour)
    python component1_screen_monitoring/src/make_composites.py              # 1 per page screenshot per class
    python component1_screen_monitoring/src/make_composites.py --per-background 2 --seed 7

    # TRAINING set (separate pages, RLVS TRAIN-split frames only, 85/15 group split, + tiles)
    python component1_screen_monitoring/src/make_composites.py \
        --pages data/screens/train_pages --frames-split train \
        --out data/screens/train_composites --n 750 --seed 7

Training mode (make_train_set) splits BEFORE compositing — websites 85/15 and
approximate RLVS video groups 85/15 (rlvs_groups.py) — and pairs train pages
only with train frames (val with val), so no page, website or video group is in
both. Frames flagged near_test (used in, grouped with, or similar to a TEST
composite frame) are never used. Each composite also yields up to 5 tile crops
(regions.tile_boxes: 2x2 + centre) labelled by how much of the pasted frame
they contain. Everything is listed in data/screens/screen_train.csv.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

try:
    from .config import COMPONENT_ROOT, SCREENS_DIR
    from .dataset_checks import site_of
    from .regions import tile_boxes
except ImportError:
    from config import COMPONENT_ROOT, SCREENS_DIR
    from dataset_checks import site_of
    from regions import tile_boxes

CLASSES = ("violence", "non_violence")
LAYOUTS = ("video_player", "article_image", "feed_post")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

BACKGROUND_DIR = SCREENS_DIR / "scraped_pages"
FRAMES_ROOT = COMPONENT_ROOT / "data" / "raw"
OUT_ROOT = SCREENS_DIR / "composite"
COMPOSITES_CSV = SCREENS_DIR / "composites.csv"
CSV_FIELDS = ["filename", "class", "background", "source_frame", "layout",
              "box_x0", "box_y0", "box_x1", "box_y1", "player_bar", "seed"]

Box = tuple[int, int, int, int]


def source_id_of(path: Path) -> str:
    """'en-wikipedia-org-3fa2c1_2.png' → 'en-wikipedia-org-3fa2c1'."""
    return path.stem.split("_", 1)[0]


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _place(rng: random.Random, size: tuple[int, int], w: int, h: int,
           x_range=(0.05, 0.95), y_range=(0.08, 0.95)) -> Box:
    """Random top-left so the w×h box sits inside the given fractions of the page."""
    W, H = size
    w, h = min(w, W), min(h, H)

    def pick(lo_frac, hi_frac, total, extent):
        lo = min(int(total * lo_frac), total - extent)       # never past the edge
        hi = max(lo, min(int(total * hi_frac) - extent, total - extent))
        return rng.randint(lo, hi)

    x, y = pick(*x_range, W, w), pick(*y_range, H, h)
    return x, y, x + w, y + h


def _player_bar(draw: ImageDraw.ImageDraw, box: Box, rng: random.Random) -> None:
    x0, y0, x1, y1 = box
    bar_h = max(10, (y1 - y0) // 11)
    top = y1 - bar_h
    draw.rectangle([x0, top, x1, y1], fill=(15, 15, 15))
    played = x0 + int((x1 - x0) * rng.uniform(0.05, 0.9))
    track_y = top + 2
    draw.rectangle([x0, track_y, x1, track_y + 2], fill=(110, 110, 110))
    draw.rectangle([x0, track_y, played, track_y + 2], fill=(230, 30, 30))
    cy, s = top + bar_h // 2 + 1, bar_h // 3
    draw.polygon([(x0 + 8, cy - s), (x0 + 8, cy + s), (x0 + 8 + int(1.6 * s), cy)],
                 fill=(240, 240, 240))                       # play/pause triangle
    draw.rectangle([x0 + 20 + 2 * s, cy - 2, x0 + 20 + 2 * s + 38, cy + 2],
                   fill=(200, 200, 200))                     # time label placeholder


def compose(background: Image.Image, frame: Image.Image, rng: random.Random,
            layout: str | None = None) -> tuple[Image.Image, dict]:
    """
    Paste *frame* into a copy of *background*. Returns (image, info) where info
    has layout, box (where the frame's content sits) and player_bar.
    """
    page = background.convert("RGB").copy()
    W, H = page.size
    frame = frame.convert("RGB")
    layout = layout or rng.choice(LAYOUTS)
    draw = ImageDraw.Draw(page)
    bar = False

    if layout == "video_player":
        w = int(W * rng.uniform(0.20, 0.60))
        box = _place(rng, (W, H), w, int(w * 9 / 16))
        bw, bh = box[2] - box[0], box[3] - box[1]
        if rng.random() < 0.5:                               # letterbox inside black player
            fitted = ImageOps.contain(frame, (bw, bh))
            draw.rectangle(box, fill=(0, 0, 0))
            page.paste(fitted, (box[0] + (bw - fitted.width) // 2, box[1] + (bh - fitted.height) // 2))
        else:                                                # fill the player
            page.paste(ImageOps.fit(frame, (bw, bh)), box[:2])
        if rng.random() < 0.6:
            _player_bar(draw, box, rng)
            bar = True

    elif layout == "article_image":
        w = int(W * rng.uniform(0.25, 0.45))
        aspect = rng.choice((3 / 4, 2 / 3))
        box = _place(rng, (W, H), w, int(w * aspect), x_range=(0.10, 0.80))
        page.paste(ImageOps.fit(frame, (box[2] - box[0], box[3] - box[1])), box[:2])

    elif layout == "feed_post":
        w = int(W * rng.uniform(0.20, 0.35))
        img_h = int(w * rng.choice((1.0, 1.25)))
        pad, header = max(6, w // 30), max(36, w // 7)
        card = _place(rng, (W, H), w + 2 * pad, img_h + header + 2 * pad)
        draw.rounded_rectangle(card, radius=10, fill=(255, 255, 255), outline=(219, 219, 219))
        cx, cy = card[0] + pad, card[1] + pad
        r = header // 3
        draw.ellipse([cx, cy + header // 2 - r, cx + 2 * r, cy + header // 2 + r], fill=(200, 200, 205))
        draw.rectangle([cx + 2 * r + 8, cy + header // 2 - 7, cx + 2 * r + 8 + w // 3, cy + header // 2 - 2],
                       fill=(60, 60, 60))
        draw.rectangle([cx + 2 * r + 8, cy + header // 2 + 3, cx + 2 * r + 8 + w // 5, cy + header // 2 + 7],
                       fill=(170, 170, 170))
        top = cy + header
        box = clamp_box((cx, top, cx + w, top + img_h), (W, H))
        page.paste(ImageOps.fit(frame, (box[2] - box[0], box[3] - box[1])), box[:2])

    else:
        raise ValueError(f"unknown layout {layout!r}")

    return page, {"layout": layout, "box": box, "player_bar": bar}


def clamp_box(box: Box, size: tuple[int, int]) -> Box:
    W, H = size
    x0, y0, x1, y1 = box
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def make_composites(
    background_dir: Path = BACKGROUND_DIR,
    frames_root: Path = FRAMES_ROOT,
    out_root: Path = OUT_ROOT,
    csv_path: Path = COMPOSITES_CSV,
    per_background: int = 1,
    seed: int = 42,
) -> list[dict]:
    """
    For every background screenshot and class, make *per_background* composites.
    Frames are drawn without replacement (cycling once all are used). Returns the log rows.
    """
    rng = random.Random(seed)
    backgrounds = _images(background_dir)
    if not backgrounds:
        raise FileNotFoundError(f"No page screenshots in {background_dir} — run scrape_screenshots.py first")

    rows: list[dict] = []
    for cls in CLASSES:
        frames = _images(frames_root / cls)
        if not frames:
            raise FileNotFoundError(f"No frames in {frames_root / cls}")
        pool: list[Path] = []
        counter: dict[str, int] = defaultdict(int)
        (out_root / cls).mkdir(parents=True, exist_ok=True)
        for old in (out_root / cls).glob("*.png"):          # previous run's output only
            old.unlink()

        for bg_path in backgrounds:
            sid = source_id_of(bg_path)
            with Image.open(bg_path) as bg:
                for _ in range(per_background):
                    if not pool:
                        pool = frames[:]
                        rng.shuffle(pool)
                    frame_path = pool.pop()
                    with Image.open(frame_path) as fr:
                        img, info = compose(bg, fr, rng)
                    name = f"{sid}_{counter[sid]}.png"
                    counter[sid] += 1
                    img.save(out_root / cls / name)
                    x0, y0, x1, y1 = info["box"]
                    rows.append({
                        "filename": f"{cls}/{name}", "class": cls,
                        "background": bg_path.name,
                        "source_frame": str(frame_path.relative_to(frames_root.parent)),
                        "layout": info["layout"],
                        "box_x0": x0, "box_y0": y0, "box_x1": x1, "box_y1": y1,
                        "player_bar": info["player_bar"], "seed": seed,
                    })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:                # rewritten each run: matches disk
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Training set
# ---------------------------------------------------------------------------

TRAIN_CSV = SCREENS_DIR / "screen_train.csv"
TRAIN_CSV_FIELDS = ["path", "label", "group", "size_bucket", "is_tile", "split",
                    "page_id", "site", "video_group", "source_frame", "layout", "tile"]
LABEL = {"violence": 1, "non_violence": 0}

#: A tile of a violent composite is labelled violent if it contains at least this
#: share of the pasted frame, safe if at most TILE_SAFE_MAX; anything between is
#: ambiguous and not saved.
TILE_VIOLENT_MIN = 0.60
TILE_SAFE_MAX = 0.02
TILE_SAVE_SCALE = 0.5            # tiles are saved at half size (model input is 224 px anyway)


def size_bucket(box: Box, page_width: int) -> str:
    """Width of the pasted frame as a share of the page — same buckets as the test eval."""
    frac = (box[2] - box[0]) / page_width
    return "under30" if frac < 0.30 else ("30-45" if frac < 0.45 else "45plus")


def overlap_share(box: Box, tile: Box) -> float:
    """Share of *box* (the pasted frame) that lies inside *tile*."""
    w = max(0, min(box[2], tile[2]) - max(box[0], tile[0]))
    h = max(0, min(box[3], tile[3]) - max(box[1], tile[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return (w * h) / area if area else 0.0


def tile_label(composite_label: int, box: Box, tile: Box) -> int | None:
    """1 / 0 for a tile, or None if it only partly contains a violent frame (skip it)."""
    if composite_label == 0:
        return 0
    share = overlap_share(box, tile)
    if share >= TILE_VIOLENT_MIN:
        return 1
    if share <= TILE_SAFE_MAX:
        return 0
    return None


def group_split(groups: list[str], val_frac: float = 0.15, seed: int = 0) -> dict[str, str]:
    """
    Assign whole groups to "train" / "val" so that ~val_frac of the ITEMS end up in
    val. *groups* has one entry per item. Deterministic for a seed.
    """
    counts = Counter(groups)
    order = sorted(counts)
    random.Random(seed).shuffle(order)
    target, n_val, out = val_frac * len(groups), 0, {}
    for g in order:                                  # shuffled; fill val, then the rest is train
        if n_val < target:
            out[g], n_val = "val", n_val + counts[g]
        else:
            out[g] = "train"
    return out


def _page_sites(log_csv: Path) -> dict[str, str]:
    """page_id → website, from the scraper log of the training pages."""
    sites = {}
    for r in csv.DictReader(open(log_csv)):
        if r["filename"] and r["status"].startswith("ok"):
            sites[source_id_of(Path(r["filename"]))] = site_of(r["url"])
    return sites


def make_train_set(
    pages_dir: Path,
    pages_log: Path,
    frames_csv: Path,
    frames_root: Path,
    out_dir: Path,
    csv_path: Path = TRAIN_CSV,
    n_per_class: int = 750,
    frames_split: str = "train",
    val_frac: float = 0.15,
    seed: int = 7,
    save_tiles: bool = True,
) -> list[dict]:
    out_dir = Path(out_dir).resolve()
    if out_dir == OUT_ROOT.resolve() or OUT_ROOT.resolve() in out_dir.parents or out_dir == BACKGROUND_DIR.resolve():
        raise ValueError("training output must not be inside the TEST set folders")
    rng = random.Random(seed)

    # pages → website split
    sites = _page_sites(pages_log)
    pages = [p for p in _images(pages_dir) if source_id_of(p) in sites]
    if not pages:
        raise FileNotFoundError(f"no logged page screenshots in {pages_dir}")
    site_split = group_split([sites[source_id_of(p)] for p in pages], val_frac, seed)
    pages_by_split = {s: [p for p in pages if site_split[sites[source_id_of(p)]] == s]
                      for s in ("train", "val")}

    # frames → video-group split (per class)
    frames = [r for r in csv.DictReader(open(frames_csv))
              if r["rlvs_split"] == frames_split and r["near_test"] == "0"]
    frames_by = {(s, c): [] for s in ("train", "val") for c in CLASSES}
    for c in CLASSES:
        fc = [r for r in frames if r["class"] == c]
        vsplit = group_split([r["video_group"] for r in fc], val_frac, seed + 1)
        for r in fc:
            frames_by[(vsplit[r["video_group"]], c)].append(r)

    for sub in ("composites", "tiles"):
        for c in CLASSES:
            d = out_dir / sub / c
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("*.png"):                 # previous run's generated output only
                old.unlink()

    rows, counter = [], Counter()
    for split in ("train", "val"):
        n = round(n_per_class * (val_frac if split == "val" else 1 - val_frac))
        for c in CLASSES:
            pool: list[dict] = []
            for _ in range(n):
                if not pool:
                    pool = frames_by[(split, c)][:]
                    rng.shuffle(pool)
                fr = pool.pop()
                page = rng.choice(pages_by_split[split])
                sid = source_id_of(page)
                with Image.open(page) as bg, Image.open(frames_root / fr["path"].replace(" /", "/")) as im:
                    img, info = compose(bg, im, rng)
                name = f"{sid}_{counter[(c, sid)]}"
                counter[(c, sid)] += 1
                rel = Path("composites") / c / f"{name}.png"
                img.save(out_dir / rel)
                common = {"split": split, "page_id": sid, "site": sites[sid],
                          "video_group": fr["video_group"], "source_frame": fr["path"],
                          "layout": info["layout"], "group": f"{sid}|{fr['video_group']}",
                          "size_bucket": size_bucket(info["box"], img.width)}
                rows.append({**common, "path": str(out_dir.name / rel), "label": LABEL[c],
                             "is_tile": 0, "tile": ""})
                if not save_tiles:
                    continue
                for tname, tbox in tile_boxes(img.size):
                    lab = tile_label(LABEL[c], info["box"], tbox)
                    if lab is None:
                        continue
                    crop = img.crop(tbox)
                    crop = crop.resize((round(crop.width * TILE_SAVE_SCALE),
                                        round(crop.height * TILE_SAVE_SCALE)), Image.Resampling.LANCZOS)
                    trel = Path("tiles") / c / f"{name}_{tname}.png"
                    crop.save(out_dir / trel)
                    rows.append({**common, "path": str(out_dir.name / trel), "label": lab,
                                 "is_tile": 1, "tile": tname})

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAIN_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--per-background", type=int, default=1,
                    help="TEST mode: composites per page screenshot per class (default 1)")
    ap.add_argument("--seed", type=int, default=None, help="default 42 (test) / 7 (training)")
    ap.add_argument("--pages", type=Path, help="TRAINING mode: folder of page screenshots")
    ap.add_argument("--pages-log", type=Path, default=SCREENS_DIR / "train_sources.csv",
                    help="scraper log for --pages (page → URL → website)")
    ap.add_argument("--frames-split", choices=("train", "val", "test"), default="train",
                    help="RLVS split the frames come from (training mode: train)")
    ap.add_argument("--frames-csv", type=Path, default=SCREENS_DIR / "rlvs_frames.csv",
                    help="output of rlvs_groups.py")
    ap.add_argument("--frames-root", type=Path, default=Path.home() / "Desktop" / "new_violence")
    ap.add_argument("--out", type=Path, help="TRAINING mode: output folder")
    ap.add_argument("--n", type=int, default=750, help="TRAINING mode: composites per class")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--no-tiles", action="store_true")
    args = ap.parse_args()

    if args.pages or args.out:
        if not (args.pages and args.out):
            ap.error("training mode needs both --pages and --out")
        rows = make_train_set(args.pages, args.pages_log, args.frames_csv, args.frames_root,
                              args.out, n_per_class=args.n, frames_split=args.frames_split,
                              val_frac=args.val_frac, seed=7 if args.seed is None else args.seed,
                              save_tiles=not args.no_tiles)
        c = Counter((r["split"], "tile" if r["is_tile"] else "composite", r["label"]) for r in rows)
        print(f"Wrote {len(rows)} images → {args.out}\nList → {TRAIN_CSV}")
        for k in sorted(c):
            print(f"  {k[0]:<5} {k[1]:<9} label {k[2]}: {c[k]}")
        return

    rows = make_composites(per_background=args.per_background,
                           seed=42 if args.seed is None else args.seed)
    by_class = defaultdict(int)
    for r in rows:
        by_class[r["class"]] += 1
    print(f"Wrote {len(rows)} composites {dict(by_class)} → {OUT_ROOT}\nLog → {COMPOSITES_CSV}")


if __name__ == "__main__":
    main()

"""Memes on screen: does C2 find them, and does the meme model still work on them?

    python -m component2_hate_speech_detection.evaluation.screen_memes --split val    # tuning
    python -m component2_hate_speech_detection.evaluation.screen_memes --split test   # report

Screens are built, not scraped, so every picture's true box is known. Each
page is 1920x1080 in one of six layouts a child sees:

    feed-light, feed-dark   social feed: avatar, name, caption, meme, buttons
    chat-dark               chat with one meme attached to a message
    gallery                 2x2 or 3x2 grid of memes with 4-16 px gutters
    article                 dense paragraphs, one inline picture
    text-chat               a busy chat with no picture at all (false alarms)

Thresholds in `src/readers/pictures.py` were set on pages built from val
memes; test memes are reported with the thresholds fixed. The Hateful Memes
images stay in the git-ignored data folder; nothing generated is saved.

Measures:
    regions   recall of true pictures at IoU >= 0.5 and >= 0.7, false
              proposals per page, merged boxes, time per frame
    scores    meme AUROC when each test meme sits on a feed, chat or article
              page and is scored from the finder's crop with the OCR text
              inside it, against the meme alone and the whole frame. A meme
              the finder misses scores 0 (no evidence at all).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..meme_model.embed import IMAGE_ROOT, MANIFEST_DIR, read_manifest
from ..src.engine.meme_scorer import MemeScorer
from ..src.readers.pictures import MIN_SIDE, find_pictures, text_inside
from ..src.readers.screen import ScreenReader

RESULTS = Path(__file__).resolve().parent / "results"
FONT = "C:/Windows/Fonts/segoeui.ttf"
BOLD = "C:/Windows/Fonts/segoeuib.ttf"
LAYOUTS = ("feed-light", "feed-dark", "chat-dark", "gallery", "article", "text-chat")
WORDS = ("the of and to in is you that it he was for on are as with his they at be this from have or by one "
         "had not but what all were when we there can an your which their said if do will each about how up "
         "out them then she many some so these would other into has more her two like him see time could no "
         "make than first been its who now people my made over did down only way find use may water long "
         "little very after words called just where most know get through back much before go good new write "
         "our used me man too any day same right look think also around another came come work three word").split()
THEMES = {"light": dict(bg=(240, 242, 245), card=(255, 255, 255), text=(28, 30, 33), meta=(101, 103, 107)),
          "dark": dict(bg=(49, 51, 56), card=(43, 45, 49), text=(219, 222, 225), meta=(148, 155, 164))}

Box = Tuple[int, int, int, int]


class Page:
    def __init__(self, theme: str, rng: np.random.Generator) -> None:
        self.t, self.rng = THEMES[theme], rng
        self.img = Image.new("RGB", (1920, 1080), self.t["bg"])
        self.d = ImageDraw.Draw(self.img)
        self.pictures: List[Tuple[Box, int]] = []

    def font(self, size: int, bold: bool = False):
        return ImageFont.truetype(BOLD if bold else FONT, size)

    def words(self, n: int) -> str:
        return " ".join(self.rng.choice(WORDS, n))

    def text(self, xy, s, size=18, bold=False, colour=None):
        self.d.text(xy, s, fill=colour or self.t["text"], font=self.font(size, bold))

    def avatar(self, x, y, r=20):
        colour = tuple(int(c) for c in self.rng.integers(60, 230, 3))
        self.d.ellipse([x, y, x + 2 * r, y + 2 * r], fill=colour)

    def button(self, x, y, label):
        w = int(self.font(16).getlength(label)) + 28
        self.d.rounded_rectangle([x, y, x + w, y + 34], radius=8, outline=self.t["meta"], width=1)
        self.text((x + 14, y + 6), label, 16, colour=self.t["meta"])
        return x + w + 10

    def chrome(self):
        self.d.rectangle([0, 0, 1920, 56], fill=self.t["card"])
        self.text((24, 14), "home   explore   messages   notifications", 18, colour=self.t["meta"])
        self.d.rounded_rectangle([1300, 10, 1700, 46], radius=18, fill=self.t["bg"])
        self.text((1320, 16), "Search", 18, colour=self.t["meta"])
        for i in range(8):
            self.d.rectangle([24, 96 + i * 44, 48, 120 + i * 44], fill=tuple(int(c) for c in self.rng.integers(80, 220, 3)))
            self.text((60, 96 + i * 44), self.words(2), 18, colour=self.t["meta"])

    def picture(self, meme_id: int, x: int, y: int, width: int, max_height: int = 10_000) -> int:
        with Image.open(IMAGE_ROOT / f"img/{meme_id:05d}.png") as src:
            im = src.convert("RGB")
        height = int(im.height * width / im.width)
        if height > max_height:
            width, height = int(width * max_height / height), max_height
        im = im.resize((width, height), Image.BICUBIC)
        self.img.paste(im, (x, y))
        box = (x, y, min(1920, x + width), min(1080, y + height))
        visible = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
        if visible >= 0.6 and min(box[2] - box[0], box[3] - box[1]) >= MIN_SIDE:
            self.pictures.append((box, meme_id))
        return y + height


def build(layout: str, memes: List[int], rng: np.random.Generator) -> Page:
    """One screen; `memes` supplies picture ids in order (1 for feed/chat/article, up to 6 for gallery)."""
    theme = "light" if layout in ("feed-light", "article") else "dark"
    p = Page(theme, rng)
    p.chrome()
    it = iter(memes)
    if layout.startswith("feed"):
        y = 80
        p.d.rounded_rectangle([620, 70, 1300, 1080], radius=10, fill=p.t["card"])
        p.avatar(640, y + 10)
        p.text((690, y + 8), p.words(2).title(), 18, bold=True)
        p.text((690, y + 32), "2h", 14, colour=p.t["meta"])
        y += 70
        for _ in range(int(rng.integers(1, 3))):
            p.text((640, y), p.words(9), 18)
            y += 28
        y = p.picture(next(it), 640, y + 10, int(rng.integers(300, 641)), max_height=760) + 12
        x = 640
        for label in ("Like", "Comment", "Share"):
            x = p.button(x, y, label)
    elif layout == "chat-dark":
        y = 90
        for i in range(int(rng.integers(2, 5))):
            p.avatar(300, y)
            p.text((356, y), p.words(1).title(), 18, bold=True, colour=(88, 101, 242))
            p.text((356, y + 26), p.words(int(rng.integers(4, 14))), 18)
            y += 80
        p.avatar(300, y)
        p.text((356, y), p.words(1).title(), 18, bold=True, colour=(88, 101, 242))
        y = p.picture(next(it), 356, y + 32, int(rng.integers(250, 451)), max_height=520) + 20
        if y < 1000:
            p.text((356, y), p.words(6), 18)
    elif layout == "gallery":
        cols, rows = (3, 2) if rng.random() < 0.5 else (2, 2)
        gap, cell = int(rng.integers(4, 17)), int(rng.integers(250, 341))
        x0 = 960 - (cols * cell + (cols - 1) * gap) // 2
        for r in range(rows):
            for c in range(cols):
                meme = next(it, None)
                if meme is None:
                    break
                p.picture(meme, x0 + c * (cell + gap), 100 + r * (cell + gap), cell, max_height=cell)
    elif layout == "article":
        p.d.rectangle([380, 56, 1540, 1080], fill=p.t["card"])
        p.text((420, 90), p.words(7).title(), 32, bold=True)
        y = 150
        inline = int(rng.integers(0, 3))
        for block in range(4):
            if block == inline:
                y = p.picture(next(it), 420, y, int(rng.integers(320, 521)), max_height=420) + 24
            for _ in range(int(rng.integers(3, 7))):
                if y > 1040:
                    break
                p.text((420, y), p.words(16), 18)
                y += 26
            y += 18
    elif layout == "text-chat":
        y = 80
        while y < 1000:
            p.avatar(300, y)
            p.text((356, y), p.words(1).title(), 18, bold=True, colour=(88, 101, 242))
            for _ in range(int(rng.integers(1, 3))):
                p.text((356, y + 26), p.words(int(rng.integers(6, 18))), 18)
                y += 24
            y += 56
    return p


def iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def evaluate_regions(ids: List[int], pages_per_layout: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    report = {}
    for layout in LAYOUTS:
        truth = found50 = found70 = false = merged = 0
        ms = []
        for _ in range(pages_per_layout):
            page = build(layout, list(rng.choice(ids, 6, replace=False)), rng)
            started = time.perf_counter()
            boxes = find_pictures(page.img)
            ms.append((time.perf_counter() - started) * 1000)
            gts = [g for g, _ in page.pictures]
            truth += len(gts)
            for g in gts:
                best = max((iou(g, b) for b in boxes), default=0.0)
                found50 += best >= 0.5
                found70 += best >= 0.7
            for b in boxes:
                hits = sum(iou(g, b) >= 0.1 for g in gts)
                false += max((iou(g, b) for g in gts), default=0.0) < 0.5
                merged += hits >= 2
        report[layout] = {
            "pages": pages_per_layout, "pictures": truth,
            "recall@0.5": round(found50 / truth, 3) if truth else None,
            "recall@0.7": round(found70 / truth, 3) if truth else None,
            "false_per_page": round(false / pages_per_layout, 2),
            "merged_boxes": merged,
            "median_ms": round(float(np.median(ms)), 1),
        }
    return report


def evaluate_scores(rows: List[dict], seed: int) -> dict:
    """Each meme on its own page: found? and its score from the crop vs alone vs the whole frame."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    scorer, ocr = MemeScorer(), ScreenReader().ocr
    single = ("feed-light", "feed-dark", "chat-dark", "article")
    y, alone, whole, crop, found, ms = [], [], [], [], [], []
    for k, row in enumerate(rows):
        page = build(single[k % len(single)], [row["id"]], rng)
        truth = page.pictures[0][0] if page.pictures else None
        reader = ScreenReader(ocr)
        read = reader.read(page.img)
        started = time.perf_counter()
        boxes = find_pictures(page.img)
        best = max(boxes, key=lambda b: iou(truth, b), default=None) if truth else None
        hit = best is not None and iou(truth, best) >= 0.5
        p_crop = scorer.probability(page.img.crop(best), text_inside(best, reader.lines, page.img.size)) if hit else 0.0
        ms.append((time.perf_counter() - started) * 1000)
        with Image.open(IMAGE_ROOT / row["img"]) as src:
            meme = src.convert("RGB")
        alone.append(scorer.probability(meme, ocr.read(meme)[0]))
        whole.append(scorer.probability(page.img, " ".join(s.text for s in read.spans)))
        crop.append(p_crop)
        found.append(hit)
        y.append(row["label"])
        if (k + 1) % 50 == 0:
            print(f"scores {k + 1}/{len(rows)}", flush=True)
    y, found = np.array(y), np.array(found)
    auc = lambda p, m=slice(None): round(float(roc_auc_score(y[m], np.array(p)[m])), 4)
    return {
        "memes": len(rows),
        "found": round(float(found.mean()), 3),
        "auroc_meme_alone_ocr_caption": auc(alone),
        "auroc_whole_frame": auc(whole),
        "auroc_screen_crops": auc(crop),
        "auroc_screen_crops_found_only": auc(crop, found),
        "auroc_alone_same_memes": auc(alone, found),
        "median_ms_find_and_score": round(float(np.median(ms)), 1),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--pages", type=int, default=40, help="pages per layout")
    parser.add_argument("--scores", action="store_true", help="also score every meme on a page (slow: real OCR)")
    args = parser.parse_args(argv)
    rows = read_manifest(MANIFEST_DIR / f"{args.split}.jsonl")
    seed = 316 if args.split == "val" else 317
    report = {"split": args.split, "regions": evaluate_regions([r["id"] for r in rows], args.pages, seed)}
    if args.scores:
        report["scores"] = evaluate_scores(rows, seed)
    print(json.dumps(report, indent=2))
    if args.split == "test":
        (RESULTS / "screen_memes.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

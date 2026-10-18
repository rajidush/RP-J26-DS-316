"""Picture regions on a screen frame: where the memes and photos are (proposal SO3).

The meme model is measured on whole memes. On a whole 1920x1080 screenshot its
AUROC falls from 0.750 to 0.68-0.71 (meme_model/README), because the meme is a
small part of what CLIP sees. So the screen path first finds the pictures on
screen and scores each crop.

Method: classical detection, which Chen et al. (ESEC/FSE 2020) found competitive
with deep detectors for large non-text GUI elements (also UIED; Xie et al.,
2020). It needs no model and costs a few milliseconds per frame:

1. Mark BLOCK x BLOCK blocks that are not a flat UI fill. Photos almost never
   have perfectly uniform blocks; app backgrounds, cards and gutters do.
   A flat block counts as UI only in one of the page's UI colours: colours
   that touch the screen edge, or that fill a long panel or card. So the
   flat white backdrop of a studio-photo meme on a dark page stays part of
   the meme. (A "busy texture" mask missed the smooth parts of photos:
   walls, sky, faces. It found 10-40 % of memes on val pages.)
2. A morphological opening wider than a text line removes single lines and
   thin UI; each shape's outer outline fills flat patches inside a picture.
3. Recursive XY-cut (Nagy & Seth, 1984) cuts a shape at every straight
   gutter of the page's background colour, so a caption above a meme, the
   buttons below it and the neighbours in a gallery all come apart; pieces
   too small to be a meme are dropped. Only background-coloured gutters
   count: a white row inside a studio-photo meme on a dark page is not one.
4. Keep boxes at least MIN_SIDE on each side, with a plausible aspect ratio,
   that are not a text paragraph. A paragraph is mostly one background tone
   *and* has almost no colour (Hasler & Suesstrunk, 2003, colourfulness); a
   meme on a white background is still colourful.

Thresholds were set on pages built from the *val* memes only
(evaluation/screen_memes.py).

`PictureReader` puts it on the live screen: each changed frame's pictures are
found, each new picture is scored once by the meme model with the OCR text
inside it as its caption, and the strongest score goes to the analyzer as a
vision signal (evidence only until the meme score passes the I-1 gate).
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from ..engine.fusion import Signal
from ..engine.meme_scorer import MemeScorer

Box = Tuple[int, int, int, int]  # x0, y0, x1, y1 in frame pixels

BLOCK = 8              # frame pixels per analysis block
FLAT_STD = 1.5         # grey-level std at or below which a block is a flat UI fill
OPEN_BLOCKS = 5        # opening kernel: removes busy shapes thinner than 40 px (text lines)
MIN_SIDE = 150         # smaller pictures (avatars, icons, emoji) are not memes
MAX_ASPECT = 3.5
MAX_FLAT = 0.45        # share of a box at its single most common grey level (text paragraph > 0.6)
MIN_COLOUR = 12.0      # colourfulness below which a mostly-flat box is text, not a picture
MIN_GUTTER = 3         # px: thinner uniform lines are not gutters
BG_TOLERANCE = 4.0     # grey levels: how close a gutter must be to the page background
PANEL_SPAN = (0.40, 0.60)  # a flat region this share of frame width or height is a UI panel/card
GUTTER_STD = 1.5       # a row/column this uniform across the box is a gutter between pictures
MAX_PICTURES = 6
WHOLE_IMAGE_SHARE = 0.8  # a dropped image whose largest picture covers this much is one meme


def _ui_colours(flat: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Grey levels (+-2) of the page's UI fills: flat regions on the screen edge or spanning a panel."""
    ui = np.zeros(256, dtype=bool)
    rows, cols = flat.shape
    for value in np.unique(level[flat]):
        same = (flat & (np.abs(level - value) <= 1)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(same, connectivity=4)
        for i in range(1, n):
            x, y, w, h, _ = stats[i]
            on_edge = x == 0 or y == 0 or x + w == cols or y + h == rows
            if on_edge or w >= PANEL_SPAN[0] * cols or h >= PANEL_SPAN[1] * rows:
                ui[max(0, value - 2):value + 3] = True
                break
    return ui


def _busy_mask(grey: np.ndarray) -> np.ndarray:
    """1 where a block is part of content (picture or text), 0 where it is a flat UI fill."""
    h, w = (grey.shape[0] // BLOCK) * BLOCK, (grey.shape[1] // BLOCK) * BLOCK
    blocks = grey[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).astype(np.float32)
    flat = blocks.std(axis=(1, 3)) <= FLAT_STD
    level = np.clip(np.round(blocks.mean(axis=(1, 3))), 0, 255).astype(np.int64)
    ui = _ui_colours(flat, level)
    return (~(flat & ui[level])).astype(np.uint8)


def _filled_shapes(mask: np.ndarray) -> List[Box]:
    """Outer outlines of the opened mask, holes filled, as block-unit boxes."""
    kernel = np.ones((OPEN_BLOCKS, OPEN_BLOCKS), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
    return boxes


def _background(grey: np.ndarray, box: Box, ring: int = 6) -> float:
    """Most common grey level just outside a box: the page or card it sits on."""
    x0, y0, x1, y1 = box
    h, w = grey.shape
    outer = grey[max(0, y0 - ring):min(h, y1 + ring), max(0, x0 - ring):min(w, x1 + ring)]
    inner = np.zeros(outer.shape, dtype=bool)
    inner[min(ring, y0):min(ring, y0) + (y1 - y0), min(ring, x0):min(ring, x0) + (x1 - x0)] = True
    values = outer[~inner]
    return float(np.bincount(values).argmax()) if values.size else float(np.median(grey))


def _gutters(patch: np.ndarray, axis: int, bg: float) -> np.ndarray:
    """Rows/columns that are uniform and the page's background colour."""
    return (patch.std(axis=axis) < GUTTER_STD) & (np.abs(patch.mean(axis=axis) - bg) <= BG_TOLERANCE)


def _xy_cut(grey: np.ndarray, box: Box, bg: float, depth: int = 0) -> List[Box]:
    """Cut a box at every background-coloured full-height/full-width gutter, recursively."""
    x0, y0, x1, y1 = box
    if depth > 8 or x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
        return [box]
    patch = grey[y0:y1, x0:x1].astype(np.float32)
    for axis in (0, 1):                        # 0: columns (vertical gutters), 1: rows
        uniform = _gutters(patch, axis, bg)
        length = len(uniform)
        runs, start = [], None
        for i, flag in enumerate(np.append(uniform, False)):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                runs.append((start, i))
                start = None
        for a, b in runs:
            if b - a >= MIN_GUTTER and a > 0 and b < length:   # content on both sides
                if axis == 0:
                    first, second = (x0, y0, x0 + a, y1), (x0 + b, y0, x1, y1)
                else:
                    first, second = (x0, y0, x1, y0 + a), (x0, y0 + b, x1, y1)
                return _xy_cut(grey, first, bg, depth + 1) + _xy_cut(grey, second, bg, depth + 1)
    return [box]


def _trim(grey: np.ndarray, box: Box, bg: float) -> Box:
    """Drop background-coloured margin rows/columns that the block grid pulled in."""
    x0, y0, x1, y1 = box
    patch = grey[y0:y1, x0:x1].astype(np.float32)
    cols = np.nonzero(~_gutters(patch, 0, bg))[0]
    rows = np.nonzero(~_gutters(patch, 1, bg))[0]
    if len(cols) == 0 or len(rows) == 0:
        return box
    return x0 + int(cols[0]), y0 + int(rows[0]), x0 + int(cols[-1]) + 1, y0 + int(rows[-1]) + 1


def _colourfulness(rgb: np.ndarray, box: Box) -> float:
    """Hasler & Suesstrunk (2003): ~0 for grey text on a plain background, > 15 for most photos."""
    x0, y0, x1, y1 = box
    r, g, b = (rgb[y0:y1, x0:x1, i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def _flat_share(grey: np.ndarray, box: Box) -> float:
    x0, y0, x1, y1 = box
    counts = np.bincount(grey[y0:y1, x0:x1].ravel(), minlength=256)
    peak = int(counts.argmax())
    return float(counts[max(0, peak - 2):peak + 3].sum() / counts.sum())


def text_inside(box: Box, lines: Sequence[dict], size: Tuple[int, int]) -> str:
    """The OCR lines whose centre falls inside a picture, top to bottom: the meme's caption.

    `lines` are screen-reader lines with boxes normalised to a frame of `size`.
    """
    x0, y0, x1, y1 = box
    width, height = size
    inside = []
    for line in lines:
        lx0, ly0, lx1, ly1 = line["box"]
        cx, cy = (lx0 + lx1) / 2 * width, (ly0 + ly1) / 2 * height
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            inside.append((ly0, lx0, line["text"]))
    return " ".join(text for _, _, text in sorted(inside))


def find_pictures(frame: Image.Image) -> List[Box]:
    """Boxes of picture regions large enough to be a meme, largest first."""
    rgb = np.asarray(frame.convert("RGB"))
    grey = np.asarray(frame.convert("L"))
    out: List[Box] = []
    for bx0, by0, bx1, by1 in _filled_shapes(_busy_mask(grey)):
        coarse = (bx0 * BLOCK, by0 * BLOCK, min(grey.shape[1], bx1 * BLOCK), min(grey.shape[0], by1 * BLOCK))
        bg = _background(grey, coarse)
        for box in _xy_cut(grey, coarse, bg):
            x0, y0, x1, y1 = _trim(grey, box, bg)
            w, h = x1 - x0, y1 - y0
            if w < MIN_SIDE or h < MIN_SIDE or max(w, h) / min(w, h) > MAX_ASPECT:
                continue
            if _flat_share(grey, (x0, y0, x1, y1)) > MAX_FLAT and _colourfulness(rgb, (x0, y0, x1, y1)) < MIN_COLOUR:
                continue
            out.append((x0, y0, x1, y1))
    out.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return out[:MAX_PICTURES]


class PictureReader:
    """Memes on screen: find the pictures, score each new one once, report the strongest."""

    def __init__(self, scorer: Optional[MemeScorer] = None, cache_size: int = 256) -> None:
        self.scorer = scorer or MemeScorer()
        self._cache: "OrderedDict[str, float]" = OrderedDict()
        self._size = cache_size
        self.scored = 0
        self.cached = 0

    @staticmethod
    def fingerprint(crop: Image.Image) -> str:
        """Coarse 16x16 grey thumbnail: the same meme seen again (re-rendered, nudged) hits the cache."""
        small = np.asarray(crop.convert("L").resize((16, 16), Image.BILINEAR), dtype=np.uint8) // 16
        return hashlib.sha1(small.tobytes()).hexdigest()

    def _score(self, crop: Image.Image, caption: str) -> Optional[float]:
        key = self.fingerprint(crop)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.cached += 1
            return self._cache[key]
        p = self.scorer.probability(crop, caption)
        if p is not None:
            self.scored += 1
            self._cache[key] = p
            while len(self._cache) > self._size:
                self._cache.popitem(last=False)
        return p

    def read(self, frame: Image.Image, lines: Sequence[dict], whole_image: bool = False) -> Optional[Signal]:
        """The strongest picture on `frame` as a vision signal, or None (no picture, no model).

        `whole_image=True` is for an image dropped into the app: when it holds one
        picture covering most of it (or none is found), it is scored as one meme,
        the way the model was measured; a dropped screenshot is read like a screen.
        """
        boxes = find_pictures(frame)
        area = frame.size[0] * frame.size[1]
        if whole_image and (not boxes or (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1]) >= WHOLE_IMAGE_SHARE * area):
            boxes = [(0, 0, frame.size[0], frame.size[1])]
        best: Optional[float] = None
        for box in boxes:
            p = self._score(frame.crop(box), text_inside(box, lines, frame.size))
            if p is not None and (best is None or p > best):
                best = p
        if best is None:
            return None
        return Signal("vision", round(best, 4), self.scorer.calibrated, self.scorer.name)

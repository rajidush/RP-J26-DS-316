"""
regions.py — Which parts of a screen frame to classify (no model code here).

Why: the whole 1440x900 screen is squashed to 224x224, so a video playing in a
normal-size window becomes a tiny part of the input and the classifier (trained
on full-frame video stills) misses it. Classifying a few crops — fixed tiles
and the region that is changing between frames — gives the video a chance to
fill the model's input.

Strategies
----------
    full          whole frame                                   (1 crop)
    tiles         whole frame + 2x2 overlapping grid + centre   (6 crops)
    motion        whole frame + largest moving region           (1–2 crops)
    tiles+motion  tiles + largest moving region                 (6–7 crops)

Motion detection works on a small grayscale copy of each frame, so the caller
only has to keep that small copy of the previous frame in memory.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image, ImageFilter

STRATEGIES = ("full", "tiles", "motion", "tiles+motion")

Box = tuple[int, int, int, int]          # (left, top, right, bottom), frame pixels

#: Each grid tile / the centre crop covers this fraction of width and height.
TILE_FRACTION = 0.6

#: Width of the grayscale copy used for motion detection.
MOTION_WIDTH = 160
#: Per-pixel change (0–255) that counts as motion.
MOTION_DIFF_THRESHOLD = 25
#: Smallest moving region kept, as a fraction of the frame area.
MOTION_MIN_AREA_FRACTION = 0.01
#: Padding added on each side of the motion box, as a fraction of its size.
MOTION_PAD_FRACTION = 0.10


# ---------------------------------------------------------------------------
# Fixed tiles
# ---------------------------------------------------------------------------

def tile_boxes(size: tuple[int, int]) -> list[tuple[str, Box]]:
    """2x2 overlapping grid + centre crop, each TILE_FRACTION of the frame."""
    w, h = size
    tw, th = round(w * TILE_FRACTION), round(h * TILE_FRACTION)
    xs, ys = (0, w - tw), (0, h - th)
    names = {(0, 0): "top_left", (1, 0): "top_right", (0, 1): "bottom_left", (1, 1): "bottom_right"}
    boxes = [(names[(i, j)], (xs[i], ys[j], xs[i] + tw, ys[j] + th))
             for j in range(2) for i in range(2)]
    cx, cy = (w - tw) // 2, (h - th) // 2
    boxes.append(("centre", (cx, cy, cx + tw, cy + th)))
    return boxes


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

def motion_reference(image: Image.Image) -> Image.Image:
    """Small grayscale copy of *image* — all a caller needs to keep for motion."""
    w, h = image.size
    if w <= MOTION_WIDTH and image.mode == "L":
        return image
    return image.convert("L").resize((MOTION_WIDTH, max(1, round(h * MOTION_WIDTH / w))),
                                     Image.Resampling.BILINEAR)


def _largest_component(mask: np.ndarray) -> tuple[int, Box] | None:
    """Area and bounding box (x0, y0, x1, y1 exclusive) of the largest 4-connected blob."""
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    best = None
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        queue, area = deque([(y0, x0)]), 0
        ymin = ymax = y0
        xmin = xmax = x0
        while queue:
            y, x = queue.popleft()
            area += 1
            ymin, ymax, xmin, xmax = min(ymin, y), max(ymax, y), min(xmin, x), max(xmax, x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if best is None or area > best[0]:
            best = (area, (int(xmin), int(ymin), int(xmax) + 1, int(ymax) + 1))
    return best


def motion_box(image: Image.Image, prev: Image.Image | None) -> Box | None:
    """
    Bounding box (frame pixels) of the largest region that changed since *prev*.

    *prev* may be a full frame or a ``motion_reference()`` copy. Returns ``None``
    when there is no previous frame or no changed region big enough.
    """
    if prev is None:
        return None
    cur = motion_reference(image)
    prev = motion_reference(prev)
    if prev.size != cur.size:
        prev = prev.resize(cur.size, Image.Resampling.BILINEAR)

    diff = np.abs(np.asarray(cur, dtype=np.int16) - np.asarray(prev, dtype=np.int16))
    mask = Image.fromarray(((diff > MOTION_DIFF_THRESHOLD) * 255).astype(np.uint8))
    # Morphological cleanup: erode away specks (cursor, blinking caret),
    # then dilate to merge the pieces of one moving area.
    mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(7))
    found = _largest_component(np.asarray(mask) > 0)
    if found is None:
        return None

    area, (x0, y0, x1, y1) = found
    mw, mh = cur.size
    if area < MOTION_MIN_AREA_FRACTION * mw * mh:
        return None

    pad_x, pad_y = (x1 - x0) * MOTION_PAD_FRACTION, (y1 - y0) * MOTION_PAD_FRACTION
    sx, sy = image.size[0] / mw, image.size[1] / mh
    return clamp_box((round((x0 - pad_x) * sx), round((y0 - pad_y) * sy),
                      round((x1 + pad_x) * sx), round((y1 + pad_y) * sy)), image.size)


def clamp_box(box: Box, size: tuple[int, int]) -> Box:
    w, h = size
    x0, y0, x1, y1 = box
    x0, y0 = max(0, min(x0, w - 1)), max(0, min(y0, h - 1))
    return x0, y0, max(x0 + 1, min(x1, w)), max(y0 + 1, min(y1, h))


# ---------------------------------------------------------------------------
# Strategy → list of regions
# ---------------------------------------------------------------------------

def region_boxes(
    image: Image.Image,
    prev: Image.Image | None = None,
    strategy: str = "tiles+motion",
) -> list[tuple[str, Box]]:
    """Named crop boxes for *strategy*. The whole frame is always the first region."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    w, h = image.size
    boxes = [("full", (0, 0, w, h))]
    if "tiles" in strategy:
        boxes += tile_boxes(image.size)
    if "motion" in strategy:
        box = motion_box(image, prev)
        if box is not None:
            boxes.append(("motion", box))
    return boxes

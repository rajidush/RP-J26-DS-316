"""Screen reader: change-gated OCR into positioned text blocks (proposal stage 2, SO3).

Reading the whole screen every check is the expensive way: full-frame OCR at
1920x1080 costs ~1.6 s on 4 cores. A child's screen mostly does not change
between checks, and when it does, usually only a small part changes (a new
chat message, a scrolled feed). So:

1. Compare a 1/8-scale greyscale thumbnail with the previous frame (~5 ms).
2. Nothing changed: no OCR, no analysis.
3. A small area changed: OCR only the changed tiles (plus a margin) and keep
   the text already read everywhere else.
4. Most of the screen changed (new window, full scroll): one full-frame OCR.

The reader keeps the screen's current text as positioned lines, then groups
them into blocks (a message, a paragraph) so the text scorer sees whole
sentences and the app knows which region to blur. Unchanged blocks are scored
from the analyzer's cache, so a check costs model time only for new text.

Raw frames are never stored: the reader keeps only a small greyscale
thumbnail for change detection, and the caller releases the frame.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..engine.analyzer import TextSpan
from .ocr import OcrEngine, Region

THUMB_SCALE = 8          # 1920x1080 -> 240x135 thumbnail
TILE = 16                # thumbnail pixels per tile (= 128 screen pixels)
PIXEL_DELTA = 12         # grey levels: a thumbnail pixel counts as changed above this
TILE_CHANGED_FRACTION = 0.02
FULL_READ_FRACTION = 0.40  # above this share of changed tiles, read the whole frame
MAX_BLOCK_CHARS = 600
BLOCK_GAP = 0.35         # max gap (in line heights) between lines of one block
BLOCK_OVERLAP = 0.5      # OCR boxes of touching lines may overlap by up to this


@dataclass
class ScreenRead:
    mode: str                                  # first | full | partial | unchanged
    spans: List[TextSpan] = field(default_factory=list)
    changed_fraction: float = 0.0
    ocr_ms: float = 0.0
    diff_ms: float = 0.0
    crops: int = 0                             # OCR calls made this check
    ocr_pixels: int = 0                        # pixels sent to OCR this check

    @property
    def changed(self) -> bool:
        return self.mode != "unchanged"


def _thumbnail(frame: Image.Image) -> np.ndarray:
    grey = frame.convert("L")
    return np.asarray(grey.reduce(THUMB_SCALE), dtype=np.int16)


def changed_tiles(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Boolean grid (rows, cols) of tiles whose content changed."""
    moved = np.abs(cur - prev) > PIXEL_DELTA
    rows, cols = -(-moved.shape[0] // TILE), -(-moved.shape[1] // TILE)
    padded = np.zeros((rows * TILE, cols * TILE), dtype=bool)
    padded[: moved.shape[0], : moved.shape[1]] = moved
    share = padded.reshape(rows, TILE, cols, TILE).mean(axis=(1, 3))
    return share > TILE_CHANGED_FRACTION


def tile_rects(grid: np.ndarray, margin: int = 1) -> List[Tuple[int, int, int, int]]:
    """Connected groups of changed tiles as (row0, col0, row1, col1) inclusive, with a margin."""
    seen = np.zeros_like(grid, dtype=bool)
    rects = []
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if not grid[r, c] or seen[r, c]:
                continue
            stack, r0, c0, r1, c1 = [(r, c)], r, c, r, c
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                r0, c0, r1, c1 = min(r0, y), min(c0, x), max(r1, y), max(c1, x)
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < rows and 0 <= nx < cols and grid[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            rects.append((max(0, r0 - margin), max(0, c0 - margin),
                          min(rows - 1, r1 + margin), min(cols - 1, c1 + margin)))
    return _merge_overlapping(rects)


def _merge_overlapping(rects):
    merged = True
    while merged:
        merged = False
        out = []
        while rects:
            a = rects.pop()
            for i, b in enumerate(rects):
                if a[0] <= b[2] + 1 and b[0] <= a[2] + 1 and a[1] <= b[3] + 1 and b[1] <= a[3] + 1:
                    rects[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    merged = True
                    break
            else:
                out.append(a)
        rects = out
    return rects


def _overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def group_blocks(regions: List[Region]) -> List[TextSpan]:
    """Group OCR lines into blocks: same column and almost touching -> one block.

    A block must never span two chat messages, or one child's help-seeking
    reply ("someone told me to kys and im scared") would make the framing
    guard discount the abuse in the message above it. Measured on PP-OCRv6
    boxes of a chat window: lines within one message (author header + text,
    wrapped lines) sit -0.14h to +0.07h apart; separate messages 0.67h to
    0.93h apart. So lines merge only within [-0.5h, 0.35h).
    """
    lines = sorted(regions, key=lambda r: (r["box"][1], r["box"][0]))
    blocks: List[List[Region]] = []
    for line in lines:
        x0, y0, x1, y1 = line["box"]
        placed = False
        for block in reversed(blocks[-6:]):
            last = block[-1]["box"]
            height = max(last[3] - last[1], 1e-4)
            overlap = min(x1, last[2]) - max(x0, last[0])
            narrow = min(x1 - x0, last[2] - last[0])
            if overlap > 0.3 * narrow and -BLOCK_OVERLAP * height <= y0 - last[3] < BLOCK_GAP * height:
                if sum(len(r["text"]) for r in block) + len(line["text"]) <= MAX_BLOCK_CHARS:
                    block.append(line)
                    placed = True
                break
        if not placed:
            blocks.append([line])
    spans = []
    for block in blocks:
        box = [round(min(r["box"][0] for r in block), 4), round(min(r["box"][1] for r in block), 4),
               round(max(r["box"][2] for r in block), 4), round(max(r["box"][3] for r in block), 4)]
        spans.append(TextSpan(" ".join(r["text"] for r in block), "ocr", box))
    return spans


class ScreenReader:
    def __init__(self, ocr: Optional[OcrEngine] = None) -> None:
        self.ocr = ocr or OcrEngine()
        self._thumb: Optional[np.ndarray] = None
        self._regions: List[Region] = []

    def reset(self) -> None:
        self._thumb, self._regions = None, []

    def _grow_to_whole_lines(self, rect: List[float]) -> List[float]:
        """Extend a re-read area to fully contain every known line it touches.

        A crop edge through a line of text makes OCR read half-glyphs ("Kavin"
        came back as "KaVIn"), and the good earlier reading of that line would
        be thrown away. Growing the crop keeps every re-read line whole.
        """
        for _ in range(3):
            touching = [g["box"] for g in self._regions if _overlaps(g["box"], rect)]
            grown = [min([rect[0]] + [b[0] for b in touching]), min([rect[1]] + [b[1] for b in touching]),
                     max([rect[2]] + [b[2] for b in touching]), max([rect[3]] + [b[3] for b in touching])]
            if grown == rect:
                break
            rect = grown
        return rect

    def read(self, frame: Image.Image) -> ScreenRead:
        t = time.perf_counter()
        thumb = _thumbnail(frame)
        if self._thumb is None or self._thumb.shape != thumb.shape:
            mode, grid, fraction = "first", None, 1.0
        else:
            grid = changed_tiles(self._thumb, thumb)
            fraction = float(grid.mean())
            mode = "unchanged" if fraction == 0 else ("full" if fraction > FULL_READ_FRACTION else "partial")
        self._thumb = thumb
        diff_ms = round((time.perf_counter() - t) * 1000, 1)
        if mode == "unchanged":
            return ScreenRead("unchanged", diff_ms=diff_ms)

        t = time.perf_counter()
        width, height = frame.size
        crops = pixels = 0
        if mode in ("first", "full"):
            _, self._regions = self.ocr.read(frame)
            crops, pixels = 1, width * height
        else:
            scale = THUMB_SCALE * TILE
            for r0, c0, r1, c1 in tile_rects(grid):
                rect = [c0 * scale / width, r0 * scale / height,
                        min(1.0, (c1 + 1) * scale / width), min(1.0, (r1 + 1) * scale / height)]
                rect = self._grow_to_whole_lines(rect)
                px = (int(rect[0] * width), int(rect[1] * height), int(rect[2] * width), int(rect[3] * height))
                # Lines touching a re-read area are replaced by the fresh reading.
                self._regions = [g for g in self._regions if not _overlaps(g["box"], rect)]
                _, found = self.ocr.read(frame.crop(px))
                cw, ch = px[2] - px[0], px[3] - px[1]
                for g in found:
                    b = g["box"]
                    g["box"] = [round((px[0] + b[0] * cw) / width, 4), round((px[1] + b[1] * ch) / height, 4),
                                round((px[0] + b[2] * cw) / width, 4), round((px[1] + b[3] * ch) / height, 4)]
                self._regions.extend(found)
                crops += 1
                pixels += cw * ch
        return ScreenRead(mode, group_blocks(self._regions), round(fraction, 4),
                          round((time.perf_counter() - t) * 1000, 1), diff_ms, crops, pixels)

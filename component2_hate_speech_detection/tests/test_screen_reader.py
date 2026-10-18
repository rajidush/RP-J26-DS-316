"""Screen reader: change gating, partial re-reads and message-safe grouping (proposal SO3)."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from component2_hate_speech_detection.src.engine.analyzer import Analyzer, Observation
from component2_hate_speech_detection.src.readers.ocr import OcrEngine
from component2_hate_speech_detection.src.readers.screen import (
    ScreenReader,
    changed_tiles,
    group_blocks,
    tile_rects,
)

FONT = Path("C:/Windows/Fonts/segoeui.ttf")


class RecordingOcr:
    """Fake OCR: records crop sizes and returns one short line near the crop's top-right."""

    def __init__(self):
        self.calls = []

    def read(self, image):
        self.calls.append(image.size)
        return "line", [{"box": [0.80, 0.05, 0.90, 0.08], "text": f"line {len(self.calls)}", "conf": 0.9}]


def frame(boxes=()):
    img = Image.new("RGB", (1920, 1080), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(box, fill=(230, 230, 230))
    return img


def test_unchanged_screen_costs_no_ocr():
    ocr = RecordingOcr()
    reader = ScreenReader(ocr)
    assert reader.read(frame()).mode == "first"
    assert reader.read(frame()).mode == "unchanged"
    assert len(ocr.calls) == 1


def test_small_change_reads_only_the_changed_area():
    ocr = RecordingOcr()
    reader = ScreenReader(ocr)
    reader.read(frame())
    result = reader.read(frame([(300, 500, 700, 530)]))  # one new chat line
    assert result.mode == "partial" and result.crops == 1
    # A 400x30 px line touches 4x2 tiles; with a one-tile margin that is ~19% of the screen.
    w, h = ocr.calls[-1]
    assert w * h < 0.25 * 1920 * 1080
    # The new reading is mapped back to full-screen coordinates, inside the re-read area.
    assert any(0.2 < s.box[1] < 0.5 for s in result.spans)


def test_large_change_falls_back_to_one_full_read():
    ocr = RecordingOcr()
    reader = ScreenReader(ocr)
    reader.read(frame())
    result = reader.read(frame([(0, 0, 1900, 1000)]))
    assert result.mode == "full" and ocr.calls[-1] == (1920, 1080)


def test_tile_grid_and_rects():
    prev = np.zeros((135, 240), dtype=np.int16)
    cur = prev.copy()
    cur[40:44, 30:90] = 200
    grid = changed_tiles(prev, cur)
    assert grid.sum() >= 1
    (r0, c0, r1, c1), = tile_rects(grid)
    assert r0 <= 2 <= r1 and c0 <= 1 and c1 >= 5


def _line(x, y0, y1, text):
    return {"box": [x, y0, x + 0.2, y1], "text": text, "conf": 0.9}


def test_header_and_message_merge_but_adjacent_messages_do_not():
    # Gaps measured on real PP-OCRv6 boxes: within a message ~0h, between messages ~0.7h.
    h = 0.03
    lines = [
        _line(0.15, 0.40, 0.40 + h, "Dilan"),
        _line(0.15, 0.40 + h, 0.40 + 2 * h, "nobody likes you, just leave the group"),
        _line(0.15, 0.40 + 2.7 * h, 0.40 + 3.7 * h, "Nethmi"),
        _line(0.15, 0.40 + 3.7 * h, 0.40 + 4.7 * h, "someone told me to kys in dm and im scared"),
    ]
    texts = [s.text for s in group_blocks(lines)]
    assert texts == ["Dilan nobody likes you, just leave the group",
                     "Nethmi someone told me to kys in dm and im scared"]


def test_side_by_side_columns_stay_separate():
    lines = [_line(0.01, 0.10, 0.13, "# general"), _line(0.15, 0.10, 0.13, "hello everyone")]
    assert len(group_blocks(lines)) == 2


def test_help_seeking_reply_does_not_hide_the_abuse_above_it(lexicon_scorer):
    """Regression: one block for both messages let the reply's framing discount the bullying."""
    h = 0.03
    lines = [
        _line(0.15, 0.40, 0.40 + h, "Dilan"),
        _line(0.15, 0.40 + h, 0.40 + 2 * h, "nobody likes you, just leave the group"),
        _line(0.15, 0.40 + 2.7 * h, 0.40 + 3.7 * h, "Nethmi"),
        _line(0.15, 0.40 + 3.7 * h, 0.40 + 4.7 * h, "someone told me to kys in dm and im scared"),
    ]
    v = Analyzer(scorer=lexicon_scorer).analyze(Observation(group_blocks(lines), source="screen"), age=12)
    assert v.rung == "L0" or v.top_category == "bullying"
    v = Analyzer(scorer=lexicon_scorer).analyze(Observation(group_blocks(lines), source="screen"), age=9)
    assert v.top_category == "bullying" and v.alerts



# -- chat headers: the sender and time are not part of the message ------------------

def _message(header_lines, body):
    """A header (list of (x0, x1, text) on one row) above one line of message text."""
    h = 0.03
    lines = [{"box": [x0, 0.40, x1, 0.40 + h], "text": t, "conf": 0.9} for x0, x1, t in header_lines]
    return lines + [_line(0.15, 0.40 + h, 0.40 + 2 * h, body)]


@pytest.mark.parametrize("header", [
    [(0.15, 0.26, "Dilan Today at 4:12 PM")],
    [(0.15, 0.25, "Dilan Perera 9:41 AM")],
    [(0.15, 0.27, "Nethmi Yesterday at 21:05")],
    [(0.15, 0.27, "xx_gamer_99 18/10/2026 16:12")],
    [(0.15, 0.18, "Kavin"), (0.181, 0.23, "Today at 4:12 PM")],  # OCR split the name from the time
])
def test_the_chat_header_is_left_out_of_the_scored_text_but_not_the_box(header):
    spans = group_blocks(_message(header, "you're worthless and everyone knows it"))
    (span,) = [s for s in spans if "worthless" in s.text]
    assert span.text == "you're worthless and everyone knows it"
    assert span.box[1] == 0.40  # the header is still blurred with its message
    # A name OCR split off on its own (no overlap with the time) is a block of its own, never the message's.
    assert all(s.text in ("you're worthless and everyone knows it", "Kavin") for s in spans)


def test_a_time_inside_the_bubble_is_dropped_but_a_short_message_beside_it_is_not():
    h = 0.03
    lines = [_line(0.03, 0.40, 0.40 + h, "kill urself"), {"box": [0.09, 0.405, 0.11, 0.40 + h], "text": "16:12", "conf": 0.9}]
    assert [s.text for s in group_blocks(lines)] == ["kill urself"]
    assert [s.text for s in group_blocks([_line(0.03, 0.40, 0.43, "you're worthless and everyone knows it 16:12")])] == [
        "you're worthless and everyone knows it"]


@pytest.mark.parametrize("first, second", [
    ("i'll be outside at 4:30", "come alone"),
    ("see you at 7:30", "dont tell your mum"),
    ("meet me today at 4:30", "dont tell anyone"),
])
def test_a_message_that_ends_in_a_time_is_never_taken_for_a_header(first, second):
    (span,) = group_blocks(_message([(0.15, 0.30, first)], second))
    assert span.text == f"{first} {second}"


def test_a_name_without_a_time_stays_because_it_cannot_be_told_from_a_short_message():
    (span,) = group_blocks(_message([(0.15, 0.20, "loser")], "nobody wants you here"))
    assert span.text == "loser nobody wants you here"


def test_a_block_that_is_only_a_header_gives_no_span():
    assert group_blocks([_line(0.15, 0.40, 0.43, "Today at 4:12 PM")]) == []

# -- real OCR on a rendered chat window -----------------------------------------------

needs_ocr = pytest.mark.skipif(not FONT.exists() or not OcrEngine().available, reason="OCR or font unavailable")


def chat(messages):
    font = ImageFont.truetype(str(FONT), 22)
    img = Image.new("RGB", (1920, 1080), (54, 57, 63))
    draw = ImageDraw.Draw(img)
    y = 100
    for who, text in messages:
        draw.text((300, y), who, fill=(88, 101, 242), font=font)
        draw.text((300, y + 30), text, fill=(220, 221, 222), font=font)
        y += 90
    return img


@needs_ocr
def test_real_chat_window_new_message_is_read_partially_and_flagged(lexicon_scorer):
    base = [("Kavin", "anyone on minecraft tonight?"), ("Nethmi", "yes after homework")]
    reader, analyzer = ScreenReader(), Analyzer(scorer=lexicon_scorer)
    reader.read(chat(base))
    result = reader.read(chat(base + [("Dilan", "nobody likes you, just leave the group")]))
    assert result.mode == "partial"
    assert any("nobody likes you" in s.text for s in result.spans)
    v = analyzer.analyze(Observation(result.spans, source="screen"), age=9)
    assert v.top_category == "bullying" and v.alerts and len(v.regions) == 1

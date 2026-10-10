"""OCR engine (proposal stage 2 "Read", FR1, NFR8)."""
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from component2_hate_speech_detection.src.readers.ocr import OcrEngine

FONT = Path("C:/Windows/Fonts/segoeui.ttf")


def test_missing_image_returns_nothing_and_never_raises():
    assert OcrEngine().read(None) == ("", [])


@pytest.mark.skipif(not FONT.exists() or not OcrEngine().available, reason="OCR or font unavailable")
def test_reads_game_chat_text_with_normalised_boxes():
    img = Image.new("RGB", (1000, 220), (20, 20, 30))
    ImageDraw.Draw(img).text((40, 90), "go back to your country", fill=(240, 240, 240),
                             font=ImageFont.truetype(str(FONT), 30))
    text, regions = OcrEngine().read(img)
    assert "back to your country" in text.lower()
    x0, y0, x1, y1 = regions[0]["box"]
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1

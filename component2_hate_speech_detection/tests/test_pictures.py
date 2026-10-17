"""Pictures on screen: finding memes and photos, and scoring each one once (SO3, FR5)."""
import numpy as np
from PIL import Image, ImageDraw

from component2_hate_speech_detection.src.readers.pictures import PictureReader, find_pictures, text_inside

DARK, LIGHT = (49, 51, 56), (240, 242, 245)


def noise(w, h, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8))


def page(bg=DARK, size=(1280, 720)):
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    for i in range(6):                                    # chat lines around the picture
        draw.text((80, 40 + 30 * i), "see you at practice later tonight", fill=(220, 220, 220))
    return img


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union


def test_finds_a_photo_posted_in_a_chat():
    img = page()
    img.paste(noise(320, 240), (400, 300))
    boxes = find_pictures(img)
    assert len(boxes) == 1 and iou(boxes[0], (400, 300, 720, 540)) > 0.9


def test_a_chat_with_only_text_has_no_pictures():
    img = page()
    draw = ImageDraw.Draw(img)
    for i in range(14):
        draw.text((80, 260 + 26 * i), "nobody likes you just leave the group " * 3, fill=(220, 220, 220))
    assert find_pictures(img) == []


def test_a_gallery_comes_apart_at_its_gutters():
    img = Image.new("RGB", (1280, 720), LIGHT)
    cells = [(300 + c * 208, 100 + r * 208) for r in range(2) for c in range(2)]
    for i, (x, y) in enumerate(cells):
        img.paste(noise(200, 200, seed=i), (x, y))
    boxes = find_pictures(img)
    assert len(boxes) == 4
    for x, y in cells:
        assert max(iou((x, y, x + 200, y + 200), b) for b in boxes) > 0.9


def test_a_studio_photo_with_a_white_backdrop_stays_whole_on_a_dark_page():
    photo = Image.new("RGB", (300, 300), (255, 255, 255))
    photo.paste(noise(80, 200), (110, 60))               # the subject; the rest is a flat white backdrop
    img = page()
    img.paste(photo, (500, 260))
    boxes = find_pictures(img)
    assert len(boxes) == 1 and iou(boxes[0], (500, 260, 800, 560)) > 0.9


def test_caption_is_the_ocr_lines_inside_the_picture_top_to_bottom():
    lines = [{"box": [0.4, 0.8, 0.6, 0.85], "text": "bottom text"},
             {"box": [0.4, 0.45, 0.6, 0.5], "text": "top text"},
             {"box": [0.0, 0.0, 0.2, 0.05], "text": "menu"}]
    assert text_inside((400, 300, 800, 700), lines, (1000, 800)) == "top text bottom text"


class Counting:
    name, calibrated = "fake-meme", False

    def __init__(self, scores):
        self.scores, self.calls = list(scores), []

    def probability(self, image, caption=""):
        self.calls.append(image.size)
        return self.scores[len(self.calls) - 1]


def test_each_picture_is_scored_once_and_the_strongest_is_reported():
    img = page()
    img.paste(noise(260, 200, seed=1), (200, 300))
    img.paste(noise(260, 200, seed=2), (700, 300))
    scorer = Counting([0.3, 0.7])
    reader = PictureReader(scorer)
    first = reader.read(img, [])
    assert first.name == "vision" and first.score == 0.7 and first.calibrated is False
    assert reader.read(img, []).score == 0.7 and len(scorer.calls) == 2   # second read: both from cache


def test_a_dropped_meme_is_scored_whole_and_a_blank_image_gives_no_picture():
    scorer = Counting([0.6])
    meme = noise(500, 400)
    assert PictureReader(scorer).read(meme, [], whole_image=True).score == 0.6
    assert scorer.calls == [(500, 400)]
    assert PictureReader(Counting([])).read(page(), []) is None

"""Text on screen vs typed: is a chat message judged the same when C2 reads it off the screen?

    python -m component2_hate_speech_detection.evaluation.screen_text --styles design    # tuning
    python -m component2_hate_speech_detection.evaluation.screen_text --styles heldout   # report once
    python -m component2_hate_speech_detection.evaluation.screen_text --styles all --write

Each of the 107 dev + held-out text cases is rendered as one message in a
1920x1080 chat screenshot between three ordinary messages, read by the
ScreenReader (change-gated OCR + block grouping) and decided on the whole
screen at ages 9, 12 and 15. The same text typed is the reference: the
screen path should give the same rung.

Chat styles (screens are built, so the true message is known):

    design    chat-dark-22   Discord-like, "Name  Today at 4:12 PM" above each message, 150 % zoom
              chat-dark-16   the same at 100 % zoom
              chat-light-20  a light web chat with the same headers
    held-out  bubbles-1to1   WhatsApp-like 1:1 chat: no names, "16:12" inside the bubble
              team-chat      Slack/Teams-like: "Dilan Perera  9:41 AM" headers

Reader rules may be tuned on the design styles only; the held-out styles
were written on 18 Oct 2026, before the header fix, and are reported once.

Results, 18 Oct 2026 (321 decisions per style; `results/screen_text.json`).
Before: the chat header was scored as part of the message. After: it is
left out of the scored text (`src/readers/screen.py`, `message_text`).

    style           split      same rung as typed       message read exactly
                               before      after        before     after
    chat-dark-22    design     94.7 %      99.4 %       0 / 107    97 / 107
    chat-dark-16    design     93.8 %      99.1 %       0 / 107    96 / 107
    chat-light-20   design     93.8 %      98.8 %       0 / 107    99 / 107
    bubbles-1to1    held-out   97.5 %      97.5 %      85 / 107    86 / 107
    team-chat       held-out   97.2 %     100.0 %       0 / 107    98 / 107

What is left is OCR misreading the text itself: stretched or starred
letters ("kiiiill" -> "kifll", "k*y*s" -> "kxyts"), letter-spaced words, and
once a dropped letter ("don't ell your parents"). Benign alerts on screen now
equal typed ones in every style (3 = one benign message at three ages, "my
brother keeps annoying me i hate it", a text-model false alert the header used
to hide).
"""
from __future__ import annotations

import argparse
import difflib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..src.engine.analyzer import Analyzer, Observation
from ..src.readers.screen import ScreenReader
from .dev_set import all_cases as dev_cases
from .heldout_set import all_cases as heldout_cases

AGES = (9, 12, 15)
RESULTS = Path(__file__).resolve().parent / "results" / "screen_text.json"
FONT = "C:/Windows/Fonts/segoeui.ttf"
BOLD = "C:/Windows/Fonts/segoeuib.ttf"
W, H = 1920, 1080

Message = Tuple[str, str, str]  # (sender, text, time); sender "me" = the child


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _discord(size: int, bg, name_c, text_c, meta_c) -> Callable[[List[Message]], Image.Image]:
    def render(messages: List[Message]) -> Image.Image:
        regular, bold, small = _font(FONT, size), _font(BOLD, size), _font(FONT, max(11, size - 6))
        img = Image.new("RGB", (W, H), bg)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 240, H], fill=tuple(max(0, c - 12) for c in bg))
        for i, ch in enumerate(["# general", "# homework-help", "# minecraft", "# memes"]):
            d.text((24, 80 + i * 40), ch, fill=meta_c, font=regular)
        d.text((280, 24), "# general", fill=text_c, font=bold)
        y = 120
        for who, text, _ in messages:
            d.text((300, y), who, fill=name_c, font=bold)
            d.text((300 + bold.getlength(who) + 12, y + 2), "Today at 4:12 PM", fill=meta_c, font=small)
            d.text((300, y + int(size * 1.4)), text, fill=text_c, font=regular)
            y += int(size * 4.2)
        d.rectangle([280, 990, 1880, 1050], fill=tuple(min(255, c + 10) for c in bg))
        d.text((300, 1005), "Message #general", fill=meta_c, font=regular)
        return img
    return render


def _bubbles(messages: List[Message]) -> Image.Image:
    """WhatsApp-like 1:1 chat: the friend's bubbles on the left, the child's on the right, time inside."""
    regular, small = _font(FONT, 18), _font(FONT, 13)
    img = Image.new("RGB", (W, H), (239, 234, 226))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 70], fill=(240, 242, 245))
    d.text((90, 22), "Dilan", fill=(17, 27, 33), font=_font(BOLD, 20))
    d.text((90, 46), "online", fill=(102, 119, 129), font=small)
    y = 120
    for who, text, when in messages:
        width = int(regular.getlength(text)) + 80
        x0 = 1880 - width if who == "me" else 60
        d.rounded_rectangle([x0, y, x0 + width, y + 44], radius=8,
                            fill=(217, 253, 211) if who == "me" else (255, 255, 255))
        d.text((x0 + 12, y + 10), text, fill=(17, 27, 33), font=regular)
        d.text((x0 + width - 46, y + 24), when, fill=(102, 119, 129), font=small)
        y += 70
    d.rectangle([0, 1010, W, H], fill=(240, 242, 245))
    d.text((90, 1035), "Type a message", fill=(102, 119, 129), font=regular)
    return img


def _team(messages: List[Message]) -> Image.Image:
    """Slack/Teams-like: avatar, full name, short time, message below."""
    regular, bold, small = _font(FONT, 17), _font(BOLD, 17), _font(FONT, 13)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 260, H], fill=(63, 14, 64))
    for i, ch in enumerate(["# class-7b", "# science", "# random"]):
        d.text((24, 90 + i * 34), ch, fill=(207, 195, 207), font=regular)
    d.text((290, 24), "# class-7b", fill=(29, 28, 29), font=bold)
    y = 110
    for who, text, when in messages:
        d.rounded_rectangle([290, y, 326, y + 36], radius=6, fill=(46, 182, 125))
        d.text((340, y - 2), who, fill=(29, 28, 29), font=bold)
        d.text((340 + bold.getlength(who) + 10, y + 1), when, fill=(97, 96, 97), font=small)
        d.text((340, y + 22), text, fill=(29, 28, 29), font=regular)
        y += 78
    d.rounded_rectangle([290, 990, 1890, 1050], radius=8, outline=(187, 187, 187))
    d.text((310, 1008), "Message #class-7b", fill=(118, 118, 118), font=regular)
    return img


STYLES: Dict[str, Tuple[str, Callable[[List[Message]], Image.Image]]] = {
    "chat-dark-22": ("design", _discord(22, (49, 51, 56), (88, 101, 242), (219, 222, 225), (148, 155, 164))),
    "chat-dark-16": ("design", _discord(16, (49, 51, 56), (88, 101, 242), (219, 222, 225), (148, 155, 164))),
    "chat-light-20": ("design", _discord(20, (255, 255, 255), (0, 120, 80), (17, 27, 33), (102, 119, 129))),
    "bubbles-1to1": ("heldout", _bubbles),
    "team-chat": ("heldout", _team),
}
# The ordinary messages around the case; per style, the senders and times that style shows.
CONTEXT = {
    "design": ([("Kavin", "anyone on minecraft tonight?", ""), ("Nethmi", "yes after homework", "")],
               [("Kavin", "ok see you at 7", "")]),
    "bubbles-1to1": ([("Dilan", "did you finish the maths sheet", "16:05"), ("me", "nearly, question 4 is hard", "16:07")],
                     [("me", "ok talk later", "16:14")]),
    "team-chat": ([("Kavin Silva", "anyone on minecraft tonight?", "9:38 AM"), ("Nethmi Fernando", "yes after homework", "9:40 AM")],
                  [("Kavin Silva", "ok see you at 7", "9:43 AM")]),
}
TARGET = {"design": ("Dilan", "16:12"), "bubbles-1to1": ("Dilan", "16:12"), "team-chat": ("Dilan Perera", "9:41 AM")}


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def evaluate_style(style: str, analyzer: Analyzer, typed: dict, cases) -> dict:
    split, render = STYLES[style]
    key = style if style in CONTEXT else "design"
    before, after = CONTEXT[key]
    who, when = TARGET[key if key in TARGET else "design"]
    context_rungs = [analyzer.analyze(Observation(ScreenReader().read(render(before + after)).spans, source="screen"), a).rung
                     for a in AGES]
    stats = defaultdict(lambda: [0, 0])
    agree, exact, sims, ocr_ms, diffs = 0, 0, [], [], []
    for i, (c, src) in enumerate(cases):
        frame = render(before + [(who, c.text, when)] + after)
        t = time.perf_counter()
        read = ScreenReader().read(frame)
        ocr_ms.append((time.perf_counter() - t) * 1000)
        best = max(((difflib.SequenceMatcher(None, _norm(s.text), _norm(c.text)).ratio(), s.text) for s in read.spans),
                   default=(0.0, ""))
        sims.append(best[0])
        exact += _norm(best[1]) == _norm(c.text)
        for a in AGES:
            v, ref = analyzer.analyze(Observation(read.spans, source="screen"), a), typed[(i, a)]
            label = "harmful" if c.hate else "benign"
            stats[(src, label, a, "screen")][0] += v.rung != "L0"
            stats[(src, label, a, "typed")][0] += ref.rung != "L0"
            stats[(src, label, a, "screen")][1] += 1
            stats[(src, label, a, "typed")][1] += 1
            agree += v.rung == ref.rung
            if v.rung != ref.rung:
                diffs.append({"age": a, "bucket": c.bucket, "text": c.text, "typed": ref.rung, "screen": v.rung,
                              "read_as": best[1]})
    n = len(cases) * len(AGES)
    return {
        "style_split": split,
        "context_only_rungs": context_rungs,
        "message_read_exactly": f"{exact}/{len(cases)}",
        "median_similarity": round(sorted(sims)[len(sims) // 2], 3),
        "median_full_frame_ms": round(sorted(ocr_ms)[len(ocr_ms) // 2]),
        "screen_rung_equals_typed": f"{agree}/{n}",
        "agreement": round(agree / n, 4),
        "alerted": {f"{src} {label} age {a}": f"screen {stats[(src, label, a, 'screen')][0]}/{stats[(src, label, a, 'screen')][1]}"
                                              f" typed {stats[(src, label, a, 'typed')][0]}"
                    for src in ("dev", "heldout") for label in ("harmful", "benign") for a in AGES},
        "differences": diffs,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--styles", choices=("design", "heldout", "all"), default="design")
    parser.add_argument("--write", action="store_true", help=f"save to {RESULTS.name}")
    args = parser.parse_args(argv)
    cases = [(c, "dev") for c in dev_cases()] + [(c, "heldout") for c in heldout_cases()]
    analyzer = Analyzer()
    analyzer.scorer.warm_up()
    typed = {(i, a): analyzer.analyze_text(c.text, a) for i, (c, _) in enumerate(cases) for a in AGES}
    report = {"engine": analyzer.scorer.name, "policy_version": analyzer.policy.version, "styles": {}}
    for style, (split, _) in STYLES.items():
        if args.styles in (split, "all"):
            report["styles"][style] = evaluate_style(style, analyzer, typed, cases)
            r = report["styles"][style]
            print(f"{style:14s} {split:8s} agree {r['screen_rung_equals_typed']} ({r['agreement']:.1%}), "
                  f"read exactly {r['message_read_exactly']}, context {r['context_only_rungs']}", flush=True)
    if args.write:
        RESULTS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

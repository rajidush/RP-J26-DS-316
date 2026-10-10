"""OCR engine for screen text (proposal stage 2 "Read", FR1, SO3).

Ported from the J26-DS-316 MVP (`analyst/extract/ocr.py`), keeping its
measured settings. RapidOCR runs PP-OCR models on ONNX Runtime, so it shares
the runtime the text heads already use.

Settings, and the measurements behind them (MVP, 1-2 Sep 2026):
- Cap the *longer* side of the detection input. RapidOCR's default caps the
  shorter side, which turned a 1000x220 chat bar into a 3345x736 input:
  18.8 s per frame before, 1.3 s after.
- PP-OCRv6 tiny. On a real 1100 px browser capture with 12 known phrases it
  read 12/12 in 738 ms, against v5 mobile's 9/12 in 2,951 ms.
- No angle classifier (screen text is upright) and no sharpening (unsharp
  masking amplified JPEG ringing and lost 2 of 4 phrases on meme thumbnails).
- Detection cap 1600 px. Raising the cap alone bought nothing; the earlier
  loss was in the capture encoder. Keep it at or above the capture width.

The engine never raises into the caller: a missing or broken install returns
no text, and the rest of C2 keeps running (proposal NFR8).
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

_MIN_WIDTH = 640
_DET_SIDE_LEN = int(os.environ.get("C2_OCR_SIDE_LEN", "1600"))
_OCR_THREADS = int(os.environ.get("C2_OCR_THREADS", "2"))

Region = dict  # {"box": [x0, y0, x1, y1] normalised 0..1, "text": str, "conf": float}


def _rapid3_params() -> dict:
    from rapidocr import ModelType, OCRVersion

    return {
        "Det.limit_type": "max",
        "Det.limit_side_len": _DET_SIDE_LEN,
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Det.model_type": ModelType.TINY,
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.model_type": ModelType.TINY,
        "EngineConfig.onnxruntime.intra_op_num_threads": _OCR_THREADS,
        "Global.use_cls": False,
    }


def _prepare(image: Image.Image) -> np.ndarray:
    """Upscale very small crops so glyphs stay above the recogniser's ~9 px floor."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w < _MIN_WIDTH:
        scale = _MIN_WIDTH / w
        rgb = rgb.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.asarray(rgb)


def _regions(polys, lines, confs, width: int, height: int) -> List[Region]:
    if polys is None or width <= 0 or height <= 0:
        return []
    out: List[Region] = []
    for i, poly in enumerate(polys):
        try:
            xs = [float(pt[0]) for pt in poly]
            ys = [float(pt[1]) for pt in poly]
        except Exception:
            continue
        text = (lines[i] if i < len(lines) else "") or ""
        if not text.strip():
            continue
        out.append({
            "box": [
                round(max(0.0, min(1.0, min(xs) / width)), 4),
                round(max(0.0, min(1.0, min(ys) / height)), 4),
                round(max(0.0, min(1.0, max(xs) / width)), 4),
                round(max(0.0, min(1.0, max(ys) / height)), 4),
            ],
            "text": text[:200],
            "conf": round(float(confs[i]), 3) if i < len(confs) else 0.0,
        })
    return out


class OcrEngine:
    def __init__(self) -> None:
        self.name = "none"
        self.last_error = ""
        self._rapid = None
        try:
            from rapidocr import RapidOCR

            self._rapid = RapidOCR(params=_rapid3_params())
            self.name = "rapidocr-ppocrv6-tiny"
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]

    @property
    def available(self) -> bool:
        return self._rapid is not None

    def read(self, image: Optional[Image.Image]) -> Tuple[str, List[Region]]:
        """One OCR pass -> (joined text, per-line regions normalised to this image)."""
        if image is None or self._rapid is None:
            return "", []
        try:
            arr = _prepare(image)
            raw = self._rapid(arr)
            height, width = arr.shape[0], arr.shape[1]
            lines = list(getattr(raw, "txts", None) or [])
            polys = getattr(raw, "boxes", None)
            confs = list(getattr(raw, "scores", None) or [])
            regions = _regions(polys, lines, confs, width, height)
            return " ".join(r["text"] for r in regions), regions
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            return "", []

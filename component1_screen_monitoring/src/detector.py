"""
detector.py — Violence detector for Component 1 (Screen Monitoring).

Loads the fine-tuned ViT checkpoint produced by
notebooks/week2_finetune_violence.ipynb and classifies a single frame
(PIL image) as "violent" or "safe".

The checkpoint is NOT in git (≈330 MB). Download violence-vit-screen-ft-v1.zip
from the GitHub Release / Drive and unzip it to:

    component1_screen_monitoring/models/violence-vit-screen-ft-v1/

Requires transformers<5 (the checkpoint was saved with 4.57).

CLI
---
    python src/detector.py path/to/image.png      # classify one image
    python src/detector.py --benchmark 20         # capture + classify 20 live frames, report latency
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = COMPONENT_ROOT / "models" / "violence-vit-screen-ft-v1"


@dataclass
class Detection:
    """Result of classifying one frame."""

    category: str          # "violence" if flagged, else "safe"
    confidence: float      # probability of the violent class, 0..1
    flagged: bool          # confidence >= threshold
    inference_ms: float    # model time for this frame (preprocess + forward)


def _pick_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"            # Apple-silicon GPU
    return "cpu"


class ViolenceDetector:
    """Wraps the fine-tuned ViT. Load once, call predict() per frame."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import ViTForImageClassification, ViTImageProcessor

        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"No model found at {model_dir}. Download violence-vit-screen-ft-v1.zip "
                "and unzip it there (see the docstring at the top of detector.py)."
            )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        self.threshold = threshold
        self.device = _pick_device(device)
        self.processor = ViTImageProcessor.from_pretrained(model_dir)
        self.model = ViTForImageClassification.from_pretrained(model_dir).to(self.device).eval()

        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        if "violent" not in label2id:
            raise ValueError(f"Checkpoint labels {label2id} have no 'violent' class")
        self.violent_idx = int(label2id["violent"])
        self._torch = torch

    def predict(self, image: Any) -> Detection:
        """Classify one PIL image (any mode/size)."""
        torch = self._torch
        t0 = time.perf_counter()
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            probs = self.model(**inputs).logits.float().softmax(-1)[0]
        p_violent = float(probs[self.violent_idx].cpu())
        if self.device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000

        flagged = p_violent >= self.threshold
        return Detection(
            category="violence" if flagged else "safe",
            confidence=round(p_violent, 4),
            flagged=flagged,
            inference_ms=round(ms, 1),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _benchmark(detector: ViolenceDetector, n: int) -> None:
    try:
        from .capture import capture_frame          # run as module
    except ImportError:
        from capture import capture_frame           # run as script from src/

    detector.predict(capture_frame(monitor_index=1).image)   # warm-up
    cap_ms, inf_ms = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        frame = capture_frame(monitor_index=1)
        cap_ms.append((time.perf_counter() - t0) * 1000)
        inf_ms.append(detector.predict(frame.image).inference_ms)

    total = [c + i for c, i in zip(cap_ms, inf_ms)]
    print(f"Device: {detector.device}   frames: {n}   frame size: {frame.metadata['width']}x{frame.metadata['height']}")
    for name, xs in [("capture", cap_ms), ("inference", inf_ms), ("total", total)]:
        print(f"  {name:<10} mean {statistics.mean(xs):7.1f} ms   median {statistics.median(xs):7.1f} ms   max {max(xs):7.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="Component 1 violence detector")
    ap.add_argument("image", nargs="?", help="image file to classify")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default=None, help="cpu / mps / cuda (default: auto)")
    ap.add_argument("--benchmark", type=int, metavar="N", help="capture + classify N live frames")
    args = ap.parse_args()

    det = ViolenceDetector(args.model_dir, threshold=args.threshold, device=args.device)
    if args.benchmark:
        _benchmark(det, args.benchmark)
    elif args.image:
        from PIL import Image
        print(det.predict(Image.open(args.image)))
    else:
        ap.error("give an image path or --benchmark N")


if __name__ == "__main__":
    main()

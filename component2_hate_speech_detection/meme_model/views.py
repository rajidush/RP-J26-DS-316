"""Meme head v2 inputs: every meme as C2 actually reads it, embedded once with frozen CLIP.

    python -m component2_hate_speech_detection.meme_model.views --workers 16
    python -m component2_hate_speech_detection.meme_model.views --encoder clip-l14

Three views of each train, val and test meme:
    clean    the original image + the dataset's caption (v1 trained on this only)
    ocr      the original image + C2's own OCR of it (a dropped meme)
    screen   the meme on a built feed, chat or article page, cut out by C2's picture
             finder and captioned with the OCR lines inside the crop (a live screen)

Pages come from evaluation/screen_memes.py's builder. Each meme gets its own page
seed (split seed + meme id), so the work runs in parallel and is reproducible, and
train pages never repeat a val or test page. If the finder misses a meme, training
uses its true box; the row is flagged found = 0 so evaluation can score it 0, as
screen_memes.py does.

Step 1, CPU, one process per core: OCR + pages -> data/views/captions.jsonl and
        data/views/crops/<id>.png (appended as it goes, so a stopped run resumes).
        If captions.jsonl is already complete (e.g. uploaded from an earlier run), only
        the crops are rebuilt from the seeded pages, with no OCR, so every encoder sees
        exactly the same views.
Step 2, GPU if present: frozen CLIP (--encoder, default ViT-B/32) ->
        data/embeddings/<encoder>-views.npz
Everything written lives in git-ignored folders.
"""
from __future__ import annotations

import os

os.environ.setdefault("C2_OCR_THREADS", "1")  # parallelism comes from processes, one OCR thread each

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .embed import DATA_DIR, EMBED_DIR, ENCODERS, IMAGE_ROOT, MANIFEST_DIR, read_manifest

SPLITS = ("train", "val", "test")
PAGE_SEED = {"train": 1316, "val": 2316, "test": 3316}
LAYOUTS = ("feed-light", "feed-dark", "chat-dark", "article")
VIEW_DIR = DATA_DIR / "views"
CROP_DIR = VIEW_DIR / "crops"
CAPTIONS = VIEW_DIR / "captions.jsonl"

_ocr = None


def tag(encoder: str) -> str:
    """File-name suffix: none for B/32 (the original run's names), else e.g. '_clip_l14'."""
    return "" if encoder == "clip-b32" else "_" + encoder.replace("-", "_")


def views_npz(encoder: str) -> Path:
    return EMBED_DIR / f"{encoder}-views.npz"


def stats_file(encoder: str) -> Path:
    return VIEW_DIR / f"views_stats{tag(encoder)}.json"


def _init_worker() -> None:
    global _ocr
    from ..src.readers.ocr import OcrEngine

    _ocr = OcrEngine()


def make_views(task: tuple) -> dict:
    """OCR caption + screen crop for one meme (runs in a worker process)."""
    from ..evaluation.screen_memes import build, iou
    from ..src.readers.pictures import find_pictures, text_inside
    from ..src.readers.screen import ScreenReader

    split, row = task
    meme_id = int(row["id"])
    with Image.open(IMAGE_ROOT / row["img"]) as src:
        meme = src.convert("RGB")
    ocr_caption = _ocr.read(meme)[0]
    rng = np.random.default_rng([PAGE_SEED[split], meme_id])
    layout = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
    page = build(layout, [meme_id], rng)
    truth = page.pictures[0][0] if page.pictures else None
    reader = ScreenReader(_ocr)
    reader.read(page.img)
    boxes = find_pictures(page.img)
    best = max(boxes, key=lambda b: iou(truth, b), default=None) if truth else None
    found = best is not None and iou(truth, best) >= 0.5
    box = best if found else truth
    if box is None:  # the meme ended up mostly off the page: use the meme itself
        crop, caption = meme, ocr_caption
    else:
        crop, caption = page.img.crop(box), text_inside(box, reader.lines, page.img.size)
    crop.save(CROP_DIR / f"{meme_id:05d}.png")
    return {"id": meme_id, "split": split, "label": int(row["label"]), "layout": layout, "ocr": ocr_caption,
            "screen": caption, "found": bool(found), "box": [int(v) for v in box] if box else None}


def remake_crop(task: tuple) -> dict:
    """Rebuild one meme's screen crop from its seeded page and stored box (no OCR)."""
    from ..evaluation.screen_memes import build, iou

    split, row, rec = task
    meme_id = int(row["id"])
    rng = np.random.default_rng([PAGE_SEED[split], meme_id])  # same draws, same order as make_views
    layout = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
    page = build(layout, [meme_id], rng)
    truth = page.pictures[0][0] if page.pictures else None
    if rec["box"] is None:
        with Image.open(IMAGE_ROOT / row["img"]) as src:
            crop = src.convert("RGB")
        same_page = truth is None
    else:
        crop = page.img.crop(tuple(rec["box"]))
        same_page = truth is not None and iou(truth, rec["box"]) >= (0.5 if rec["found"] else 0.999)
    crop.save(CROP_DIR / f"{meme_id:05d}.png")
    return {"id": meme_id, "same_page": bool(same_page and layout == rec["layout"])}


def all_rows(limit: Optional[int] = None) -> list[tuple]:
    return [(s, r) for s in SPLITS for r in read_manifest(MANIFEST_DIR / f"{s}.jsonl")[:limit]]


def read_captions() -> dict[int, dict]:
    if not CAPTIONS.exists():
        return {}
    out = {}
    for line in CAPTIONS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = rec  # a re-run line replaces the earlier one
    return out


def build_views(workers: int, limit: Optional[int] = None) -> dict:
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    done = read_captions()
    rows = all_rows(limit)
    has_crop = lambda r: (CROP_DIR / f"{int(r['id']):05d}.png").exists()
    todo = [(s, r) for s, r in rows if int(r["id"]) not in done]
    remake = [(s, r, done[int(r["id"])]) for s, r in rows if int(r["id"]) in done and not has_crop(r)]
    print(f"views: {len(todo)} memes to do, {len(remake)} crops to rebuild from stored captions "
          f"({len(done)} captions already done), {workers} workers", flush=True)
    started = time.perf_counter()
    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    report = {"memes": len(todo), "crops_rebuilt": len(remake)}
    if remake:
        with ctx.Pool(workers) as pool:
            same = [r["same_page"] for r in pool.imap_unordered(remake_crop, remake, chunksize=8)]
        report["crops_rebuilt_on_same_page"] = round(sum(same) / len(same), 4)
        print(f"  rebuilt {len(remake)} crops; {sum(same)}/{len(same)} pages match the stored layout and box", flush=True)
    if todo:
        from ..src.readers.ocr import OcrEngine

        if not OcrEngine().available:  # also downloads the OCR models once, before the workers start
            raise SystemExit("RapidOCR is not working: pip install rapidocr onnxruntime")
        with open(CAPTIONS, "a", encoding="utf-8") as out, ctx.Pool(workers, initializer=_init_worker) as pool:
            for k, rec in enumerate(pool.imap_unordered(make_views, todo, chunksize=4), 1):
                out.write(json.dumps(rec) + "\n")
                if k % 250 == 0 or k == len(todo):
                    out.flush()
                    rate = k / (time.perf_counter() - started)
                    print(f"  {k}/{len(todo)}  {rate:.1f} memes/s  ~{(len(todo) - k) / rate / 60:.0f} min left", flush=True)
    report["seconds"] = round(time.perf_counter() - started, 1)
    return report


def embed_views(batch_size: int, device_name: str = "auto", limit: Optional[int] = None, encoder: str = "clip-b32") -> dict:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    recs = read_captions()
    rows = all_rows(limit)
    missing = [int(r["id"]) for _, r in rows if int(r["id"]) not in recs]
    if missing:
        raise SystemExit(f"{len(missing)} memes have no views yet: run without --embed-only first")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    model = CLIPModel.from_pretrained(ENCODERS[encoder]).to(device).eval()
    processor = CLIPProcessor.from_pretrained(ENCODERS[encoder])

    cols = {k: [] for k in ("image_orig", "image_screen", "text_clean", "text_ocr", "text_screen")}
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            images = []
            for _, r in batch:
                with Image.open(IMAGE_ROOT / r["img"]) as src:
                    images.append(src.convert("RGB"))
            for _, r in batch:
                with Image.open(CROP_DIR / f"{int(r['id']):05d}.png") as src:
                    images.append(src.convert("RGB"))
            texts = ([r["text"] for _, r in batch] + [recs[int(r["id"])]["ocr"] for _, r in batch]
                     + [recs[int(r["id"])]["screen"] for _, r in batch])
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Pooled output -> CLIP's own projection, exactly as embed.py (v1's features).
            img = model.visual_projection(model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output).float().cpu().numpy()
            txt = model.text_projection(model.text_model(input_ids=inputs["input_ids"],
                                                         attention_mask=inputs["attention_mask"]).pooler_output).float().cpu().numpy()
            n = len(batch)
            cols["image_orig"].append(img[:n]); cols["image_screen"].append(img[n:])
            cols["text_clean"].append(txt[:n]); cols["text_ocr"].append(txt[n:2 * n]); cols["text_screen"].append(txt[2 * n:])
            done = start + n
            if (start // batch_size) % 20 == 0 or done == len(rows):
                print(f"  embedded {done}/{len(rows)}  {done / (time.perf_counter() - started):.1f} memes/s", flush=True)

    arrays = {k: np.concatenate(v).astype(np.float32) for k, v in cols.items()}
    ids = np.array([int(r["id"]) for _, r in rows], dtype=np.int64)
    meta = {"ids": ids, "split": np.array([s for s, _ in rows]), "label": np.array([int(r["label"]) for _, r in rows], dtype=np.int64),
            "found": np.array([recs[int(r["id"])]["found"] for _, r in rows], dtype=bool)}
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(views_npz(encoder), **meta, **arrays)

    stats = {"encoder": encoder, "model_id": ENCODERS[encoder], "device": str(device),
             "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
             "seconds": round(time.perf_counter() - started, 1), "dim": int(arrays["image_orig"].shape[1])}
    v1 = EMBED_DIR / f"{encoder}.npz"  # stored clean features (v1's, for B/32): the clean view must reproduce them
    if v1.exists():
        old = np.load(v1)
        pos = {int(i): k for k, i in enumerate(old["ids"])}
        idx = [pos[i] for i in ids if int(i) in pos]
        keep = np.array([int(i) in pos for i in ids])
        cos = lambda a, b: float(np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))
        stats["check_vs_v1_features"] = {"memes": int(keep.sum()),
                                         "image_mean_cosine": round(cos(arrays["image_orig"][keep], old["image"][idx]), 6),
                                         "text_mean_cosine": round(cos(arrays["text_clean"][keep], old["text"][idx]), 6)}
    return stats


def view_stats() -> dict:
    recs = list(read_captions().values())
    out = {}
    for s in SPLITS:
        r = [x for x in recs if x["split"] == s]
        if r:
            out[s] = {"memes": len(r), "finder_found": round(sum(x["found"] for x in r) / len(r), 4),
                      "ocr_caption_empty": round(sum(not x["ocr"].strip() for x in r) / len(r), 4),
                      "screen_caption_empty": round(sum(not x["screen"].strip() for x in r) / len(r), 4),
                      "layouts": {l: sum(x["layout"] == l for x in r) for l in LAYOUTS}}
    return out


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, help="first N memes per split (smoke test)")
    parser.add_argument("--embed-only", action="store_true", help="skip OCR/pages; captions.jsonl must be complete")
    parser.add_argument("--views-only", action="store_true", help="OCR/pages only, no CLIP")
    parser.add_argument("--encoder", default="clip-b32", choices=sorted(ENCODERS), help="frozen CLIP to embed with")
    args = parser.parse_args(argv)

    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "workers": args.workers, "limit": args.limit}
    if not args.embed_only:
        report["views"] = build_views(args.workers, args.limit)
    report["view_stats"] = view_stats()
    if not args.views_only:
        report["embedding"] = embed_views(args.batch_size, args.device, args.limit, args.encoder)
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    stats_file(args.encoder).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

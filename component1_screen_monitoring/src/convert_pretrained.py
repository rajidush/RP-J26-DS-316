"""
convert_pretrained.py — Save the TRUE pretrained model in Hugging Face format.

jaranohaal/vit-base-violence-detection stores timm-style weight names, so
ViTForImageClassification.from_pretrained() cannot load it directly. This applies
the same timm → HF conversion as notebooks/finetune_violence.ipynb (section 3,
the loader cell) and saves the result to models/violence-vit-pretrained-hf/
(config.PRETRAINED_MODEL_DIR), so the detector, eval_screens.py and
compare_models.py can use it like any other checkpoint. No training happens.

    labels: whichever output index is "violent" (ft-v1's training_info.json records
            it as base_violent_output_index; checked here on RLVS val frames), the
            weights are NOT re-ordered
    preprocessing: 224×224, mean/std 0.5 — the same processor the notebook used

Then checks it on the RLVS test split (1,661 frames; notebook section 4 baseline:
accuracy 0.996).

Usage
-----
    python component1_screen_monitoring/src/convert_pretrained.py            # convert (if needed) + verify
    python component1_screen_monitoring/src/convert_pretrained.py --force    # re-convert
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.config import PRETRAINED_MODEL_DIR, RLVS_FT_V1_MODEL_DIR, SPLITS_CSV

BASE_MODEL = "jaranohaal/vit-base-violence-detection"
RLVS_ROOT = Path.home() / "Desktop" / "new_violence"


def timm_to_hf(src: dict) -> tuple[dict, list[str]]:
    """timm ViT state dict → Hugging Face ViTForImageClassification names (notebook section 3)."""
    out, unknown = {}, []
    block_map = {"norm1": "layernorm_before", "norm2": "layernorm_after",
                 "attn.proj": "attention.output.dense",
                 "mlp.fc1": "intermediate.dense", "mlp.fc2": "output.dense"}
    for key, v in src.items():
        k = re.sub(r"^(model\.|module\.|base_model\.)+", "", key)
        name, _, param = k.rpartition(".")
        m = re.match(r"blocks\.(\d+)\.(.+)", name)
        if k == "cls_token":
            out["vit.embeddings.cls_token"] = v
        elif k == "pos_embed":
            out["vit.embeddings.position_embeddings"] = v
        elif name == "patch_embed.proj":
            out[f"vit.embeddings.patch_embeddings.projection.{param}"] = v
        elif name in ("norm", "fc_norm"):
            out[f"vit.layernorm.{param}"] = v
        elif name == "head":
            out[f"classifier.{param}"] = v
        elif m and m.group(2) == "attn.qkv":
            p = f"vit.encoder.layer.{m.group(1)}.attention.attention"
            q, kk, vv = v.chunk(3, dim=0)
            out[f"{p}.query.{param}"], out[f"{p}.key.{param}"], out[f"{p}.value.{param}"] = q, kk, vv
        elif m and m.group(2) in block_map:
            out[f"vit.encoder.layer.{m.group(1)}.{block_map[m.group(2)]}.{param}"] = v
        else:
            unknown.append(key)
    return out, unknown


def load_pretrained():
    """(model, processor) exactly as the notebook builds base_model in section 3."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

    src = load_file(hf_hub_download(BASE_MODEL, "model.safetensors"))
    is_timm = any(re.search(r"(^|\.)blocks\.\d+\.attn\.qkv", k) for k in src)
    state, unknown = timm_to_hf(src) if is_timm else (src, [])
    config = ViTConfig.from_pretrained(BASE_MODEL)
    config.num_labels = 2
    if is_timm:
        config.layer_norm_eps = 1e-6             # timm's ViT uses 1e-6
    model = ViTForImageClassification(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"{len(missing)} weights NOT loaded, e.g. {missing[:5]}")
    print(f"loaded {len(model.state_dict())} tensors ({'timm → HF' if is_timm else 'HF names'}); "
          f"ignored: {unexpected + unknown or 'nothing'}")
    # the notebook's processor: google/vit-base-patch16-224-in21k = 224×224, mean/std 0.5, bilinear
    processor = ViTImageProcessor(do_resize=True, size={"height": 224, "width": 224}, resample=2,
                                  do_rescale=True, rescale_factor=1 / 255, do_normalize=True,
                                  image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    return model.eval(), processor


def _frames(split: str) -> list[tuple[Path, int]]:
    out = []
    for r in csv.DictReader(open(SPLITS_CSV)):
        if r["split"] == split:
            cls, name = r["path"].split(" /")
            out.append((RLVS_ROOT / cls / name, int(r["label"])))
    return out


def probs(model, processor, frames, device, batch=32):
    import torch
    from PIL import Image
    model = model.to(device).eval()
    out = []
    for i in range(0, len(frames), batch):
        imgs = []
        for p, _ in frames[i:i + batch]:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
        x = processor(images=imgs, return_tensors="pt").to(device)
        with torch.inference_mode():
            out.append(model(**x).logits.float().softmax(-1).cpu())
    return torch.cat(out)


def violent_index(model, processor, device) -> int:
    """Which output is 'violent': ft-v1's record, cross-checked on 200 RLVS val frames."""
    recorded = json.loads((RLVS_FT_V1_MODEL_DIR / "training_info.json").read_text())["base_violent_output_index"]
    frames = _frames("val")[::8][:200]
    p = probs(model, processor, frames, device)
    y = [lab for _, lab in frames]
    acc1 = sum(int(a == b) for a, b in zip((p[:, 1] >= 0.5).int().tolist(), y)) / len(y)
    found = 1 if acc1 >= 0.5 else 0
    print(f"output 1 = violent gives {acc1:.3f} accuracy on {len(y)} RLVS val frames → violent index {found} "
          f"(ft-v1 records {recorded})")
    if found != recorded:
        raise RuntimeError("violent output index disagrees with ft-v1's training_info.json")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--force", action="store_true", help="re-convert even if the folder exists")
    args = ap.parse_args()
    from component1_screen_monitoring.src.detector import ViolenceDetector, _pick_device
    from component1_screen_monitoring.src.eval_screens import compute_metrics

    out = PRETRAINED_MODEL_DIR
    if args.force or not (out / "config.json").exists():
        model, processor = load_pretrained()
        vi = violent_index(model, processor, _pick_device())
        model.config.id2label = {vi: "violent", 1 - vi: "safe"}
        model.config.label2id = {"violent": vi, "safe": 1 - vi}
        out.mkdir(parents=True, exist_ok=True)
        model.cpu().save_pretrained(out)
        processor.save_pretrained(out)
        (out / "conversion_info.json").write_text(json.dumps({
            "source": BASE_MODEL, "conversion": "timm → HF, notebook section 3 loader (no training)",
            "violent_output_index": vi, "script": "src/convert_pretrained.py"}, indent=2))
        print(f"saved → {out}")
    else:
        print(f"{out} exists — verifying only")

    det = ViolenceDetector(out)
    frames = _frames("test")
    p = probs(det.model, det.processor, frames, det.device)[:, det.violent_idx].tolist()
    m = compute_metrics([lab for _, lab in frames], [int(x >= 0.5) for x in p])
    print(f"RLVS test (n={m['n']}): accuracy {m['accuracy']:.4f}  precision {m['precision']:.4f}  "
          f"recall {m['recall']:.4f}  F1 {m['f1']:.4f}  confusion {m['confusion']}")
    ok = abs(m["accuracy"] - 0.996) < 0.0025
    print("✓ matches the notebook baseline (~0.996)" if ok else "✗ does NOT match the notebook baseline (~0.996)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

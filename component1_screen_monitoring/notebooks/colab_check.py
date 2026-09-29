# ============================================================
# notebooks/colab_check.py
# One-cell Colab-compatible environment check for Component 1
#
# Paste the entire contents of this file into a single Colab
# code cell and run it.  It will:
#   1. Install missing packages (torch, transformers, Pillow).
#   2. Confirm GPU availability via torch.cuda.is_available().
#   3. Load the jaranohaal/vit-base-violence-detection checkpoint
#      and run a smoke-test forward pass on a blank image.
# ============================================================

# ── 0. Install dependencies (no-op if already present in Colab) ──────────────
import subprocess, sys

def _pip(*pkgs):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *pkgs]
    )

_pip(
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "transformers>=4.40.0",
    "Pillow>=10.3.0",
)

# ── 1. GPU check ──────────────────────────────────────────────────────────────
import torch

gpu_available = torch.cuda.is_available()
device        = "cuda" if gpu_available else "cpu"

if gpu_available:
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU available  : {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    print(f"   CUDA version   : {torch.version.cuda}")
else:
    print("⚠️  No GPU detected — running on CPU.")
    print("   → In Colab: Runtime ▸ Change runtime type ▸ T4 GPU")

print(f"   PyTorch version : {torch.__version__}")
print(f"   Device selected : {device}\n")

# ── 2. Load ViT violence-detection checkpoint ─────────────────────────────────
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import numpy as np

CHECKPOINT = "jaranohaal/vit-base-violence-detection"

print(f"Loading checkpoint  : {CHECKPOINT}")
print("  (downloading model weights on first run — ~330 MB) …\n")

try:
    extractor = AutoFeatureExtractor.from_pretrained(CHECKPOINT)
    model     = AutoModelForImageClassification.from_pretrained(CHECKPOINT)
    model.to(device).eval()

    # ── smoke-test: forward pass on a blank 224×224 RGB image ────────────────
    dummy_img = Image.fromarray(
        np.zeros((224, 224, 3), dtype=np.uint8), mode="RGB"
    )
    inputs  = extractor(images=dummy_img, return_tensors="pt")
    inputs  = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits                          # shape: (1, num_labels)
    probs  = torch.softmax(logits, dim=-1)[0]
    labels = model.config.id2label

    print("✅ Model loaded successfully!")
    print(f"   Architecture    : {model.config.model_type}")
    print(f"   Num labels      : {len(labels)}")
    print(f"   Label map       : {labels}")
    print(f"\n   Smoke-test (blank image) probabilities:")
    for idx, prob in enumerate(probs.tolist()):
        print(f"     {labels[idx]:>20s}  {prob:.4f}")

except Exception as exc:
    print(f"❌ Checkpoint load FAILED: {exc}")
    print("\nDebug tips:")
    print("  • Check internet access in Colab (Runtime ▸ Reconnect)")
    print("  • Verify checkpoint name: https://huggingface.co/jaranohaal/vit-base-violence-detection")
    print("  • Ensure transformers>=4.40.0 is installed")
    raise

# ── 3. Summary ────────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("Environment check complete.")
print(f"  GPU     : {'YES — ' + gpu_name if gpu_available else 'NO (CPU only)'}")
print(f"  Model   : {CHECKPOINT}  ✅")
print("─" * 55)

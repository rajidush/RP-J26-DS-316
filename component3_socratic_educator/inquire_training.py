"""
inquire_training.py

The testable half of notebooks/train_inquire_lora_colab.ipynb: the fixed LoRA
and trainer config, the dataset manifest check and the run record (ticket 04),
and the comparison generation settings and raw-output records (ticket 06). Standard library only, so it runs on Colab alone (upload it next to the
`build` output) and in the local tests without torch or TRL.

    verify_dataset(folder)   stop early unless manifest.json exists and every
                             split file matches the hash `build` recorded
    run_record(...)          the evidence JSON for one training run
    generate_arm(...)        one arm's raw outputs on the test prompts, in the
                             format `inquire_finetune.py sheet` reads (ticket 06)

The config is fixed, not searched: r=16, alpha=32, attention + MLP projections,
LR 1e-4, 3 epochs, evaluate and save every epoch, keep the lowest validation
loss. Settings the decision left open are fixed here once and recorded.
"""
import hashlib
import json
import re
import time
from pathlib import Path

BASE_MODEL = "google/gemma-3-1b-it"
SEED = 20261008
NOT_RECORDED = "not recorded"
SPLIT_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl")

# Installed by the notebook's first cell (a test keeps the two identical). torch is
# Colab's preinstalled build; its version goes into the run record.
PINNED_PACKAGES = (
    "transformers==4.56.2",
    "trl==0.23.1",
    "peft==0.17.1",
    "accelerate==1.10.1",
    "datasets==4.0.0",
)

PRECISION = "fp32"
PRECISION_REASON = (
    "T4 has no bf16, and Gemma 3 activations overflow in fp16; the 1B model plus LoRA fits a T4 in fp32 "
    "and the dataset is small, so fp32 costs only runtime.")

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINER_CONFIG = {
    "learning_rate": 1e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.0,
    "weight_decay": 0.0,
    "optim": "adamw_torch",
    "max_length": 1024,
    "packing": False,
    "completion_only_loss": True,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "fp16": False,
    "bf16": False,
    "seed": SEED,
    "data_seed": SEED,
    "report_to": "none",
}


class DatasetError(Exception):
    """The dataset folder is not the `build` output its manifest describes."""


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_sha256(folder):
    return _sha256(Path(folder) / "manifest.json")


def verify_dataset(folder):
    """The manifest, if every split file exists and matches its recorded hash; DatasetError otherwise."""
    folder = Path(folder)
    path = folder / "manifest.json"
    if not path.is_file():
        raise DatasetError(f"{path} is missing: upload the whole `build` output folder")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    recorded = manifest.get("outputs", {})
    for name in SPLIT_FILES:
        if name not in recorded:
            raise DatasetError(f"manifest.json records no hash for {name}: rebuild with `inquire_finetune.py build`")
        if not (folder / name).is_file():
            raise DatasetError(f"{folder / name} is missing")
        if _sha256(folder / name) != recorded[name]:
            raise DatasetError(f"{name} does not match the hash in manifest.json: stale or edited file, rebuild it")
    return manifest


def _epoch(value):
    return int(round(value))


def epoch_losses(log_history):
    """Per-epoch train and validation loss from Trainer.state.log_history."""
    epochs = {}
    for entry in log_history:
        if "epoch" not in entry or "train_runtime" in entry:
            continue
        row = epochs.setdefault(_epoch(entry["epoch"]), {"epoch": _epoch(entry["epoch"]), "step": entry.get("step"),
                                                       "train_loss": NOT_RECORDED, "eval_loss": NOT_RECORDED})
        if "loss" in entry:
            row["train_loss"] = entry["loss"]
        if "eval_loss" in entry:
            row["eval_loss"] = entry["eval_loss"]
    return [epochs[e] for e in sorted(epochs)]


def _kept_checkpoint(best_checkpoint, best_metric, log_history):
    if not best_checkpoint:
        return NOT_RECORDED
    name = Path(best_checkpoint).name
    step = re.search(r"checkpoint-(\d+)$", name)
    epoch = next((row["epoch"] for row in epoch_losses(log_history) if step and row["step"] == int(step.group(1))),
                 NOT_RECORDED)
    return {"checkpoint": name, "epoch": epoch, "eval_loss": NOT_RECORDED if best_metric is None else best_metric}


def _or_not_recorded(value):
    return NOT_RECORDED if value is None else value


def run_record(*, manifest_sha256, dataset_folder, versions, gpu, runtime_seconds, log_history,
               best_checkpoint, best_metric, adapter_location, extra=None):
    """The evidence JSON for one run. Anything unavailable is recorded as "not recorded"."""
    final = next((e["train_loss"] for e in log_history if "train_loss" in e), None)
    record = {
        "model_id": BASE_MODEL,
        "dataset": {"folder": dataset_folder, "manifest_sha256": manifest_sha256},
        "library_versions": {k: _or_not_recorded(v) for k, v in versions.items()},
        "gpu": _or_not_recorded(gpu),
        "precision": PRECISION,
        "precision_reason": PRECISION_REASON,
        "seed": SEED,
        "lora_config": LORA_CONFIG,
        "trainer_config": TRAINER_CONFIG,
        "runtime_seconds": _or_not_recorded(runtime_seconds),
        "epochs": epoch_losses(log_history),
        "kept_checkpoint": _kept_checkpoint(best_checkpoint, best_metric, log_history),
        "final_train_loss": _or_not_recorded(final),
        "adapter_location": _or_not_recorded(adapter_location),
    }
    clashes = sorted(set(extra or {}) & set(record))
    if clashes:
        raise ValueError(f"extra fields would overwrite recorded ones: {', '.join(clashes)}")
    record.update({k: _or_not_recorded(v) for k, v in (extra or {}).items()})
    return record


def write_run_record(path, record):
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- comparison generation (ticket 06)
# Greedy (temperature 0), one output per prompt. The INQUIRE format allows at most 30 words
# (about 45 tokens); 128 leaves room for an over-long reply to show up as a format failure
# rather than a cut-off, and `hit_max_new_tokens` marks any reply that was cut off anyway.
GENERATION_CONFIG = {
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "max_new_tokens": 128,
}
ARMS = ("control", "adapted")


def generation_settings(*, manifest_sha256, model_revision, versions, gpu):
    """Everything both arms share; `sheet` refuses outputs whose settings differ."""
    return {
        "generation_config": GENERATION_CONFIG,
        "temperature": 0,
        "precision": PRECISION,
        "attn_implementation": "eager",
        "chat_template": "tokenizer.apply_chat_template(prompt, add_generation_prompt=True)",
        "postprocessing": "decode new tokens with skip_special_tokens; strip surrounding whitespace only, no repair",
        "prompt_source": {"dataset_manifest_sha256": manifest_sha256, "split": "test.jsonl"},
        "model_revision": _or_not_recorded(model_revision),
        "library_versions": {k: _or_not_recorded(v) for k, v in versions.items()},
        "gpu": _or_not_recorded(gpu),
        "seed": SEED,
    }


def generate_arm(test_rows, arm, generate, *, settings, adapter=None, clock=time.perf_counter):
    """One raw-output record per test row. `generate(prompt_messages)` returns (text, new_tokens);
    latency is the wall-clock time of that call."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, not {arm!r}")
    if (arm == "adapted") != bool(adapter):
        raise ValueError("the adapted arm needs an adapter; the control arm must not have one")
    records = []
    for row in test_rows:
        start = clock()
        text, new_tokens = generate(row["prompt"])
        latency = clock() - start
        records.append({
            "example_id": row["example_id"], "arm": arm, "model_id": BASE_MODEL,
            "adapter": adapter, "generation_settings": settings,
            "response": text.strip(), "latency_seconds": round(latency, 3), "new_tokens": new_tokens,
            "hit_max_new_tokens": new_tokens >= GENERATION_CONFIG["max_new_tokens"]})
    return records


def write_raw_outputs(path, records):
    """JSONL, one record per line; never overwrites earlier recorded outputs."""
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)

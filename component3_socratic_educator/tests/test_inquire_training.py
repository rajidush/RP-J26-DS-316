"""
Tests for inquire_training.py, the Colab-standalone half of the training
notebook: the fixed LoRA/trainer config, the manifest hash check, and the run
record. No model, torch or TRL is loaded; the notebook itself is only checked
structurally (ticket 04: no tests for live training).
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inquire_finetune as ft  # noqa: E402
import inquire_training as tr  # noqa: E402

DATASET_DIR = PROJECT_ROOT / "evidence" / "inquire_dataset_2026-10-10"
NOTEBOOK = PROJECT_ROOT / "notebooks" / "train_inquire_lora_colab.ipynb"


@pytest.fixture
def dataset(tmp_path):
    d = tmp_path / "inquire_dataset"
    shutil.copytree(DATASET_DIR, d)
    return d


# ---------------------------------------------------------------- manifest check
def test_verify_dataset_accepts_the_committed_build(dataset):
    manifest = tr.verify_dataset(dataset)
    assert manifest["dataset"] == "inquire-finetune"


def test_manifest_sha256_hashes_the_manifest_file(dataset):
    assert tr.manifest_sha256(dataset) == hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()


def test_verify_dataset_refuses_a_missing_manifest(dataset):
    (dataset / "manifest.json").unlink()
    with pytest.raises(tr.DatasetError, match="manifest.json"):
        tr.verify_dataset(dataset)


@pytest.mark.parametrize("name", tr.SPLIT_FILES)
def test_verify_dataset_refuses_a_missing_split_file(dataset, name):
    (dataset / name).unlink()
    with pytest.raises(tr.DatasetError, match=name):
        tr.verify_dataset(dataset)


@pytest.mark.parametrize("name", tr.SPLIT_FILES)
def test_verify_dataset_refuses_a_changed_split_file(dataset, name):
    with open(dataset / name, "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(tr.DatasetError, match=f"{name}.*hash"):
        tr.verify_dataset(dataset)


def test_verify_dataset_refuses_a_manifest_without_output_hashes(dataset):
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["outputs"]["validation.jsonl"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(tr.DatasetError, match="validation.jsonl"):
        tr.verify_dataset(dataset)


# ---------------------------------------------------------------- fixed config
def test_lora_config_is_the_fixed_config():
    assert tr.BASE_MODEL == "google/gemma-3-1b-it"
    assert tr.LORA_CONFIG["r"] == 16
    assert tr.LORA_CONFIG["lora_alpha"] == 32
    assert set(tr.LORA_CONFIG["target_modules"]) == {
        "q_proj", "k_proj", "v_proj", "o_proj",      # attention
        "gate_proj", "up_proj", "down_proj"}         # MLP
    assert tr.LORA_CONFIG["task_type"] == "CAUSAL_LM"


def test_trainer_config_is_the_fixed_config():
    cfg = tr.TRAINER_CONFIG
    assert cfg["learning_rate"] == 1e-4
    assert cfg["num_train_epochs"] == 3
    assert cfg["eval_strategy"] == cfg["save_strategy"] == cfg["logging_strategy"] == "epoch"
    assert cfg["load_best_model_at_end"] is True
    assert cfg["metric_for_best_model"] == "eval_loss" and cfg["greater_is_better"] is False
    assert cfg["completion_only_loss"] is True
    assert cfg["seed"] == cfg["data_seed"] == tr.SEED
    assert cfg["fp16"] is False and cfg["bf16"] is False  # fp32 on the T4 (see PRECISION_REASON)
    assert cfg["report_to"] == "none"


def test_config_values_are_single_values_not_a_search():
    for cfg in (tr.LORA_CONFIG, tr.TRAINER_CONFIG):
        for key, value in cfg.items():
            if key != "target_modules":
                assert not isinstance(value, (list, tuple, set, range)), key


# ---------------------------------------------------------------- losses and run record
LOG_HISTORY = [
    {"loss": 2.1, "grad_norm": 1.0, "learning_rate": 9e-5, "epoch": 1.0, "step": 26},
    {"eval_loss": 1.6, "eval_runtime": 3.0, "epoch": 1.0, "step": 26},
    {"loss": 1.2, "epoch": 2.0, "step": 52},
    {"eval_loss": 1.4, "epoch": 2.0, "step": 52},
    {"loss": 0.8, "epoch": 3.0, "step": 78},
    {"eval_loss": 1.5, "epoch": 3.0, "step": 78},
    {"train_runtime": 240.5, "train_loss": 1.37, "epoch": 3.0, "step": 78},
]


def test_epoch_losses_pairs_train_and_validation_loss_per_epoch():
    assert tr.epoch_losses(LOG_HISTORY) == [
        {"epoch": 1, "step": 26, "train_loss": 2.1, "eval_loss": 1.6},
        {"epoch": 2, "step": 52, "train_loss": 1.2, "eval_loss": 1.4},
        {"epoch": 3, "step": 78, "train_loss": 0.8, "eval_loss": 1.5},
    ]


def test_epoch_losses_marks_a_missing_value_not_recorded():
    rows = tr.epoch_losses([{"eval_loss": 1.6, "epoch": 1.0, "step": 26}])
    assert rows == [{"epoch": 1, "step": 26, "train_loss": tr.NOT_RECORDED, "eval_loss": 1.6}]


def test_run_record_fills_gaps_with_not_recorded_and_carries_the_config():
    record = tr.run_record(
        manifest_sha256="abc", dataset_folder="inquire_dataset_2026-10-10",
        versions={"torch": "2.8.0", "trl": None}, gpu=None, runtime_seconds=240.5,
        log_history=LOG_HISTORY, best_checkpoint="/drive/run/checkpoint-52", best_metric=1.4,
        adapter_location=None, extra={"train_rows": 104, "model_revision": None})
    assert record["model_id"] == "google/gemma-3-1b-it"
    assert record["dataset"] == {"folder": "inquire_dataset_2026-10-10", "manifest_sha256": "abc"}
    assert record["library_versions"] == {"torch": "2.8.0", "trl": tr.NOT_RECORDED}
    assert record["gpu"] == tr.NOT_RECORDED
    assert record["adapter_location"] == tr.NOT_RECORDED
    assert record["precision"] == "fp32" and record["precision_reason"]
    assert record["seed"] == tr.SEED
    assert record["lora_config"] == tr.LORA_CONFIG
    assert record["trainer_config"] == tr.TRAINER_CONFIG
    assert record["runtime_seconds"] == 240.5
    assert record["epochs"] == tr.epoch_losses(LOG_HISTORY)
    assert record["kept_checkpoint"] == {"checkpoint": "checkpoint-52", "epoch": 2, "eval_loss": 1.4}
    assert record["final_train_loss"] == 1.37
    assert record["train_rows"] == 104 and record["model_revision"] == tr.NOT_RECORDED
    json.dumps(record)  # plain JSON, committable as evidence


def test_run_record_when_no_best_checkpoint_was_recorded():
    record = tr.run_record(manifest_sha256="abc", dataset_folder="d", versions={}, gpu="Tesla T4",
                           runtime_seconds=None, log_history=[], best_checkpoint=None, best_metric=None,
                           adapter_location="/drive/adapter")
    assert record["kept_checkpoint"] == tr.NOT_RECORDED
    assert record["runtime_seconds"] == tr.NOT_RECORDED
    assert record["final_train_loss"] == tr.NOT_RECORDED


def test_run_record_refuses_extra_fields_that_would_overwrite_recorded_ones():
    with pytest.raises(ValueError, match="seed"):
        tr.run_record(manifest_sha256="abc", dataset_folder="d", versions={}, gpu=None, runtime_seconds=None,
                      log_history=[], best_checkpoint=None, best_metric=None, adapter_location=None,
                      extra={"seed": 1})


# ---------------------------------------------------------------- comparison generation (ticket 06)

TEST_ROWS = [json.loads(line) for line in (DATASET_DIR / "test.jsonl").read_text(encoding="utf-8").splitlines()]
ADAPTER = "/content/drive/MyDrive/c3_inquire_training/2026-10-10_1249/adapter"


def _settings():
    return tr.generation_settings(manifest_sha256="abc", model_revision="dcc83ea",
                                  versions={"torch": "2.11.0", "peft": None}, gpu="Tesla T4")


def _fake_generate(text, new_tokens=12):
    seen = []

    def generate(messages):
        seen.append(messages)
        return text, new_tokens
    generate.seen = seen
    return generate


def _ticking_clock(step=0.5):
    now = [0.0]

    def clock():
        now[0] += step
        return now[0]
    return clock


def test_generation_config_is_greedy_single_output_with_a_fixed_token_budget():
    cfg = tr.GENERATION_CONFIG
    assert cfg["do_sample"] is False                 # temperature 0: greedy decoding
    assert cfg["num_beams"] == 1 and cfg["num_return_sequences"] == 1
    assert isinstance(cfg["max_new_tokens"], int) and cfg["max_new_tokens"] > 0


def test_generation_settings_record_everything_both_arms_share():
    s = _settings()
    assert s["generation_config"] == tr.GENERATION_CONFIG
    assert s["temperature"] == 0
    assert s["precision"] == tr.PRECISION
    assert s["prompt_source"] == {"dataset_manifest_sha256": "abc", "split": "test.jsonl"}
    assert s["model_revision"] == "dcc83ea"
    assert s["library_versions"] == {"torch": "2.11.0", "peft": tr.NOT_RECORDED}
    assert s["gpu"] == "Tesla T4"
    assert s["seed"] == tr.SEED
    assert "apply_chat_template" in s["chat_template"]
    assert "whitespace" in s["postprocessing"]
    json.dumps(s)


def test_generate_arm_answers_every_test_prompt_once_with_the_prompt_only():
    generate = _fake_generate("How did that feel?")
    records = tr.generate_arm(TEST_ROWS, "control", generate, settings=_settings(), clock=_ticking_clock())
    assert [r["example_id"] for r in records] == [r["example_id"] for r in TEST_ROWS]
    assert len(records) == 50
    assert generate.seen == [r["prompt"] for r in TEST_ROWS]   # nothing but the prompt reaches the model


def test_generate_arm_writes_the_raw_output_record_format():
    records = tr.generate_arm(TEST_ROWS[:1], "adapted", _fake_generate("Okay?", 7), settings=_settings(),
                              adapter=ADAPTER, clock=_ticking_clock(0.25))
    assert records == [{
        "example_id": TEST_ROWS[0]["example_id"], "arm": "adapted", "model_id": tr.BASE_MODEL,
        "adapter": ADAPTER, "generation_settings": _settings(), "response": "Okay?",
        "latency_seconds": 0.25, "new_tokens": 7, "hit_max_new_tokens": False}]


def test_generate_arm_strips_whitespace_only_and_never_repairs():
    raw = "  \n Sure. That sounds hard. Want to talk about it?? \"quoted\" \n\n"
    [record] = tr.generate_arm(TEST_ROWS[:1], "control", _fake_generate(raw), settings=_settings())
    assert record["response"] == raw.strip()


def test_generate_arm_flags_a_response_cut_off_at_the_token_budget():
    budget = tr.GENERATION_CONFIG["max_new_tokens"]
    [record] = tr.generate_arm(TEST_ROWS[:1], "control", _fake_generate("and then", budget), settings=_settings())
    assert record["hit_max_new_tokens"] is True


@pytest.mark.parametrize("arm, adapter, problem", [
    ("control", ADAPTER, "control"),
    ("adapted", None, "adapter"),
    ("baseline", None, "arm"),
])
def test_generate_arm_refuses_an_arm_and_adapter_that_do_not_match(arm, adapter, problem):
    with pytest.raises(ValueError, match=problem):
        tr.generate_arm(TEST_ROWS[:1], arm, _fake_generate("Hi?"), settings=_settings(), adapter=adapter)


def test_write_raw_outputs_refuses_to_overwrite_recorded_outputs(tmp_path):
    path = tmp_path / "raw_outputs_control.jsonl"
    records = tr.generate_arm(TEST_ROWS[:2], "control", _fake_generate("Hi?"), settings=_settings())
    tr.write_raw_outputs(path, records)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        tr.write_raw_outputs(path, records)
    assert path.read_bytes() == before
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == records


def test_sheet_accepts_both_arms_written_by_generate_arm(tmp_path):
    paths = []
    for arm, adapter, text in (("control", None, "What happened next?"), ("adapted", ADAPTER, "Do you feel okay?")):
        records = tr.generate_arm(TEST_ROWS, arm, _fake_generate(text), settings=_settings(), adapter=adapter)
        paths.append(tmp_path / f"raw_outputs_{arm}.jsonl")
        tr.write_raw_outputs(paths[-1], records)
    out = tmp_path / "comparison"
    ft.main(["sheet", "--test", str(DATASET_DIR / "test.jsonl"), "--outputs", *map(str, paths), "--out", str(out)])
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    assert len(key["responses"]) == 100


# ---------------------------------------------------------------- notebook and repo hygiene
def _notebook_code():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [("".join(c["source"]), c) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_installs_exactly_the_pinned_versions():
    pip = next(src for src, _ in _notebook_code() if "pip" in src and "install" in src)
    assert sorted(tok.strip('"') for tok in pip.split() if "==" in tok) == sorted(tr.PINNED_PACKAGES)


def test_notebook_checks_the_manifest_before_training_and_writes_no_outputs():
    cells = _notebook_code()
    sources = [src for src, _ in cells]
    verify = next(i for i, s in enumerate(sources) if "verify_dataset(" in s)
    train = next(i for i, s in enumerate(sources) if ".train()" in s)
    assert verify < train
    assert all(not c.get("outputs") for _, c in cells)  # committed clean, no run output


def test_notebook_saves_the_adapter_to_drive_only():
    code = "\n".join(src for src, _ in _notebook_code())
    assert re.search(r'^RUN_DIR = f?"/content/drive/MyDrive/', code, re.M)
    assert re.search(r'^ADAPTER_DIR = f"\{RUN_DIR\}/', code, re.M)
    saves = re.findall(r"\.(?:save_model|save_pretrained)\((.*)\)", code)
    assert saves and all(arg.startswith("ADAPTER_DIR") for arg in saves), saves
    assert re.search(r'output_dir=f"\{RUN_DIR\}/', code)  # checkpoints go to Drive too


def test_notebook_generates_both_arms_from_one_base_loader_with_shared_settings():
    sources = [src for src, _ in _notebook_code()]
    code = "\n".join(sources)
    verify = next(i for i, s in enumerate(sources) if "verify_dataset(" in s)
    arms = [i for i, s in enumerate(sources) if "generate_arm(" in s]
    assert len(arms) == 2 and min(arms) > verify
    calls = re.findall(r"generate_arm\((.*)\)", code)
    assert ['"control"' in c for c in calls] == [True, False] and '"adapted"' in calls[1]
    assert all("settings=SETTINGS" in c for c in calls)             # one settings object, so identical
    assert "**tr.GENERATION_CONFIG" in code
    loader = re.search(r"def load_base\(\):\n((?:    .*\n?)+)", code).group(1)
    assert "tr.BASE_MODEL" in loader and "dtype=torch.float32" in loader
    assert re.search(r"control = load_base\(\)", code)
    assert re.search(r"PeftModel\.from_pretrained\(load_base\(\), ADAPTER_SOURCE\)", code)


def test_notebook_loads_the_adapter_from_drive_and_writes_outputs_there():
    code = "\n".join(src for src, _ in _notebook_code())
    assert re.search(r'^ADAPTER_SOURCE = "/content/drive/MyDrive/', code, re.M)
    assert re.findall(r"write_raw_outputs\((.*?),", code) == ["path"]
    assert 'path = f"{COMPARISON_DIR}/raw_outputs_{arm}.jsonl"' in code
    assert re.search(r'^COMPARISON_DIR = f"\{os\.path\.dirname\(ADAPTER_SOURCE\)\}/', code, re.M)


@pytest.mark.parametrize("path", [
    "component3_socratic_educator/evidence/run/adapter_model.safetensors",
    "component3_socratic_educator/models/adapter/adapter_config.json",
])
def test_gitignore_still_covers_adapter_weights(path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=PROJECT_ROOT.parent)
    assert result.returncode == 0, f"{path} is not gitignored"

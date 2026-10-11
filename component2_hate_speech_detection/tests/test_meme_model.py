"""Meme fusion heads and the training protocol (SO3, FR5). Synthetic data only: fast and offline."""
import argparse
import json

import numpy as np
import pytest
import torch

from component2_hate_speech_detection.meme_model import embed, train
from component2_hate_speech_detection.meme_model.heads import FUSIONS, FusionHead


@pytest.mark.parametrize("fusion", FUSIONS)
def test_every_fusion_gives_one_logit_per_meme(fusion):
    head = FusionHead(dim=8, fusion=fusion, proj_dim=4, pre_layers=2, hidden=6).eval()
    out = head(torch.randn(5, 8), torch.randn(5, 8))
    assert out.shape == (5,)


def test_align_fusion_is_the_diagonal_of_the_cross_interaction_matrix():
    # Hate-CLIPper: align = diag(p_i outer p_t). Same projections must give the same diagonal.
    torch.manual_seed(0)
    cross = FusionHead(dim=8, fusion="cross", proj_dim=4, pre_layers=1, hidden=6).eval()
    align = FusionHead(dim=8, fusion="align", proj_dim=4, pre_layers=1, hidden=6).eval()
    align.image_proj.load_state_dict(cross.image_proj.state_dict())
    align.text_proj.load_state_dict(cross.text_proj.state_dict())
    image, text = torch.randn(3, 8), torch.randn(3, 8)
    matrix = cross.fuse(image, text).view(3, 4, 4)
    assert torch.allclose(torch.diagonal(matrix, dim1=1, dim2=2), align.fuse(image, text), atol=1e-6)


def test_unimodal_heads_ignore_the_other_modality():
    head = FusionHead(dim=8, fusion="image", proj_dim=4, pre_layers=1, hidden=6).eval()
    image = torch.randn(2, 8)
    assert torch.allclose(head(image, torch.randn(2, 8)), head(image, torch.randn(2, 8)))


def test_unknown_fusion_is_rejected():
    with pytest.raises(ValueError):
        FusionHead(dim=8, fusion="sum")


def test_temperature_scaling_recovers_a_known_temperature():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 4, 20000)
    y = (rng.random(20000) < train.sigmoid(logits / 2.0)).astype(np.float32)
    assert train.fit_temperature(logits, y) == pytest.approx(2.0, rel=0.1)


def test_ece_is_zero_when_confidence_matches_frequency():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
    assert train.expected_calibration_error(y, np.full(8, 0.5)) == pytest.approx(0.0)
    assert train.expected_calibration_error(y, np.where(y == 1, 0.0, 1.0)) == pytest.approx(1.0)


def test_threshold_is_chosen_to_maximise_validation_accuracy():
    y = np.array([0, 0, 0, 1, 1], dtype=np.float32)
    p = np.array([0.1, 0.2, 0.3, 0.35, 0.9])
    assert train.best_threshold(y, p) == pytest.approx(0.35)


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300).astype(np.float32)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    low, high = train.bootstrap_auroc(y, p, rounds=200)
    assert low <= train.roc_auc_score(y, p) <= high


def _write_manifests(folder, rows_by_split):
    for name, rows in rows_by_split.items():
        with open(folder / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def test_all_memes_lists_each_id_once(tmp_path):
    row = {"id": 1, "img": "img/00001.png", "text": "a", "label": 0}
    _write_manifests(tmp_path, {"train": [row], "val": [row], "test": [{**row, "id": 2}],
                                "unlabelled_test_seen": [{**row, "id": 3, "label": None}]})
    assert [r["id"] for r in embed.all_memes(tmp_path)] == [1, 2, 3]


def test_split_arrays_keep_ids_labels_and_vectors_aligned():
    embeddings = {7: (np.full(4, 7.0), np.full(4, -7.0)), 3: (np.full(4, 3.0), np.full(4, -3.0))}
    rows = [{"id": 3, "label": 1, "label_conflict": True}, {"id": 7, "label": 0}]
    split = train.split_arrays(rows, embeddings)
    assert split["ids"].tolist() == [3, 7] and split["y"].tolist() == [1.0, 0.0]
    assert split["image"][0, 0] == 3.0 and split["text"][1, 0] == -7.0
    assert split["conflict"].tolist() == [True, False]


def _synthetic_split(n, rng, interaction):
    image, text = rng.normal(size=(n, 16)), rng.normal(size=(n, 16))
    # Label depends on the image-text *interaction*, as in a hateful meme.
    y = ((image[:, 0] * text[:, 0]) > 0) if interaction else (image[:, 0] > 0)
    return {"ids": np.arange(n), "image": image.astype(np.float32), "text": text.astype(np.float32),
            "y": y.astype(np.float32), "conflict": np.zeros(n, dtype=bool)}


def _args(fusion):
    return argparse.Namespace(fusion=fusion, proj_dim=32, pre_layers=1, hidden=32, dropout=0.0, lr=3e-3,
                              weight_decay=0.0, grad_clip=1.0, batch_size=64, epochs=25, patience=25,
                              class_weight=False)


def test_align_head_learns_an_image_text_interaction_that_image_alone_cannot():
    rng = np.random.default_rng(316)
    splits = {name: _synthetic_split(n, rng, interaction=True) for name, n in
              (("train", 2000), ("val", 400), ("test", 400))}
    align = train.run_seed(splits, _args("align"), 316)["test"]["auroc"]
    image_only = train.run_seed(splits, _args("image"), 316)["test"]["auroc"]
    assert align > 0.85 and image_only < 0.65


def test_rgcl_loss_is_finite_and_trains_the_head():
    torch.manual_seed(0)
    head = FusionHead(dim=8, fusion="align", proj_dim=8, pre_layers=1, hidden=8, dropout=0.0)
    image, text = torch.randn(40, 8), torch.randn(40, 8)
    y = torch.tensor([0.0, 1.0] * 20)
    bank = train.embed_split(head, image, text)
    head.train()
    loss = train.rgcl_loss(head, image, text, y, torch.arange(16), bank)
    loss.backward()
    assert torch.isfinite(loss) and head.out.weight.grad is None  # contrastive loss acts on features only
    assert head.pre_output[0].weight.grad.abs().sum() > 0


def test_rgcl_with_knn_retrieval_learns_the_interaction():
    rng = np.random.default_rng(316)
    splits = {name: _synthetic_split(n, rng, interaction=True) for name, n in
              (("train", 1500), ("val", 300), ("test", 300))}
    args = _args("align")
    args.rgcl, args.knn = 1.0, 10
    result = train.run_seed(splits, args, 316)
    assert result["test"]["auroc"] > 0.85 and result["test_knn"]["auroc"] > 0.85


def test_device_auto_falls_back_to_cpu_and_tensors_follow_it(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert train.resolve_device("auto").type == "cpu"
    assert train.resolve_device("cpu").type == "cpu"
    split = {"image": np.zeros((2, 4), np.float32), "text": np.ones((2, 4), np.float32),
             "y": np.array([0, 1], np.float32)}
    image, text, y = train.tensors(split, "cpu")
    assert image.device.type == "cpu" and y.tolist() == [0.0, 1.0]

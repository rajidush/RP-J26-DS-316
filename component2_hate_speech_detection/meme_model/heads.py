"""Fusion heads over frozen CLIP features, after Hate-CLIPper (Kumar & Nandakumar, 2022).

Each modality gets a trainable projection p_i = W_i x_i, p_t = W_t x_t, then:

    concat   [p_i ; p_t]           late fusion; a baseline that cannot model interaction
    align    p_i * p_t             Hadamard product = the diagonal of the feature
                                   interaction matrix. Best HMC test_seen AUROC in the
                                   paper (85.8, 3 pre-output layers) with ~5M parameters
    cross    vec(p_i outer p_t)    the full interaction matrix; ~1.1B parameters at the
                                   paper's n = 1024, so keep proj_dim small here
    image    p_i                   unimodal ablations (D6: where does the harm sit?)
    text     p_t

followed by `pre_layers` x (Linear, ReLU, Dropout) and one output logit.
`features()` returns the last hidden layer before its ReLU, the space RGCL's
contrastive loss and k-NN retrieval work in (Mei et al., 2024).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

FUSIONS = ("concat", "align", "cross", "image", "text")


class FusionHead(nn.Module):
    def __init__(self, dim: int, fusion: str = "align", proj_dim: int = 1024, pre_layers: int = 3,
                 hidden: int = 1024, dropout: float = 0.2) -> None:
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        self.fusion = fusion
        self.config = {"dim": dim, "fusion": fusion, "proj_dim": proj_dim, "pre_layers": pre_layers,
                       "hidden": hidden, "dropout": dropout}
        self.image_proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.Dropout(dropout))
        self.text_proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.Dropout(dropout))
        width = {"concat": 2 * proj_dim, "align": proj_dim, "cross": proj_dim * proj_dim,
                 "image": proj_dim, "text": proj_dim}[fusion]
        layers: list[nn.Module] = []
        for i in range(pre_layers):
            if i:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
            layers.append(nn.Linear(width, hidden))
            width = hidden
        self.pre_output = nn.Sequential(*layers)
        # The embedding is taken *before* the last ReLU: after it every value is
        # >= 0, and a contrastive loss there drives units dead (measured: RGCL
        # collapsed to AUROC 0.48 on the synthetic interaction test).
        self.activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout)) if pre_layers else nn.Identity()
        self.out = nn.Linear(width, 1)

    def fuse(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        # CLIP features are compared by angle, so both sides are unit-normalised
        # before and after projection; this keeps the products on one scale.
        p_i = F.normalize(self.image_proj(F.normalize(image, dim=-1)), dim=-1)
        p_t = F.normalize(self.text_proj(F.normalize(text, dim=-1)), dim=-1)
        if self.fusion == "concat":
            return torch.cat([p_i, p_t], dim=-1)
        if self.fusion == "align":
            return p_i * p_t
        if self.fusion == "cross":
            return torch.einsum("bi,bj->bij", p_i, p_t).flatten(1)
        return p_i if self.fusion == "image" else p_t

    def features(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.pre_output(self.fuse(image, text))

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.out(self.activation(self.features(image, text))).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

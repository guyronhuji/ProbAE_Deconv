from __future__ import annotations

import torch
from torch import nn


def _build_activation(name: str) -> nn.Module:
    lowered = name.lower()
    if lowered == "relu":
        return nn.ReLU()
    if lowered == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class ProbabilisticArchetypalAutoencoder(nn.Module):
    def __init__(
        self,
        n_markers: int,
        n_archetypes: int,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_markers = n_markers
        self.n_archetypes = n_archetypes

        layers: list[nn.Module] = []
        dims = [n_markers, *encoder_hidden_dims, n_archetypes]
        act = _build_activation(activation)
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(in_dim, out_dim), act])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.encoder = nn.Sequential(*layers)

        self.archetype_means = nn.Parameter(torch.randn(n_archetypes, n_markers) * 0.05)
        self.archetype_logvars = nn.Parameter(torch.zeros(n_archetypes, n_markers))

    def encode_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.encode_logits(x), dim=-1)

    def decode_params(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = weights @ self.archetype_means
        logvar = weights @ self.archetype_logvars
        return mean, logvar

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = self.encode(x)
        mean, logvar = self.decode_params(weights)
        return mean, logvar, weights

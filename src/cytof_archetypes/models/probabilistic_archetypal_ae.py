from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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
        decoder_family: str = "gaussian",
        dispersion: str = "gene",
    ) -> None:
        super().__init__()
        self.n_markers = n_markers
        self.n_archetypes = n_archetypes
        self.decoder_family = str(decoder_family).lower()
        if self.decoder_family not in {"gaussian", "nb"}:
            raise ValueError(f"Unsupported decoder_family: {decoder_family}")
        self.dispersion = str(dispersion).lower()
        if self.dispersion not in {"gene"}:
            raise ValueError(f"Unsupported dispersion mode: {dispersion}")

        layers: list[nn.Module] = []
        dims = [n_markers, *encoder_hidden_dims, n_archetypes]
        act = _build_activation(activation)
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(in_dim, out_dim), act])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.encoder = nn.Sequential(*layers)

        if self.decoder_family == "gaussian":
            self.archetype_means = nn.Parameter(torch.randn(n_archetypes, n_markers) * 0.05)
            self.archetype_logvars = nn.Parameter(torch.zeros(n_archetypes, n_markers))
            self.archetype_logits = None
            self.log_theta = None
        else:
            self.archetype_logits = nn.Parameter(torch.randn(n_archetypes, n_markers))
            self.log_theta = nn.Parameter(torch.zeros(n_markers))
            self.archetype_means = None
            self.archetype_logvars = None

    def encode_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.encode_logits(x), dim=-1)

    def decode_params(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.decoder_family != "gaussian":
            raise RuntimeError("decode_params is only available for gaussian decoder.")
        mean = weights @ self.archetype_means
        logvar = weights @ self.archetype_logvars
        return mean, logvar

    def _decode_nb(
        self,
        weights: torch.Tensor,
        library_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rho = torch.softmax(self.archetype_logits, dim=-1)
        rho_i = weights @ rho
        lib = library_size.view(-1, 1).to(rho_i.dtype)
        mu = torch.clamp(lib * rho_i, min=1e-8)
        theta = F.softplus(self.log_theta).view(1, -1).expand_as(mu)
        return mu, theta

    def diversity_basis(self) -> torch.Tensor:
        if self.decoder_family == "gaussian":
            return self.archetype_means
        return self.archetype_logits

    def forward(
        self,
        x: torch.Tensor,
        library_size: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        weights = self.encode(x)
        if self.decoder_family == "gaussian":
            recon, logvar = self.decode_params(weights)
            return {
                "weights": weights,
                "recon": recon,
                "logvar": logvar,
                "mu": None,
                "theta": None,
            }

        if library_size is None:
            library_size = torch.sum(torch.clamp(x, min=0.0), dim=1)
        mu, theta = self._decode_nb(weights, library_size)
        return {
            "weights": weights,
            "recon": None,
            "logvar": None,
            "mu": mu,
            "theta": theta,
        }

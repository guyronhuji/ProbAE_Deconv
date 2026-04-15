from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _build_activation(name: str) -> nn.Module:
    lowered = str(name).lower()
    if lowered == "relu":
        return nn.ReLU()
    if lowered == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class MultimodalProbabilisticArchetypalAutoencoder(nn.Module):
    def __init__(
        self,
        modality_specs: dict[str, dict[str, Any]],
        n_archetypes: int,
    ) -> None:
        super().__init__()
        if n_archetypes <= 1:
            raise ValueError("n_archetypes must be > 1")
        if len(modality_specs) < 2:
            raise ValueError("At least two modalities are required for multimodal deconvolution.")

        self.n_archetypes = int(n_archetypes)
        self.modalities = list(modality_specs.keys())
        self.n_markers: dict[str, int] = {}
        self.decoder_family: dict[str, str] = {}

        self.encoders = nn.ModuleDict()
        self.archetype_means = nn.ParameterDict()
        self.archetype_logvars = nn.ParameterDict()
        self.archetype_logits = nn.ParameterDict()
        self.log_theta = nn.ParameterDict()
        self.log_concentration = nn.ParameterDict()

        for modality, spec in modality_specs.items():
            n_markers = int(spec["n_markers"])
            hidden_dims = tuple(int(v) for v in spec.get("encoder_hidden_dims", [128, 64]))
            activation = str(spec.get("activation", "relu"))
            dropout = float(spec.get("dropout", 0.1))
            decoder_family = str(spec.get("decoder_family", "gaussian")).lower()
            dispersion = str(spec.get("dispersion", "gene")).lower()

            if decoder_family not in {"gaussian", "nb", "beta_binomial"}:
                raise ValueError(f"Unsupported decoder_family for modality '{modality}': {decoder_family}")
            if decoder_family in {"nb", "beta_binomial"} and dispersion != "gene":
                raise ValueError("Only dispersion='gene' is currently supported.")

            self.n_markers[modality] = n_markers
            self.decoder_family[modality] = decoder_family
            self.encoders[modality] = self._make_encoder(
                n_markers=n_markers,
                n_archetypes=self.n_archetypes,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
            )

            if decoder_family == "gaussian":
                self.archetype_means[modality] = nn.Parameter(torch.randn(self.n_archetypes, n_markers) * 0.05)
                self.archetype_logvars[modality] = nn.Parameter(torch.zeros(self.n_archetypes, n_markers))
            elif decoder_family == "nb":
                self.archetype_logits[modality] = nn.Parameter(torch.randn(self.n_archetypes, n_markers))
                self.log_theta[modality] = nn.Parameter(torch.zeros(n_markers))
            else:
                self.archetype_logits[modality] = nn.Parameter(torch.randn(self.n_archetypes, n_markers))
                self.log_concentration[modality] = nn.Parameter(torch.zeros(n_markers))

    @staticmethod
    def _make_encoder(
        n_markers: int,
        n_archetypes: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        dropout: float,
    ) -> nn.Module:
        dims = [n_markers, *hidden_dims, n_archetypes]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(_build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*layers)

    def encode_logits(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        if modality not in self.encoders:
            raise KeyError(f"Unknown modality: {modality}")
        return self.encoders[modality](x)

    def encode(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.encode_logits(modality, x), dim=-1)

    def _decode_gaussian(self, modality: str, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        means = self.archetype_means[modality]
        logvars = self.archetype_logvars[modality]
        return weights @ means, weights @ logvars

    def _decode_nb(self, modality: str, weights: torch.Tensor, library_size: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rho = torch.softmax(self.archetype_logits[modality], dim=-1)
        rho_i = weights @ rho
        lib = library_size.view(-1, 1).to(rho_i.dtype)
        mu = torch.clamp(lib * rho_i, min=1e-8)
        theta = F.softplus(self.log_theta[modality]).view(1, -1).expand_as(mu)
        return mu, theta

    def _decode_beta_binomial(self, modality: str, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rho = torch.softmax(self.archetype_logits[modality], dim=-1)
        probs = torch.clamp(weights @ rho, min=1e-8, max=1.0 - 1e-8)
        concentration = F.softplus(self.log_concentration[modality]).view(1, -1).expand_as(probs)
        return probs, concentration

    def diversity_basis(self, modality: str) -> torch.Tensor:
        family = self.decoder_family[modality]
        if family == "gaussian":
            return self.archetype_means[modality]
        return self.archetype_logits[modality]

    def forward_modality(
        self,
        modality: str,
        x: torch.Tensor,
        library_size: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        family = self.decoder_family[modality]
        weights = self.encode(modality, x)

        if family == "gaussian":
            recon, logvar = self._decode_gaussian(modality, weights)
            return {
                "weights": weights,
                "recon": recon,
                "logvar": logvar,
                "mu": None,
                "theta": None,
                "probs": None,
                "concentration": None,
            }

        if family == "nb":
            if library_size is None:
                library_size = torch.sum(torch.clamp(x, min=0.0), dim=1)
            mu, theta = self._decode_nb(modality, weights, library_size)
            return {
                "weights": weights,
                "recon": None,
                "logvar": None,
                "mu": mu,
                "theta": theta,
                "probs": None,
                "concentration": None,
            }

        probs, concentration = self._decode_beta_binomial(modality, weights)
        return {
            "weights": weights,
            "recon": None,
            "logvar": None,
            "mu": None,
            "theta": None,
            "probs": probs,
            "concentration": concentration,
        }

    def forward(
        self,
        batch_by_modality: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, dict[str, torch.Tensor | None]]:
        out: dict[str, dict[str, torch.Tensor | None]] = {}
        for modality, payload in batch_by_modality.items():
            out[modality] = self.forward_modality(
                modality=modality,
                x=payload["x_encoder"],
                library_size=payload.get("library_size"),
            )
        return out

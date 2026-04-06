from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    inv_var = torch.exp(-logvar)
    per_marker = 0.5 * (logvar + (target - mean) ** 2 * inv_var + math.log(2.0 * math.pi))
    per_sample = per_marker.sum(dim=-1)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()


def entropy_penalty(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)
    return entropy.mean()


def diversity_penalty(archetype_means: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(archetype_means, p=2, dim=1)
    gram = normalized @ normalized.t()
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.mean((gram - identity) ** 2)


def variance_regularization(archetype_logvars: torch.Tensor) -> torch.Tensor:
    return torch.mean(archetype_logvars**2)

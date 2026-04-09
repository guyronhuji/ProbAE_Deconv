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


def nb_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    eps = 1e-8
    theta = torch.clamp(theta, min=eps)
    mu = torch.clamp(mu, min=eps)
    x = torch.clamp(x, min=0.0)
    log_likelihood = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu + eps))
        + x * (torch.log(mu) - torch.log(theta + mu + eps))
    )
    per_sample = -log_likelihood.sum(dim=1)
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
    """Penalise large archetype variances.

    Uses mean(exp(logvar)) so the penalty grows exponentially with logvar,
    preventing the optimiser from parking unused archetypes at infinite variance.
    The old mean(logvar^2) was a quadratic penalty that became too weak relative
    to the NLL gains from high variance once logvars grew large.
    """
    return torch.mean(torch.exp(archetype_logvars))

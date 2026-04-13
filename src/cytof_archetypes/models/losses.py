from __future__ import annotations

import math

import torch
import torch.nn.functional as F

EPS = 1e-8


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
    theta = torch.clamp(theta, min=EPS)
    mu = torch.clamp(mu, min=EPS)
    x = torch.clamp(x, min=0.0)
    log_likelihood = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu + EPS))
        + x * (torch.log(mu) - torch.log(theta + mu + EPS))
    )
    per_sample = -log_likelihood.sum(dim=1)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()


def beta_binomial_nll(
    m_counts: torch.Tensor,
    n_counts: torch.Tensor,
    probs: torch.Tensor,
    concentration: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    probs = torch.clamp(probs, EPS, 1.0 - EPS)
    concentration = torch.clamp(concentration, min=EPS)

    if n_counts.ndim == 1:
        n_counts = n_counts.unsqueeze(1).expand_as(probs)
    n_counts = torch.clamp(n_counts, min=0.0)
    m_counts = torch.clamp(m_counts, min=0.0)
    m_counts = torch.minimum(m_counts, n_counts)

    alpha = probs * concentration
    beta = (1.0 - probs) * concentration

    log_choose = (
        torch.lgamma(n_counts + 1.0)
        - torch.lgamma(m_counts + 1.0)
        - torch.lgamma(n_counts - m_counts + 1.0)
    )
    log_beta_ratio = (
        torch.lgamma(m_counts + alpha)
        + torch.lgamma(n_counts - m_counts + beta)
        - torch.lgamma(n_counts + alpha + beta)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
    )
    per_sample = -(log_choose + log_beta_ratio).sum(dim=1)
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

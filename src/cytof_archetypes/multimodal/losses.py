from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def alignment_distance(left: torch.Tensor, right: torch.Tensor, metric: str = "l2") -> torch.Tensor:
    metric_l = str(metric).lower()
    if metric_l == "l2":
        return torch.mean(torch.sum((left - right) ** 2, dim=1))
    if metric_l == "cosine":
        cosine_sim = F.cosine_similarity(left, right, dim=1)
        return torch.mean(1.0 - cosine_sim)
    if metric_l in {"jsd", "jensen_shannon"}:
        eps = 1.0e-8
        p = torch.clamp(left, eps, 1.0)
        q = torch.clamp(right, eps, 1.0)
        m = 0.5 * (p + q)
        kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=1)
        kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=1)
        return torch.mean(0.5 * (kl_pm + kl_qm))
    raise ValueError(f"Unsupported alignment distance metric: {metric}")


def per_cell_alignment_loss(left_weights: torch.Tensor, right_weights: torch.Tensor, metric: str = "l2") -> torch.Tensor:
    if left_weights.shape != right_weights.shape:
        raise ValueError(
            f"Per-cell alignment requires matching shapes, got {left_weights.shape} and {right_weights.shape}."
        )
    return alignment_distance(left_weights, right_weights, metric=metric)


def sample_mean_weights(weights: torch.Tensor, sample_ids: Iterable[str], ordered_samples: list[str]) -> torch.Tensor:
    if len(ordered_samples) == 0:
        return torch.zeros((0, weights.shape[1]), dtype=weights.dtype, device=weights.device)

    sample_ids_list = [str(v) for v in sample_ids]
    sample_to_indices: dict[str, list[int]] = {sample: [] for sample in ordered_samples}
    for idx, sample in enumerate(sample_ids_list):
        if sample in sample_to_indices:
            sample_to_indices[sample].append(idx)

    means: list[torch.Tensor] = []
    for sample in ordered_samples:
        idxs = sample_to_indices.get(sample, [])
        if len(idxs) == 0:
            means.append(torch.zeros((weights.shape[1],), dtype=weights.dtype, device=weights.device))
            continue
        means.append(weights[idxs].mean(dim=0))
    return torch.stack(means, dim=0)


def per_sample_alignment_loss(
    mean_weights_by_modality: dict[str, torch.Tensor],
    metric: str = "l2",
) -> torch.Tensor:
    modalities = list(mean_weights_by_modality.keys())
    if len(modalities) < 2:
        raise ValueError("Per-sample alignment requires at least two modalities.")

    pair_losses: list[torch.Tensor] = []
    for i, left_modality in enumerate(modalities[:-1]):
        for right_modality in modalities[i + 1 :]:
            left = mean_weights_by_modality[left_modality]
            right = mean_weights_by_modality[right_modality]
            if left.shape != right.shape:
                raise ValueError(
                    f"Per-sample alignment requires same shape for all modalities. "
                    f"Got {left_modality}: {left.shape}, {right_modality}: {right.shape}."
                )
            pair_losses.append(alignment_distance(left, right, metric=metric))

    if not pair_losses:
        first = next(iter(mean_weights_by_modality.values()))
        return torch.zeros((), dtype=first.dtype, device=first.device)
    return torch.stack(pair_losses).mean()

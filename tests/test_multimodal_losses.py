from __future__ import annotations

import torch

from cytof_archetypes.multimodal.losses import (
    alignment_distance,
    per_cell_alignment_loss,
    per_sample_alignment_loss,
    sample_mean_weights,
)


def test_alignment_distance_zero_for_identical_vectors() -> None:
    x = torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float32)
    assert torch.isclose(alignment_distance(x, x, metric="l2"), torch.tensor(0.0), atol=1.0e-7)
    assert torch.isclose(alignment_distance(x, x, metric="cosine"), torch.tensor(0.0), atol=1.0e-7)
    assert torch.isclose(alignment_distance(x, x, metric="jsd"), torch.tensor(0.0), atol=1.0e-7)


def test_per_cell_alignment_shape_guard() -> None:
    left = torch.randn(8, 3)
    right = torch.randn(8, 3)
    loss = per_cell_alignment_loss(left, right, metric="l2")
    assert loss.ndim == 0


def test_sample_mean_weights_and_per_sample_alignment() -> None:
    weights = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=torch.float32,
    )
    sample_ids = ["s1", "s1", "s2", "s2"]
    ordered = ["s1", "s2"]

    means = sample_mean_weights(weights, sample_ids=sample_ids, ordered_samples=ordered)
    assert means.shape == (2, 2)
    assert torch.allclose(means[0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(means[1], torch.tensor([0.5, 0.5]))

    aligned = per_sample_alignment_loss({"m1": means, "m2": means.clone()}, metric="l2")
    assert torch.isclose(aligned, torch.tensor(0.0), atol=1.0e-7)

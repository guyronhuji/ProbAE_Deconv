import torch

from cytof_archetypes.models import ProbabilisticArchetypalAutoencoder


def test_forward_shapes():
    model = ProbabilisticArchetypalAutoencoder(
        n_markers=32,
        n_archetypes=8,
        encoder_hidden_dims=(64, 32),
        dropout=0.0,
    )
    x = torch.randn(16, 32)
    mean, logvar, weights = model(x)
    assert mean.shape == (16, 32)
    assert logvar.shape == (16, 32)
    assert weights.shape == (16, 8)

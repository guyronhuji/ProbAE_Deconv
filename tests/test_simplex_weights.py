import torch

from cytof_archetypes.models import ProbabilisticArchetypalAutoencoder


def test_simplex_constraints():
    model = ProbabilisticArchetypalAutoencoder(
        n_markers=12,
        n_archetypes=5,
        encoder_hidden_dims=(16,),
        dropout=0.0,
    )
    x = torch.randn(10, 12)
    weights = model.encode(x)
    row_sums = weights.sum(dim=1)
    assert torch.all(weights >= 0)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

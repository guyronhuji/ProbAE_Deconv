import torch

from cytof_archetypes.models import (
    diversity_penalty,
    entropy_penalty,
    gaussian_nll,
    variance_regularization,
)


def test_losses_are_finite():
    x = torch.randn(20, 8)
    mean = torch.randn(20, 8)
    logvar = torch.zeros(20, 8)
    weights = torch.softmax(torch.randn(20, 5), dim=-1)
    archetype_means = torch.randn(5, 8)
    archetype_logvars = torch.randn(5, 8)

    nll = gaussian_nll(x, mean, logvar)
    ent = entropy_penalty(weights)
    div = diversity_penalty(archetype_means)
    var_reg = variance_regularization(archetype_logvars)

    for value in [nll, ent, div, var_reg]:
        assert torch.isfinite(value).item()

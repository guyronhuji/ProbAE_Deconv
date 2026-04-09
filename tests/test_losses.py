import torch

from cytof_archetypes.models import (
    diversity_penalty,
    entropy_penalty,
    gaussian_nll,
    nb_nll,
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


def test_nb_loss_is_finite():
    x = torch.poisson(torch.ones(25, 10) * 1.7)
    mu = torch.rand(25, 10) * 3.0 + 0.2
    theta = torch.rand(25, 10) * 2.0 + 0.1
    loss = nb_nll(x, mu, theta)
    assert torch.isfinite(loss).item()

import torch

from cytof_archetypes.models import ProbabilisticArchetypalAutoencoder, nb_nll


def test_forward_shapes_gaussian():
    model = ProbabilisticArchetypalAutoencoder(
        n_markers=32,
        n_archetypes=8,
        encoder_hidden_dims=(64, 32),
        dropout=0.0,
        decoder_family="gaussian",
    )
    x = torch.randn(16, 32)
    out = model(x)
    assert out["recon"].shape == (16, 32)
    assert out["logvar"].shape == (16, 32)
    assert out["mu"] is None
    assert out["theta"] is None
    weights = out["weights"]
    assert weights.shape == (16, 8)
    assert torch.allclose(weights.sum(dim=1), torch.ones(16), atol=1e-6)


def test_forward_shapes_nb_and_positive_params():
    model = ProbabilisticArchetypalAutoencoder(
        n_markers=20,
        n_archetypes=5,
        encoder_hidden_dims=(32, 16),
        dropout=0.0,
        decoder_family="nb",
    )
    x_target = torch.poisson(torch.ones(12, 20) * 2.5)
    lib = x_target.sum(dim=1)
    out = model(x_target, library_size=lib)
    assert out["recon"] is None
    assert out["logvar"] is None
    assert out["mu"].shape == (12, 20)
    assert out["theta"].shape == (12, 20)
    assert torch.all(out["mu"] > 0)
    assert torch.all(out["theta"] > 0)
    weights = out["weights"]
    assert torch.allclose(weights.sum(dim=1), torch.ones(12), atol=1e-6)

    loss = nb_nll(x_target, out["mu"], out["theta"])
    assert torch.isfinite(loss).item()

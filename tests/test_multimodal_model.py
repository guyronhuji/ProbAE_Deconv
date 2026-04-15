from __future__ import annotations

import torch

from cytof_archetypes.multimodal.model import MultimodalProbabilisticArchetypalAutoencoder


def test_multimodal_model_shapes() -> None:
    model = MultimodalProbabilisticArchetypalAutoencoder(
        modality_specs={
            "cytof": {
                "n_markers": 11,
                "decoder_family": "gaussian",
                "encoder_hidden_dims": [16, 8],
                "activation": "relu",
                "dropout": 0.0,
            },
            "rna": {
                "n_markers": 7,
                "decoder_family": "nb",
                "encoder_hidden_dims": [12, 6],
                "activation": "relu",
                "dropout": 0.0,
            },
        },
        n_archetypes=5,
    )

    x_cytof = torch.randn(13, 11)
    x_rna = torch.rand(13, 7)
    lib_rna = torch.sum(x_rna, dim=1)

    out_cytof = model.forward_modality("cytof", x_cytof)
    out_rna = model.forward_modality("rna", x_rna, library_size=lib_rna)

    assert out_cytof["weights"].shape == (13, 5)
    assert out_cytof["recon"].shape == (13, 11)
    assert out_cytof["logvar"].shape == (13, 11)

    assert out_rna["weights"].shape == (13, 5)
    assert out_rna["mu"].shape == (13, 7)
    assert out_rna["theta"].shape == (13, 7)

    out_joint = model(
        {
            "cytof": {"x_encoder": x_cytof, "library_size": torch.sum(torch.abs(x_cytof), dim=1)},
            "rna": {"x_encoder": x_rna, "library_size": lib_rna},
        }
    )
    assert set(out_joint.keys()) == {"cytof", "rna"}

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cytof_archetypes.multimodal.trainer import train_multimodal_from_config


def _write_modality_csv(path: Path, n_cells: int, n_markers: int, prefix: str) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    sample_ids = np.array([f"s{i % 6}" for i in range(n_cells)], dtype=object)
    cell_ids = np.array([f"{prefix}_c{i}" for i in range(n_cells)], dtype=object)
    data = {
        "cell_id": cell_ids,
        "sample_id": sample_ids,
        "batch": np.array([f"b{i % 2}" for i in range(n_cells)], dtype=object),
        "treatment": np.array(["treated" if i % 2 else "control" for i in range(n_cells)], dtype=object),
    }
    for j in range(n_markers):
        data[f"{prefix}_feat{j}"] = rng.normal(loc=0.0, scale=1.0, size=n_cells)
    frame = pd.DataFrame(data)
    frame.to_csv(path, index=False)
    return frame


def test_multimodal_trainer_smoke(tmp_path: Path) -> None:
    n_cells = 120
    left = _write_modality_csv(tmp_path / "cytof.csv", n_cells=n_cells, n_markers=8, prefix="left")
    right = _write_modality_csv(tmp_path / "rna.csv", n_cells=n_cells, n_markers=6, prefix="right")

    pairs = pd.DataFrame(
        {
            "left_cell_id": left["cell_id"].to_numpy(dtype=str),
            "right_cell_id": right["cell_id"].to_numpy(dtype=str),
        }
    )
    pairs_path = tmp_path / "pairs.csv"
    pairs.to_csv(pairs_path, index=False)

    config = {
        "seed": 7,
        "device": "cpu",
        "modalities": [
            {
                "name": "cytof",
                "input_path": str(tmp_path / "cytof.csv"),
                "cell_id_column": "cell_id",
                "sample_id_column": "sample_id",
                "obs_columns": ["batch", "treatment"],
                "marker_columns": [f"left_feat{i}" for i in range(8)],
                "preprocessing": {
                    "transform": "none",
                    "normalization": "zscore",
                    "arcsinh_cofactor": 5.0,
                    "clip_min": None,
                    "clip_max": None,
                },
                "model": {
                    "decoder_family": "gaussian",
                    "encoder_hidden_dims": [32, 16],
                    "activation": "relu",
                    "dropout": 0.0,
                },
            },
            {
                "name": "rna",
                "input_path": str(tmp_path / "rna.csv"),
                "cell_id_column": "cell_id",
                "sample_id_column": "sample_id",
                "obs_columns": ["batch", "treatment"],
                "marker_columns": [f"right_feat{i}" for i in range(6)],
                "preprocessing": {
                    "transform": "none",
                    "normalization": "zscore",
                    "arcsinh_cofactor": 5.0,
                    "clip_min": None,
                    "clip_max": None,
                },
                "model": {
                    "decoder_family": "gaussian",
                    "encoder_hidden_dims": [24, 12],
                    "activation": "relu",
                    "dropout": 0.0,
                },
            },
        ],
        "split": {
            "level": "sample",
            "val_fraction": 0.2,
            "test_fraction": 0.2,
        },
        "data": {
            "encoder_input": "log1p_normalized",
            "decoder_target": "raw_counts",
        },
        "shared_latent": {"n_archetypes": 4},
        "alignment": {
            "mode": "both",
            "distance": "l2",
            "per_cell_weight": 0.5,
            "per_sample_weight": 0.5,
            "warmup_epochs": 1,
            "cell_pairs_path": str(pairs_path),
            "left_modality": "cytof",
            "right_modality": "rna",
            "left_cell_id_column": "left_cell_id",
            "right_cell_id_column": "right_cell_id",
            "cell_batch_size": 64,
            "sample_batch_size": 4,
            "sample_max_cells_per_modality": 64,
        },
        "loss": {
            "reconstruction_weight": 1.0,
            "modality_reconstruction_weights": {"cytof": 1.0, "rna": 1.0},
            "entropy_reg_weight": 1.0e-3,
            "diversity_reg_weight": 1.0e-3,
            "variance_reg_weight": 1.0e-5,
        },
        "training": {
            "batch_size": 32,
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "max_epochs": 4,
            "patience": 3,
            "early_stopping_min_delta": 0.0,
            "grad_clip": 1.0,
            "progress_bar": False,
            "progress_bar_leave": False,
            "progress_bar_desc": "test-multimodal",
        },
        "output": {
            "base_dir": str(tmp_path / "outputs"),
            "run_name": "smoke",
        },
    }

    run_dir = train_multimodal_from_config(config)
    assert (run_dir / "metrics" / "training_history.csv").exists()
    assert (run_dir / "metrics" / "summary.json").exists()
    assert (run_dir / "metrics" / "reconstruction_metrics.csv").exists()
    assert (run_dir / "modalities" / "cytof" / "train" / "weights.csv").exists()
    assert (run_dir / "modalities" / "rna" / "train" / "weights.csv").exists()

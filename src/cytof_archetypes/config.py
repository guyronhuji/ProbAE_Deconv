from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "cpu",
    "dataset": {
        "name": "levine32",
        "input_path": "data/levine32_processed.h5ad",
        "marker_columns": None,
        "label_column": "label",
        "cell_id_column": "cell_id",
        "val_fraction": 0.15,
        "test_fraction": 0.15,
    },
    "preprocessing": {
        "transform": "none",
        "arcsinh_cofactor": 5.0,
        "normalization": "zscore",
        "clip_min": None,
        "clip_max": None,
    },
    "model": {
        "type": "probabilistic_archetypal_autoencoder",
        "n_archetypes": 8,
        "encoder_hidden_dims": [128, 64],
        "activation": "relu",
        "dropout": 0.1,
    },
    "loss": {
        "type": "gaussian_nll",
        "entropy_reg_weight": 1.0e-3,
        "diversity_reg_weight": 1.0e-3,
        "variance_reg_weight": 1.0e-5,
    },
    "training": {
        "batch_size": 256,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "max_epochs": 100,
        "patience": 15,
        "grad_clip": 1.0,
        "mixed_precision": False,
    },
    "output": {
        "base_dir": "outputs",
        "run_name": None,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}
    return deep_update(DEFAULT_CONFIG, user_cfg)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def resolve_device(config: dict[str, Any]) -> str:
    requested = str(config.get("device", "cpu"))
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    if bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

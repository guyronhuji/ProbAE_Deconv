from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MULTIMODAL_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "modalities": [
        {
            "name": "modality_1",
            "input_path": "data/modality_1.csv",
            "marker_columns": None,
            "obs_columns": None,
            "cell_id_column": "cell_id",
            "sample_id_column": "sample_id",
            "label_column": None,
            "preprocessing": {
                "transform": "none",
                "arcsinh_cofactor": 5.0,
                "normalization": "zscore",
                "clip_min": None,
                "clip_max": None,
            },
            "model": {
                "decoder_family": "gaussian",
                "encoder_hidden_dims": [128, 64],
                "activation": "relu",
                "dropout": 0.1,
                "use_observed_library_size": True,
                "size_factor_key": None,
                "dispersion": "gene",
            },
        },
        {
            "name": "modality_2",
            "input_path": "data/modality_2.csv",
            "marker_columns": None,
            "obs_columns": None,
            "cell_id_column": "cell_id",
            "sample_id_column": "sample_id",
            "label_column": None,
            "preprocessing": {
                "transform": "none",
                "arcsinh_cofactor": 5.0,
                "normalization": "zscore",
                "clip_min": None,
                "clip_max": None,
            },
            "model": {
                "decoder_family": "gaussian",
                "encoder_hidden_dims": [128, 64],
                "activation": "relu",
                "dropout": 0.1,
                "use_observed_library_size": True,
                "size_factor_key": None,
                "dispersion": "gene",
            },
        },
    ],
    "split": {
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "level": "sample",
    },
    "data": {
        "encoder_input": "log1p_normalized",
        "decoder_target": "raw_counts",
    },
    "shared_latent": {
        "n_archetypes": 8,
    },
    "alignment": {
        "mode": "none",
        "distance": "l2",
        "per_cell_weight": 1.0,
        "per_sample_weight": 1.0,
        "warmup_epochs": 0,
        "cell_pairs_path": None,
        "left_modality": None,
        "right_modality": None,
        "left_cell_id_column": "left_cell_id",
        "right_cell_id_column": "right_cell_id",
        "sample_id_column": "sample_id",
        "cell_batch_size": 256,
        "sample_batch_size": 64,
        "sample_max_cells_per_modality": 512,
    },
    "loss": {
        "reconstruction_weight": 1.0,
        "entropy_reg_weight": 1.0e-3,
        "diversity_reg_weight": 1.0e-3,
        "variance_reg_weight": 1.0e-5,
        "modality_reconstruction_weights": {},
    },
    "training": {
        "batch_size": 256,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "max_epochs": 100,
        "patience": 15,
        "early_stopping_min_delta": 0.0,
        "grad_clip": 1.0,
        "mixed_precision": False,
        "progress_bar": True,
        "progress_bar_leave": False,
        "progress_bar_desc": "multimodal-train",
    },
    "output": {
        "base_dir": "outputs/multimodal",
        "run_name": None,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_multimodal_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_MULTIMODAL_CONFIG)
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}
    merged = deep_update(DEFAULT_MULTIMODAL_CONFIG, user_cfg)
    return resolve_multimodal_paths(merged, cfg_path.parent)


def save_multimodal_config(config: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def resolve_multimodal_paths(config: dict[str, Any], config_dir: str | Path | None = None) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    if config_dir is None:
        return resolved

    root = Path(config_dir).expanduser().resolve()
    for modality in resolved.get("modalities", []):
        input_path = modality.get("input_path")
        if input_path is not None:
            path_obj = Path(input_path).expanduser()
            if not path_obj.is_absolute():
                modality["input_path"] = str((root / path_obj).resolve())
        marker_path = modality.get("marker_columns_path")
        if marker_path is not None:
            marker_obj = Path(marker_path).expanduser()
            if not marker_obj.is_absolute():
                modality["marker_columns_path"] = str((root / marker_obj).resolve())

    alignment_cfg = resolved.get("alignment", {})
    pairs_path = alignment_cfg.get("cell_pairs_path")
    if pairs_path:
        pairs_obj = Path(pairs_path).expanduser()
        if not pairs_obj.is_absolute():
            alignment_cfg["cell_pairs_path"] = str((root / pairs_obj).resolve())

    out_cfg = resolved.get("output", {})
    base_dir = out_cfg.get("base_dir")
    if base_dir is not None:
        base_obj = Path(base_dir).expanduser()
        if not base_obj.is_absolute():
            out_cfg["base_dir"] = str((root / base_obj).resolve())

    return resolved

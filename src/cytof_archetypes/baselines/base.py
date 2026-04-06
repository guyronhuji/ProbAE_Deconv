from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    split: str
    x_true: np.ndarray
    x_recon: np.ndarray
    logvar: np.ndarray
    latent: np.ndarray
    weights: np.ndarray | None
    labels: np.ndarray | None
    cell_ids: np.ndarray


@dataclass
class MethodRunResult:
    method: str
    seed: int
    representation_dim: int
    params: dict[str, Any]
    components_mean: np.ndarray | None
    components_var: np.ndarray | None
    marker_names: list[str]
    split_results: dict[str, SplitResult]
    training_history: pd.DataFrame | None = None


class BaseMethod:
    method_name: str

    def run(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        labels_val: np.ndarray | None,
        labels_test: np.ndarray | None,
        cell_ids_val: np.ndarray,
        cell_ids_test: np.ndarray,
        marker_names: list[str],
        seed: int,
        representation_dim: int,
        config: dict[str, Any],
    ) -> MethodRunResult:
        raise NotImplementedError


def gaussian_nll_per_cell(x: np.ndarray, mean: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    inv_var = np.exp(-logvar)
    per_marker = 0.5 * (logvar + ((x - mean) ** 2) * inv_var + np.log(2.0 * np.pi))
    return per_marker.sum(axis=1)


def make_unit_logvar(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x, dtype=np.float32)


def write_method_artifacts(result: MethodRunResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if result.components_mean is not None:
        pd.DataFrame(
            result.components_mean,
            index=[f"component_{i}" for i in range(result.components_mean.shape[0])],
            columns=result.marker_names,
        ).to_csv(out / "component_means.csv")
    if result.components_var is not None:
        pd.DataFrame(
            result.components_var,
            index=[f"component_{i}" for i in range(result.components_var.shape[0])],
            columns=result.marker_names,
        ).to_csv(out / "component_vars.csv")

    if result.training_history is not None:
        result.training_history.to_csv(out / "training_history.csv", index=False)

    for split, split_res in result.split_results.items():
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(split_res.x_recon, columns=result.marker_names).to_csv(
            split_dir / "reconstructions.csv", index=False
        )
        pd.DataFrame(split_res.logvar, columns=result.marker_names).to_csv(
            split_dir / "logvars.csv", index=False
        )
        pd.DataFrame(split_res.latent).to_csv(split_dir / "latent.csv", index=False)

        payload = pd.DataFrame({"cell_id": split_res.cell_ids})
        if split_res.labels is not None:
            payload["label"] = split_res.labels
        if split_res.weights is not None:
            weight_cols = [f"w_{i}" for i in range(split_res.weights.shape[1])]
            for idx, col in enumerate(weight_cols):
                payload[col] = split_res.weights[:, idx]
        payload.to_csv(split_dir / "weights.csv", index=False)

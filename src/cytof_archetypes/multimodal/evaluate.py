from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from cytof_archetypes.io import ensure_dir
from cytof_archetypes.models import beta_binomial_nll, gaussian_nll, nb_nll
from cytof_archetypes.multimodal.data import PreparedMultimodalData
from cytof_archetypes.multimodal.model import MultimodalProbabilisticArchetypalAutoencoder


class _SplitDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, x_encoder: np.ndarray, x_target: np.ndarray, library_size: np.ndarray) -> None:
        self.x_encoder = torch.from_numpy(x_encoder.astype(np.float32))
        self.x_target = torch.from_numpy(x_target.astype(np.float32))
        self.library_size = torch.from_numpy(library_size.astype(np.float32))

    def __len__(self) -> int:
        return int(self.x_encoder.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_encoder": self.x_encoder[idx],
            "x_target": self.x_target[idx],
            "library_size": self.library_size[idx],
        }


def _predict_modality_split(
    model: MultimodalProbabilisticArchetypalAutoencoder,
    modality: str,
    split,
    device: torch.device,
    batch_size: int = 2048,
) -> dict[str, np.ndarray]:
    n_markers = model.n_markers[modality]
    if len(split.x_encoder) == 0:
        return {
            "weights": np.zeros((0, model.n_archetypes), dtype=np.float32),
            "recon": np.zeros((0, n_markers), dtype=np.float32),
            "logvar": np.zeros((0, n_markers), dtype=np.float32),
            "mu": np.zeros((0, n_markers), dtype=np.float32),
            "theta": np.zeros((0, n_markers), dtype=np.float32),
            "probs": np.zeros((0, n_markers), dtype=np.float32),
            "concentration": np.zeros((0, n_markers), dtype=np.float32),
        }

    loader = DataLoader(
        _SplitDataset(split.x_encoder, split.x_target, split.library_size),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    weights: list[np.ndarray] = []
    recon: list[np.ndarray] = []
    logvar: list[np.ndarray] = []
    mu: list[np.ndarray] = []
    theta: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    concentration: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_t = {k: v.to(device) for k, v in batch.items()}
            out = model.forward_modality(modality, batch_t["x_encoder"], batch_t["library_size"])
            weights.append(out["weights"].cpu().numpy())
            if out["recon"] is not None:
                recon.append(out["recon"].cpu().numpy())
                logvar.append(out["logvar"].cpu().numpy())
                mu.append(out["recon"].cpu().numpy())
            if out["mu"] is not None:
                mu.append(out["mu"].cpu().numpy())
                theta.append(out["theta"].cpu().numpy())
            if out["probs"] is not None:
                probs.append(out["probs"].cpu().numpy())
                concentration.append(out["concentration"].cpu().numpy())
                mu.append((batch_t["library_size"].unsqueeze(1) * out["probs"]).cpu().numpy())

    return {
        "weights": np.vstack(weights).astype(np.float32),
        "recon": np.vstack(recon).astype(np.float32) if recon else np.zeros((0, n_markers), dtype=np.float32),
        "logvar": np.vstack(logvar).astype(np.float32) if logvar else np.zeros((0, n_markers), dtype=np.float32),
        "mu": np.vstack(mu).astype(np.float32) if mu else np.zeros((0, n_markers), dtype=np.float32),
        "theta": np.vstack(theta).astype(np.float32) if theta else np.zeros((0, n_markers), dtype=np.float32),
        "probs": np.vstack(probs).astype(np.float32) if probs else np.zeros((0, n_markers), dtype=np.float32),
        "concentration": (
            np.vstack(concentration).astype(np.float32)
            if concentration
            else np.zeros((0, n_markers), dtype=np.float32)
        ),
    }


def _reconstruction_metrics(
    decoder_family: str,
    x_target: np.ndarray,
    preds: dict[str, np.ndarray],
    library_size: np.ndarray,
) -> dict[str, float]:
    if len(x_target) == 0:
        return {"mse": float("nan"), "nll": float("nan")}

    mean_pred = preds["mu"] if preds["mu"].shape[0] else preds["recon"]
    mse = float(np.mean((x_target - mean_pred) ** 2))
    with torch.no_grad():
        if decoder_family == "gaussian":
            nll = float(
                gaussian_nll(
                    torch.from_numpy(x_target.astype(np.float32)),
                    torch.from_numpy(preds["recon"].astype(np.float32)),
                    torch.from_numpy(preds["logvar"].astype(np.float32)),
                    reduction="mean",
                ).item()
            )
        elif decoder_family == "nb":
            nll = float(
                nb_nll(
                    torch.from_numpy(x_target.astype(np.float32)),
                    torch.from_numpy(preds["mu"].astype(np.float32)),
                    torch.from_numpy(preds["theta"].astype(np.float32)),
                    reduction="mean",
                ).item()
            )
        else:
            nll = float(
                beta_binomial_nll(
                    m_counts=torch.from_numpy(x_target.astype(np.float32)),
                    n_counts=torch.from_numpy(library_size[:, None].astype(np.float32)),
                    probs=torch.from_numpy(preds["probs"].astype(np.float32)),
                    concentration=torch.from_numpy(preds["concentration"].astype(np.float32)),
                    reduction="mean",
                ).item()
            )
    return {"mse": mse, "nll": nll}


def _write_split_outputs(
    split_dir: Path,
    marker_names: list[str],
    split,
    preds: dict[str, np.ndarray],
) -> None:
    ensure_dir(split_dir)
    payload = pd.DataFrame({"cell_id": split.cell_ids, "sample_id": split.sample_ids})
    if split.labels is not None:
        payload["label"] = split.labels
    for idx in range(preds["weights"].shape[1]):
        payload[f"w_{idx}"] = preds["weights"][:, idx]
    payload.to_csv(split_dir / "weights.csv", index=False)

    pd.DataFrame(preds["mu"], columns=marker_names).to_csv(split_dir / "mu.csv", index=False)
    if preds["recon"].shape[0] > 0:
        pd.DataFrame(preds["recon"], columns=marker_names).to_csv(split_dir / "recon.csv", index=False)
        pd.DataFrame(preds["logvar"], columns=marker_names).to_csv(split_dir / "logvar.csv", index=False)
    if preds["theta"].shape[0] > 0:
        pd.DataFrame(preds["theta"], columns=marker_names).to_csv(split_dir / "theta.csv", index=False)
    if preds["probs"].shape[0] > 0:
        pd.DataFrame(preds["probs"], columns=marker_names).to_csv(split_dir / "probs.csv", index=False)
        pd.DataFrame(preds["concentration"], columns=marker_names).to_csv(
            split_dir / "concentration.csv", index=False
        )


def save_multimodal_outputs(
    model: MultimodalProbabilisticArchetypalAutoencoder,
    prepared: PreparedMultimodalData,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    out_modalities = ensure_dir(run_dir / "modalities")
    metrics_rows: list[dict[str, Any]] = []

    for modality_name in prepared.modality_order:
        modality = prepared.modalities[modality_name]
        modality_dir = ensure_dir(out_modalities / modality_name)

        if modality.decoder_family == "gaussian":
            means = model.archetype_means[modality_name].detach().cpu().numpy()
            logvars = model.archetype_logvars[modality_name].detach().cpu().numpy()
            pd.DataFrame(means, columns=modality.marker_names).to_csv(modality_dir / "archetype_means.csv", index=False)
            pd.DataFrame(logvars, columns=modality.marker_names).to_csv(
                modality_dir / "archetype_logvars.csv", index=False
            )
        else:
            logits = model.archetype_logits[modality_name].detach().cpu().numpy()
            pd.DataFrame(logits, columns=modality.marker_names).to_csv(
                modality_dir / "archetype_logits.csv", index=False
            )

        for split_name in ("train", "val", "test"):
            split = getattr(modality, split_name)
            preds = _predict_modality_split(
                model=model,
                modality=modality_name,
                split=split,
                device=device,
            )
            split_dir = ensure_dir(modality_dir / split_name)
            _write_split_outputs(split_dir=split_dir, marker_names=modality.marker_names, split=split, preds=preds)
            metrics = _reconstruction_metrics(
                decoder_family=modality.decoder_family,
                x_target=split.x_target,
                preds=preds,
                library_size=split.library_size,
            )
            metrics_rows.append(
                {
                    "modality": modality_name,
                    "split": split_name,
                    "mse": metrics["mse"],
                    "nll": metrics["nll"],
                    "n_cells": int(split.x_target.shape[0]),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(run_dir / "metrics" / "reconstruction_metrics.csv", index=False)

    summary: dict[str, Any] = {
        "modalities": prepared.modality_order,
        "n_archetypes": int(model.n_archetypes),
        "reconstruction_metrics": metrics_rows,
        "shared_samples": {
            split: [str(v) for v in values.tolist()] for split, values in prepared.shared_samples.items()
        },
    }
    return summary

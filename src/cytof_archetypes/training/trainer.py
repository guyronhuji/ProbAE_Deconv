from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None

from cytof_archetypes.config import resolve_device, save_config
from cytof_archetypes.datasets import Levine32Bundle, load_levine32_bundle
from cytof_archetypes.evaluation.archetypes import (
    save_archetype_outputs,
    save_cell_weights,
    save_class_weight_summary,
    save_diagnostics,
)
from cytof_archetypes.evaluation.embeddings import save_embeddings_npz
from cytof_archetypes.evaluation.metrics import compute_metrics
from cytof_archetypes.evaluation.plots import (
    plot_class_weight_heatmap,
    plot_heatmap,
    plot_loss_curve,
    plot_scatter2d,
)
from cytof_archetypes.io import ensure_dir, next_run_dir, write_json
from cytof_archetypes.models import (
    ProbabilisticArchetypalAutoencoder,
    beta_binomial_nll,
    diversity_penalty,
    entropy_penalty,
    gaussian_nll,
    nb_nll,
    variance_regularization,
)
from cytof_archetypes.preprocessing import MarkerPreprocessor
from cytof_archetypes.training.callbacks import EarlyStopping
from cytof_archetypes.utils import get_logger, set_seed


@dataclass
class PreparedSplit:
    x_encoder: np.ndarray
    x_target: np.ndarray
    library_size: np.ndarray


@dataclass
class PreparedData:
    bundle: Levine32Bundle
    preprocessor: MarkerPreprocessor | None
    train: PreparedSplit
    val: PreparedSplit
    test: PreparedSplit


class _BatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        x_encoder: np.ndarray,
        x_target: np.ndarray,
        library_size: np.ndarray,
    ) -> None:
        self._x_encoder = torch.from_numpy(x_encoder.astype(np.float32))
        self._x_target = torch.from_numpy(x_target.astype(np.float32))
        self._library_size = torch.from_numpy(library_size.astype(np.float32))

    def __len__(self) -> int:
        return int(self._x_encoder.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_encoder": self._x_encoder[idx],
            "x_target": self._x_target[idx],
            "library_size": self._library_size[idx],
        }


def _compute_library_size(x_target: np.ndarray) -> np.ndarray:
    lib = np.sum(x_target, axis=1).astype(np.float32)
    return np.clip(lib, 1e-8, None)


def _is_count_decoder(decoder_family: str) -> bool:
    return str(decoder_family).lower() in {"nb", "beta_binomial"}


def _recon_metric_key(decoder_family: str) -> str:
    family = str(decoder_family).lower()
    if family == "gaussian":
        return "nll_mean"
    if family == "nb":
        return "nb_nll_mean"
    if family == "beta_binomial":
        return "beta_binomial_nll_mean"
    raise ValueError(f"Unsupported decoder_family: {decoder_family}")


def _per_cell_nll_col(decoder_family: str) -> str:
    family = str(decoder_family).lower()
    if family == "nb":
        return "nb_nll"
    if family == "beta_binomial":
        return "beta_binomial_nll"
    raise ValueError(f"Unsupported decoder_family for count NLL: {decoder_family}")


def _build_encoder_input(x_target: np.ndarray, encoder_input: str) -> np.ndarray:
    mode = str(encoder_input).lower()
    if mode == "raw_counts":
        return x_target.astype(np.float32)
    if mode == "log1p_normalized":
        library_size = _compute_library_size(x_target)
        normalized = x_target / library_size[:, None] * 1.0e4
        return np.log1p(normalized).astype(np.float32)
    raise ValueError(f"Unsupported data.encoder_input mode: {encoder_input}")


def _prepare_nb_split(
    split_x: np.ndarray,
    split_frame: pd.DataFrame,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
) -> PreparedSplit:
    x_target = np.clip(np.asarray(split_x, dtype=np.float32), 0.0, None)
    x_encoder = _build_encoder_input(x_target, data_cfg.get("encoder_input", "log1p_normalized"))
    observed_lib = _compute_library_size(x_target)

    if bool(model_cfg.get("use_observed_library_size", True)):
        library_size = observed_lib
    else:
        size_factor_key = model_cfg.get("size_factor_key")
        if size_factor_key is None:
            library_size = observed_lib
        elif size_factor_key not in split_frame.columns:
            raise ValueError(
                f"Requested size_factor_key='{size_factor_key}' but it does not exist in split metadata."
            )
        else:
            library_size = np.asarray(split_frame[size_factor_key].to_numpy(dtype=np.float32), dtype=np.float32)
            library_size = np.clip(library_size, 1e-8, None)

    return PreparedSplit(x_encoder=x_encoder, x_target=x_target, library_size=library_size)


def _prepare_data(config: dict[str, Any]) -> PreparedData:
    dataset_cfg = config["dataset"]
    pre_cfg = config["preprocessing"]
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    decoder_family = str(model_cfg.get("decoder_family", "gaussian")).lower()

    bundle = load_levine32_bundle(
        input_path=dataset_cfg["input_path"],
        marker_columns=dataset_cfg.get("marker_columns"),
        label_column=dataset_cfg.get("label_column"),
        cell_id_column=dataset_cfg.get("cell_id_column", "cell_id"),
        val_fraction=dataset_cfg.get("val_fraction", 0.15),
        test_fraction=dataset_cfg.get("test_fraction", 0.15),
        seed=config.get("seed", 42),
    )

    if _is_count_decoder(decoder_family):
        decoder_target = str(data_cfg.get("decoder_target", "raw_counts")).lower()
        if decoder_target != "raw_counts":
            raise ValueError(f"{decoder_family} decoder currently requires data.decoder_target='raw_counts'.")
        preprocessor = None
        train = _prepare_nb_split(bundle.train.x, bundle.train.frame, model_cfg=model_cfg, data_cfg=data_cfg)
        val = _prepare_nb_split(bundle.val.x, bundle.val.frame, model_cfg=model_cfg, data_cfg=data_cfg)
        test = _prepare_nb_split(bundle.test.x, bundle.test.frame, model_cfg=model_cfg, data_cfg=data_cfg)
        return PreparedData(bundle=bundle, preprocessor=preprocessor, train=train, val=val, test=test)

    preprocessor = MarkerPreprocessor(
        transform=pre_cfg.get("transform", "none"),
        arcsinh_cofactor=pre_cfg.get("arcsinh_cofactor", 5.0),
        normalization=pre_cfg.get("normalization", "zscore"),
        clip_min=pre_cfg.get("clip_min"),
        clip_max=pre_cfg.get("clip_max"),
    )
    train_x = preprocessor.fit_transform(bundle.train.x)
    val_x = preprocessor.transform_array(bundle.val.x) if len(bundle.val.x) else bundle.val.x.astype(np.float32)
    test_x = preprocessor.transform_array(bundle.test.x) if len(bundle.test.x) else bundle.test.x.astype(np.float32)

    train = PreparedSplit(
        x_encoder=train_x,
        x_target=train_x,
        library_size=_compute_library_size(train_x),
    )
    val = PreparedSplit(
        x_encoder=val_x,
        x_target=val_x,
        library_size=_compute_library_size(val_x) if len(val_x) else np.zeros((0,), dtype=np.float32),
    )
    test = PreparedSplit(
        x_encoder=test_x,
        x_target=test_x,
        library_size=_compute_library_size(test_x) if len(test_x) else np.zeros((0,), dtype=np.float32),
    )
    return PreparedData(bundle=bundle, preprocessor=preprocessor, train=train, val=val, test=test)


def _make_loader(split: PreparedSplit, batch_size: int, shuffle: bool) -> DataLoader:
    ds = _BatchDataset(
        x_encoder=split.x_encoder,
        x_target=split.x_target,
        library_size=split.library_size,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _loss_terms(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    loss_cfg: dict[str, Any],
    decoder_family: str,
) -> dict[str, torch.Tensor]:
    out = model(batch["x_encoder"], library_size=batch["library_size"])
    weights = out["weights"]
    recon_weight = float(loss_cfg.get("reconstruction_weight", 1.0))
    if decoder_family == "nb":
        recon = nb_nll(batch["x_target"], out["mu"], out["theta"], reduction="mean")
        var_reg = torch.zeros((), device=batch["x_encoder"].device)
    elif decoder_family == "beta_binomial":
        n_counts = batch["library_size"].unsqueeze(1).expand_as(out["probs"])
        recon = beta_binomial_nll(
            m_counts=batch["x_target"],
            n_counts=n_counts,
            probs=out["probs"],
            concentration=out["concentration"],
            reduction="mean",
        )
        var_reg = torch.zeros((), device=batch["x_encoder"].device)
    else:
        recon = gaussian_nll(batch["x_target"], out["recon"], out["logvar"], reduction="mean")
        var_reg = variance_regularization(model.archetype_logvars)
    ent = entropy_penalty(weights)
    div = diversity_penalty(model.diversity_basis())
    total = (
        recon_weight * recon
        + loss_cfg.get("entropy_reg_weight", 0.0) * ent
        + loss_cfg.get("diversity_reg_weight", 0.0) * div
        + loss_cfg.get("variance_reg_weight", 0.0) * var_reg
    )
    return {
        "loss": total,
        "recon": recon,
        "entropy": ent,
        "diversity": div,
        "var_reg": var_reg,
    }


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
    decoder_family: str,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {"loss": 0.0, "recon": 0.0, "entropy": 0.0, "diversity": 0.0, "var_reg": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch_t = {k: v.to(device) for k, v in batch.items()}
            terms = _loss_terms(model, batch_t, loss_cfg, decoder_family=decoder_family)
            bs = int(batch_t["x_encoder"].shape[0])
            for key, value in terms.items():
                sums[key] += float(value.item()) * bs
            count += bs
    if count == 0:
        return {key: float("nan") for key in sums}
    return {key: value / count for key, value in sums.items()}


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_cfg: dict[str, Any],
    grad_clip: float | None,
    decoder_family: str,
) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {"loss": 0.0, "recon": 0.0, "entropy": 0.0, "diversity": 0.0, "var_reg": 0.0}
    count = 0
    for batch in loader:
        batch_t = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        terms = _loss_terms(model, batch_t, loss_cfg, decoder_family=decoder_family)
        terms["loss"].backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = int(batch_t["x_encoder"].shape[0])
        for key, value in terms.items():
            sums[key] += float(value.item()) * bs
        count += bs

    return {key: value / max(count, 1) for key, value in sums.items()}


def _predict(
    model: nn.Module,
    split: PreparedSplit,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if len(split.x_encoder) == 0:
        empty_g = np.zeros((0, model.n_markers), dtype=np.float32)
        empty_w = np.zeros((0, model.n_archetypes), dtype=np.float32)
        return {
            "weights": empty_w,
            "recon": empty_g,
            "logvar": empty_g,
            "mu": empty_g,
            "theta": empty_g,
            "probs": empty_g,
            "concentration": empty_g,
        }
    model.eval()
    loader = _make_loader(split, batch_size=2048, shuffle=False)
    recons: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    mus: list[np.ndarray] = []
    thetas: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    concentrations: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_t = {k: v.to(device) for k, v in batch.items()}
            out = model(batch_t["x_encoder"], library_size=batch_t["library_size"])
            weights.append(out["weights"].cpu().numpy())
            if out["recon"] is not None:
                recons.append(out["recon"].cpu().numpy())
                logvars.append(out["logvar"].cpu().numpy())
            if out["mu"] is not None:
                mus.append(out["mu"].cpu().numpy())
                thetas.append(out["theta"].cpu().numpy())
            if out["probs"] is not None:
                probs_batch = out["probs"]
                probs.append(probs_batch.cpu().numpy())
                mus.append((batch_t["library_size"].unsqueeze(1) * probs_batch).cpu().numpy())
                concentration_batch = out["concentration"]
                concentrations.append(concentration_batch.cpu().numpy())
    return {
        "weights": np.vstack(weights) if weights else np.zeros((0, model.n_archetypes), dtype=np.float32),
        "recon": np.vstack(recons) if recons else np.zeros((0, model.n_markers), dtype=np.float32),
        "logvar": np.vstack(logvars) if logvars else np.zeros((0, model.n_markers), dtype=np.float32),
        "mu": np.vstack(mus) if mus else np.zeros((0, model.n_markers), dtype=np.float32),
        "theta": np.vstack(thetas) if thetas else np.zeros((0, model.n_markers), dtype=np.float32),
        "probs": np.vstack(probs) if probs else np.zeros((0, model.n_markers), dtype=np.float32),
        "concentration": np.vstack(concentrations)
        if concentrations
        else np.zeros((0, model.n_markers), dtype=np.float32),
    }


def _count_per_gene_frame(x_target: np.ndarray, mu: np.ndarray, marker_names: list[str]) -> pd.DataFrame:
    if len(x_target) == 0:
        return pd.DataFrame(columns=["gene", "target_mean", "mu_mean", "mse", "mae"])
    mse = np.mean((x_target - mu) ** 2, axis=0)
    mae = np.mean(np.abs(x_target - mu), axis=0)
    return pd.DataFrame(
        {
            "gene": marker_names,
            "target_mean": np.mean(x_target, axis=0),
            "mu_mean": np.mean(mu, axis=0),
            "mse": mse,
            "mae": mae,
        }
    )


def _count_per_cell_frame(
    x_target: np.ndarray,
    mu: np.ndarray,
    library_size: np.ndarray,
    theta: np.ndarray | None,
    probs: np.ndarray | None,
    concentration: np.ndarray | None,
    decoder_family: str,
    cell_ids: np.ndarray,
    labels: np.ndarray | None,
) -> pd.DataFrame:
    nll_col = _per_cell_nll_col(decoder_family)
    if len(x_target) == 0:
        return pd.DataFrame(columns=["cell_id", nll_col, "mse", "mae", "label"])
    with torch.no_grad():
        if decoder_family == "nb":
            nll = nb_nll(
                torch.from_numpy(x_target.astype(np.float32)),
                torch.from_numpy(mu.astype(np.float32)),
                torch.from_numpy(theta.astype(np.float32)),
                reduction="none",
            ).cpu().numpy()
        elif decoder_family == "beta_binomial":
            nll = beta_binomial_nll(
                m_counts=torch.from_numpy(x_target.astype(np.float32)),
                n_counts=torch.from_numpy(library_size.astype(np.float32)),
                probs=torch.from_numpy(probs.astype(np.float32)),
                concentration=torch.from_numpy(concentration.astype(np.float32)),
                reduction="none",
            ).cpu().numpy()
        else:
            raise ValueError(f"Unsupported count decoder for per-cell metrics: {decoder_family}")
    frame = pd.DataFrame(
        {
            "cell_id": cell_ids,
            nll_col: nll,
            "mse": np.mean((x_target - mu) ** 2, axis=1),
            "mae": np.mean(np.abs(x_target - mu), axis=1),
        }
    )
    if labels is not None and len(labels) == len(frame):
        frame["label"] = labels
    return frame


def _evaluate_split_outputs(
    model: ProbabilisticArchetypalAutoencoder,
    split: PreparedSplit,
    labels: np.ndarray | None,
    cell_ids: np.ndarray,
    split_name: str,
    run_dir: Path,
    device: torch.device,
    marker_names: list[str],
) -> dict[str, float]:
    pred = _predict(model, split, device)

    weights_dir = ensure_dir(run_dir / "weights")
    metrics_dir = ensure_dir(run_dir / "metrics")
    embeddings_dir = ensure_dir(run_dir / "embeddings")

    save_cell_weights(
        weights=pred["weights"],
        cell_ids=cell_ids,
        labels=labels,
        out_path=weights_dir / f"cell_weights_{split_name}.csv",
    )
    if model.decoder_family == "gaussian":
        metrics = compute_metrics(split.x_target, pred["recon"], pred["logvar"], labels=labels)
        save_diagnostics(
            x=split.x_target,
            mean=pred["recon"],
            logvar=pred["logvar"],
            cell_ids=cell_ids,
            labels=labels,
            out_path=metrics_dir / f"{split_name}_diagnostics.csv",
        )
    else:
        per_cell_nll_key = _per_cell_nll_col(model.decoder_family)
        per_cell_df = _count_per_cell_frame(
            x_target=split.x_target,
            mu=pred["mu"],
            library_size=split.library_size,
            theta=pred["theta"] if model.decoder_family == "nb" else None,
            probs=pred["probs"] if model.decoder_family == "beta_binomial" else None,
            concentration=pred["concentration"] if model.decoder_family == "beta_binomial" else None,
            decoder_family=model.decoder_family,
            cell_ids=cell_ids,
            labels=labels,
        )
        per_gene_df = _count_per_gene_frame(
            x_target=split.x_target,
            mu=pred["mu"],
            marker_names=marker_names,
        )
        metrics = {
            f"{per_cell_nll_key}_mean": float(per_cell_df[per_cell_nll_key].mean()) if len(per_cell_df) else float("nan"),
            f"{per_cell_nll_key}_std": float(per_cell_df[per_cell_nll_key].std()) if len(per_cell_df) else float("nan"),
            "mse_mean": float(per_cell_df["mse"].mean()) if len(per_cell_df) else float("nan"),
            "mae_mean": float(per_cell_df["mae"].mean()) if len(per_cell_df) else float("nan"),
        }
        per_cell_df.to_csv(metrics_dir / f"{split_name}_per_cell_loss.csv", index=False)
        per_gene_df.to_csv(metrics_dir / f"{split_name}_per_gene_reconstruction.csv", index=False)

    save_embeddings_npz(
        weights=pred["weights"],
        labels=labels,
        out_path=embeddings_dir / f"archetype_weight_embedding_{split_name}.npz",
    )
    write_json(metrics, metrics_dir / f"{split_name}_metrics.json")
    return metrics


def _write_run_readme(run_dir: Path, config: dict[str, Any], val_metrics: dict[str, float], test_metrics: dict[str, float]) -> None:
    decoder_family = str(config.get("model", {}).get("decoder_family", "gaussian")).lower()
    nll_key = _recon_metric_key(decoder_family)
    lines = [
        "# Run Summary",
        "",
        f"- Device: `{config['device']}`",
        f"- Archetypes: `{config['model']['n_archetypes']}`",
        f"- Decoder family: `{decoder_family}`",
        f"- Markers: inferred from data",
        "",
        "## Validation",
        f"- NLL: `{val_metrics.get(nll_key, float('nan')):.6f}`",
        f"- MSE: `{val_metrics.get('mse_mean', float('nan')):.6f}`",
        "",
        "## Test",
        f"- NLL: `{test_metrics.get(nll_key, float('nan')):.6f}`",
        f"- MSE: `{test_metrics.get('mse_mean', float('nan')):.6f}`",
    ]
    (run_dir / "README_run.md").write_text("\n".join(lines), encoding="utf-8")


def _finalize_outputs(
    model: ProbabilisticArchetypalAutoencoder,
    prepared: PreparedData,
    config: dict[str, Any],
    run_dir: Path,
    history_df: pd.DataFrame,
    device: torch.device,
) -> None:
    plots_dir = ensure_dir(run_dir / "plots")
    archetypes_dir = ensure_dir(run_dir / "archetypes")
    weights_dir = ensure_dir(run_dir / "weights")
    metrics_dir = ensure_dir(run_dir / "metrics")

    plot_loss_curve(history_df, plots_dir / "loss_curve.png")

    save_archetype_outputs(
        model=model,
        marker_names=prepared.bundle.markers,
        out_dir=archetypes_dir,
    )

    if model.decoder_family == "gaussian":
        plot_heatmap(
            matrix=model.archetype_means.detach().cpu().numpy(),
            row_labels=[f"arch_{i}" for i in range(model.n_archetypes)],
            col_labels=prepared.bundle.markers,
            title="Archetype Means",
            out_path=plots_dir / "archetype_mean_heatmap.png",
        )
        plot_heatmap(
            matrix=np.exp(model.archetype_logvars.detach().cpu().numpy()),
            row_labels=[f"arch_{i}" for i in range(model.n_archetypes)],
            col_labels=prepared.bundle.markers,
            title="Archetype Variances",
            out_path=plots_dir / "archetype_variance_heatmap.png",
        )
    else:
        gene_fractions = torch.softmax(model.archetype_logits.detach(), dim=1).cpu().numpy()
        plot_heatmap(
            matrix=gene_fractions,
            row_labels=[f"arch_{i}" for i in range(model.n_archetypes)],
            col_labels=prepared.bundle.markers,
            title="Archetype Gene Fractions",
            out_path=plots_dir / "archetype_gene_fraction_heatmap.png",
        )

    val_metrics = _evaluate_split_outputs(
        model=model,
        split=prepared.val,
        labels=prepared.bundle.val.labels,
        cell_ids=prepared.bundle.val.cell_ids,
        split_name="val",
        run_dir=run_dir,
        device=device,
        marker_names=prepared.bundle.markers,
    )
    test_metrics = _evaluate_split_outputs(
        model=model,
        split=prepared.test,
        labels=prepared.bundle.test.labels,
        cell_ids=prepared.bundle.test.cell_ids,
        split_name="test",
        run_dir=run_dir,
        device=device,
        marker_names=prepared.bundle.markers,
    )

    train_pred = _predict(model, prepared.train, device)
    val_pred = _predict(model, prepared.val, device)
    test_pred = _predict(model, prepared.test, device)
    combined_weights = np.vstack([train_pred["weights"], val_pred["weights"], test_pred["weights"]])
    combined_cell_ids = np.concatenate(
        [prepared.bundle.train.cell_ids, prepared.bundle.val.cell_ids, prepared.bundle.test.cell_ids]
    )
    combined_split = np.concatenate(
        [
            np.repeat("train", len(prepared.bundle.train.cell_ids)),
            np.repeat("val", len(prepared.bundle.val.cell_ids)),
            np.repeat("test", len(prepared.bundle.test.cell_ids)),
        ]
    )
    full_weights = pd.DataFrame(combined_weights, columns=[f"w_{k}" for k in range(model.n_archetypes)])
    full_weights.insert(0, "cell_id", combined_cell_ids)
    full_weights.insert(1, "split", combined_split)
    full_weights.to_csv(weights_dir / "cell_weights.csv", index=False)

    summary_key = _recon_metric_key(model.decoder_family)
    write_json(
        {
            "decoder_family": model.decoder_family,
            "val_reconstruction_nll": float(val_metrics.get(summary_key, float("nan"))),
            "test_reconstruction_nll": float(test_metrics.get(summary_key, float("nan"))),
        },
        metrics_dir / "evaluation_summary.json",
    )

    chosen = None
    for split_x, split_labels in [
        (prepared.val, prepared.bundle.val.labels),
        (prepared.test, prepared.bundle.test.labels),
    ]:
        weights = _predict(model, split_x, device)["weights"]
        if split_labels is not None and len(split_labels) == len(weights):
            chosen = (weights, split_labels)
            break

    if chosen is not None:
        weights, labels = chosen
        class_df = save_class_weight_summary(
            weights=weights,
            labels=labels,
            out_path=weights_dir / "class_mean_weights.csv",
        )
        plot_class_weight_heatmap(
            class_df=class_df,
            out_path=plots_dir / "class_mean_weight_heatmap.png",
        )

    emb_path = run_dir / "embeddings" / "archetype_weight_embedding_val.npz"
    if emb_path.exists() and prepared.bundle.val.labels is not None:
        emb_data = np.load(emb_path, allow_pickle=True)
        coords_key = "umap_2d" if "umap_2d" in emb_data and np.any(emb_data["umap_2d"]) else "pca_2d"
        if coords_key in emb_data and emb_data[coords_key].size > 0:
            plot_scatter2d(
                coords=emb_data[coords_key],
                labels=prepared.bundle.val.labels,
                out_path=plots_dir / "umap_weights_by_label.png",
                title="Validation archetype-weight embedding",
            )

    _write_run_readme(run_dir, config, val_metrics, test_metrics)


def train_from_config(config: dict[str, Any]) -> Path:
    logger = get_logger()
    set_seed(int(config.get("seed", 42)))
    config["device"] = resolve_device(config)
    device = torch.device(config["device"])

    output_cfg = config["output"]
    if output_cfg.get("run_name"):
        run_dir = ensure_dir(Path(output_cfg.get("base_dir", "outputs")) / output_cfg["run_name"])
    else:
        run_dir = next_run_dir(output_cfg.get("base_dir", "outputs"))
    logger.info("Run directory: %s", run_dir)
    save_config(config, run_dir / "config_resolved.yaml")

    prepared = _prepare_data(config)
    if prepared.preprocessor is None:
        write_json({"mode": "none", "reason": "count_decoder_uses_raw_count_pipeline"}, run_dir / "preprocessor.json")
    else:
        write_json(prepared.preprocessor.state_dict(), run_dir / "preprocessor.json")

    model_cfg = config["model"]
    model = ProbabilisticArchetypalAutoencoder(
        n_markers=len(prepared.bundle.markers),
        n_archetypes=int(model_cfg["n_archetypes"]),
        encoder_hidden_dims=tuple(model_cfg.get("encoder_hidden_dims", [128, 64])),
        activation=model_cfg.get("activation", "relu"),
        dropout=float(model_cfg.get("dropout", 0.1)),
        decoder_family=model_cfg.get("decoder_family", "gaussian"),
        dispersion=model_cfg.get("dispersion", "gene"),
    ).to(device)
    decoder_family = str(model_cfg.get("decoder_family", "gaussian")).lower()
    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    frozen_params = int(total_params - trainable_params)

    print("\n=== Model Architecture ===")
    print(model)
    print("\n=== Model Parameters ===")
    print(f"decoder_family: {decoder_family}")
    print(f"total_params: {total_params:,}")
    print(f"trainable_params: {trainable_params:,}")
    print(f"frozen_params: {frozen_params:,}")
    print("parameter_tensors:")
    for name, param in model.named_parameters():
        shape = tuple(int(x) for x in param.shape)
        print(
            f"  - {name}: shape={shape}, count={param.numel():,}, requires_grad={bool(param.requires_grad)}"
        )
    print("")
    logger.info(
        "model decoder_family=%s total_params=%d trainable_params=%d frozen_params=%d",
        decoder_family,
        total_params,
        trainable_params,
        frozen_params,
    )

    model_summary_lines = [
        "Model Architecture",
        "===================",
        str(model),
        "",
        "Model Parameters",
        "================",
        f"decoder_family: {decoder_family}",
        f"total_params: {total_params}",
        f"trainable_params: {trainable_params}",
        f"frozen_params: {frozen_params}",
        "",
        "parameter_tensors:",
    ]
    for name, param in model.named_parameters():
        shape = tuple(int(x) for x in param.shape)
        model_summary_lines.append(
            f"- {name}: shape={shape}, count={int(param.numel())}, requires_grad={bool(param.requires_grad)}"
        )
    (run_dir / "model_summary.txt").write_text("\n".join(model_summary_lines), encoding="utf-8")

    training_cfg = config["training"]
    loss_cfg = config["loss"]
    train_loader = _make_loader(prepared.train, batch_size=int(training_cfg["batch_size"]), shuffle=True)
    val_loader = _make_loader(prepared.val, batch_size=int(training_cfg["batch_size"]), shuffle=False)
    use_progress_bar = bool(training_cfg.get("progress_bar", False)) and (tqdm is not None)
    progress_leave = bool(training_cfg.get("progress_bar_leave", False))
    progress_desc = str(training_cfg.get("progress_bar_desc", "train"))
    run_label = str(output_cfg.get("run_name") or run_dir.name)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    early_stopper = EarlyStopping(patience=int(training_cfg.get("patience", 15)))

    history_rows: list[dict[str, float]] = []
    best_ckpt_path = run_dir / "best_checkpoint.pt"
    final_ckpt_path = run_dir / "final_checkpoint.pt"

    epoch_iter: Any = range(1, int(training_cfg["max_epochs"]) + 1)
    if use_progress_bar:
        epoch_iter = tqdm(
            epoch_iter,
            desc=f"{progress_desc} [{run_label}]",
            unit="epoch",
            leave=progress_leave,
        )

    for epoch in epoch_iter:
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_cfg=loss_cfg,
            grad_clip=training_cfg.get("grad_clip"),
            decoder_family=decoder_family,
        )
        val_stats = _evaluate_loader(
            model,
            val_loader,
            device=device,
            loss_cfg=loss_cfg,
            decoder_family=decoder_family,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_recon": train_stats["recon"],
            "train_nll": train_stats["recon"],
            "train_entropy": train_stats["entropy"],
            "train_diversity": train_stats["diversity"],
            "train_var_reg": train_stats["var_reg"],
            "val_loss": val_stats["loss"],
            "val_recon": val_stats["recon"],
            "val_nll": val_stats["recon"],
            "val_entropy": val_stats["entropy"],
            "val_diversity": val_stats["diversity"],
            "val_var_reg": val_stats["var_reg"],
        }
        history_rows.append(row)

        if use_progress_bar and hasattr(epoch_iter, "set_postfix"):
            best_now = min(
                [r["val_recon"] for r in history_rows if np.isfinite(r["val_recon"])],
                default=float("nan"),
            )
            epoch_iter.set_postfix(
                {
                    "train": f"{train_stats['recon']:.4f}",
                    "val": f"{val_stats['recon']:.4f}",
                    "best": f"{best_now:.4f}" if np.isfinite(best_now) else "nan",
                },
                refresh=False,
            )
        else:
            logger.info(
                "epoch=%d train_recon=%.6f val_recon=%.6f",
                epoch,
                train_stats["recon"],
                val_stats["recon"],
            )

        improved = np.isfinite(val_stats["recon"]) and (val_stats["recon"] <= early_stopper.best)
        if improved or not best_ckpt_path.exists():
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_ckpt_path)

        should_stop = early_stopper.update(val_stats["recon"]) if np.isfinite(val_stats["recon"]) else False
        if should_stop:
            if not use_progress_bar:
                logger.info("Early stopping at epoch %d", epoch)
            break

    if use_progress_bar and hasattr(epoch_iter, "close"):
        epoch_iter.close()

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(run_dir / "training_log.csv", index=False)
    write_json(
        {
            "best_val_recon": float(history_df["val_recon"].min()),
            "final_val_recon": float(history_df["val_recon"].iloc[-1]),
            "best_val_nll": float(history_df["val_recon"].min()),
            "final_val_nll": float(history_df["val_recon"].iloc[-1]),
            "epochs_trained": int(history_df["epoch"].iloc[-1]),
            "decoder_family": decoder_family,
        },
        run_dir / "training_summary.json",
    )
    torch.save({"model_state_dict": model.state_dict(), "epoch": int(history_df["epoch"].iloc[-1])}, final_ckpt_path)

    best_state = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    _finalize_outputs(model, prepared, config, run_dir, history_df, device)
    return run_dir


def evaluate_run_dir(run_dir: str | Path, checkpoint: str = "best_checkpoint.pt") -> Path:
    run_path = Path(run_dir)
    config_path = run_path / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config in run dir: {config_path}")

    from cytof_archetypes.config import load_config

    config = load_config(config_path)
    config["device"] = resolve_device(config)
    device = torch.device(config["device"])
    prepared = _prepare_data(config)

    model_cfg = config["model"]
    model = ProbabilisticArchetypalAutoencoder(
        n_markers=len(prepared.bundle.markers),
        n_archetypes=int(model_cfg["n_archetypes"]),
        encoder_hidden_dims=tuple(model_cfg.get("encoder_hidden_dims", [128, 64])),
        activation=model_cfg.get("activation", "relu"),
        dropout=float(model_cfg.get("dropout", 0.1)),
        decoder_family=model_cfg.get("decoder_family", "gaussian"),
        dispersion=model_cfg.get("dispersion", "gene"),
    ).to(device)

    checkpoint_path = run_path / checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    history_df = pd.read_csv(run_path / "training_log.csv")
    _finalize_outputs(model, prepared, config, run_path, history_df, device)
    return run_path

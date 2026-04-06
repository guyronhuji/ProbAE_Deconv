from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    diversity_penalty,
    entropy_penalty,
    gaussian_nll,
    variance_regularization,
)
from cytof_archetypes.preprocessing import MarkerPreprocessor
from cytof_archetypes.training.callbacks import EarlyStopping
from cytof_archetypes.utils import get_logger, set_seed


@dataclass
class PreparedData:
    bundle: Levine32Bundle
    preprocessor: MarkerPreprocessor
    train_x: np.ndarray
    val_x: np.ndarray
    test_x: np.ndarray


def _prepare_data(config: dict[str, Any]) -> PreparedData:
    dataset_cfg = config["dataset"]
    pre_cfg = config["preprocessing"]
    bundle = load_levine32_bundle(
        input_path=dataset_cfg["input_path"],
        marker_columns=dataset_cfg.get("marker_columns"),
        label_column=dataset_cfg.get("label_column"),
        cell_id_column=dataset_cfg.get("cell_id_column", "cell_id"),
        val_fraction=dataset_cfg.get("val_fraction", 0.15),
        test_fraction=dataset_cfg.get("test_fraction", 0.15),
        seed=config.get("seed", 42),
    )
    preprocessor = MarkerPreprocessor(
        transform=pre_cfg.get("transform", "none"),
        arcsinh_cofactor=pre_cfg.get("arcsinh_cofactor", 5.0),
        normalization=pre_cfg.get("normalization", "zscore"),
        clip_min=pre_cfg.get("clip_min"),
        clip_max=pre_cfg.get("clip_max"),
    )
    train_x = preprocessor.fit_transform(bundle.train.x)
    val_x = preprocessor.transform_array(bundle.val.x) if len(bundle.val.x) else bundle.val.x
    test_x = preprocessor.transform_array(bundle.test.x) if len(bundle.test.x) else bundle.test.x
    return PreparedData(
        bundle=bundle,
        preprocessor=preprocessor,
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
    )


def _make_loader(x: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(x.astype(np.float32))
    ds = TensorDataset(tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _loss_terms(
    model: nn.Module,
    batch_x: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    mean, logvar, weights = model(batch_x)
    recon = gaussian_nll(batch_x, mean, logvar, reduction="mean")
    ent = entropy_penalty(weights)
    div = diversity_penalty(model.archetype_means)
    var_reg = variance_regularization(model.archetype_logvars)
    total = (
        recon
        + loss_cfg.get("entropy_reg_weight", 0.0) * ent
        + loss_cfg.get("diversity_reg_weight", 0.0) * div
        + loss_cfg.get("variance_reg_weight", 0.0) * var_reg
    )
    return {
        "loss": total,
        "nll": recon,
        "entropy": ent,
        "diversity": div,
        "var_reg": var_reg,
    }


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {"loss": 0.0, "nll": 0.0, "entropy": 0.0, "diversity": 0.0, "var_reg": 0.0}
    count = 0
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            terms = _loss_terms(model, batch_x, loss_cfg)
            bs = int(batch_x.shape[0])
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
) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {"loss": 0.0, "nll": 0.0, "entropy": 0.0, "diversity": 0.0, "var_reg": 0.0}
    count = 0
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad(set_to_none=True)
        terms = _loss_terms(model, batch_x, loss_cfg)
        terms["loss"].backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = int(batch_x.shape[0])
        for key, value in terms.items():
            sums[key] += float(value.item()) * bs
        count += bs

    return {key: value / max(count, 1) for key, value in sums.items()}


def _predict(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x) == 0:
        empty = np.zeros((0, model.n_markers), dtype=np.float32)
        empty_w = np.zeros((0, model.n_archetypes), dtype=np.float32)
        return empty, empty, empty_w
    model.eval()
    loader = _make_loader(x, batch_size=2048, shuffle=False)
    means: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            mean_t, logvar_t, w_t = model(batch_x)
            means.append(mean_t.cpu().numpy())
            logvars.append(logvar_t.cpu().numpy())
            weights.append(w_t.cpu().numpy())
    return np.vstack(means), np.vstack(logvars), np.vstack(weights)


def _evaluate_split_outputs(
    model: ProbabilisticArchetypalAutoencoder,
    x: np.ndarray,
    labels: np.ndarray | None,
    cell_ids: np.ndarray,
    split_name: str,
    run_dir: Path,
    device: torch.device,
) -> dict[str, float]:
    mean, logvar, weights = _predict(model, x, device)
    metrics = compute_metrics(x, mean, logvar, labels=labels)

    weights_dir = ensure_dir(run_dir / "weights")
    metrics_dir = ensure_dir(run_dir / "metrics")
    embeddings_dir = ensure_dir(run_dir / "embeddings")

    save_cell_weights(
        weights=weights,
        cell_ids=cell_ids,
        labels=labels,
        out_path=weights_dir / f"cell_weights_{split_name}.csv",
    )
    save_diagnostics(
        x=x,
        mean=mean,
        logvar=logvar,
        cell_ids=cell_ids,
        labels=labels,
        out_path=metrics_dir / f"{split_name}_diagnostics.csv",
    )
    save_embeddings_npz(
        weights=weights,
        labels=labels,
        out_path=embeddings_dir / f"archetype_weight_embedding_{split_name}.npz",
    )
    write_json(metrics, metrics_dir / f"{split_name}_metrics.json")
    return metrics


def _write_run_readme(run_dir: Path, config: dict[str, Any], val_metrics: dict[str, float], test_metrics: dict[str, float]) -> None:
    lines = [
        "# Run Summary",
        "",
        f"- Device: `{config['device']}`",
        f"- Archetypes: `{config['model']['n_archetypes']}`",
        f"- Markers: inferred from data",
        "",
        "## Validation",
        f"- NLL: `{val_metrics.get('nll_mean', float('nan')):.6f}`",
        f"- MSE: `{val_metrics.get('mse_mean', float('nan')):.6f}`",
        "",
        "## Test",
        f"- NLL: `{test_metrics.get('nll_mean', float('nan')):.6f}`",
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

    plot_loss_curve(history_df, plots_dir / "loss_curve.png")

    save_archetype_outputs(
        model=model,
        marker_names=prepared.bundle.markers,
        out_dir=archetypes_dir,
    )

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

    val_metrics = _evaluate_split_outputs(
        model=model,
        x=prepared.val_x,
        labels=prepared.bundle.val.labels,
        cell_ids=prepared.bundle.val.cell_ids,
        split_name="val",
        run_dir=run_dir,
        device=device,
    )
    test_metrics = _evaluate_split_outputs(
        model=model,
        x=prepared.test_x,
        labels=prepared.bundle.test.labels,
        cell_ids=prepared.bundle.test.cell_ids,
        split_name="test",
        run_dir=run_dir,
        device=device,
    )

    chosen = None
    for split_x, split_labels in [
        (prepared.val_x, prepared.bundle.val.labels),
        (prepared.test_x, prepared.bundle.test.labels),
    ]:
        _, _, weights = _predict(model, split_x, device)
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
    write_json(prepared.preprocessor.state_dict(), run_dir / "preprocessor.json")

    model_cfg = config["model"]
    model = ProbabilisticArchetypalAutoencoder(
        n_markers=len(prepared.bundle.markers),
        n_archetypes=int(model_cfg["n_archetypes"]),
        encoder_hidden_dims=tuple(model_cfg.get("encoder_hidden_dims", [128, 64])),
        activation=model_cfg.get("activation", "relu"),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    training_cfg = config["training"]
    loss_cfg = config["loss"]
    train_loader = _make_loader(prepared.train_x, batch_size=int(training_cfg["batch_size"]), shuffle=True)
    val_loader = _make_loader(prepared.val_x, batch_size=int(training_cfg["batch_size"]), shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    early_stopper = EarlyStopping(patience=int(training_cfg.get("patience", 15)))

    history_rows: list[dict[str, float]] = []
    best_ckpt_path = run_dir / "best_checkpoint.pt"
    final_ckpt_path = run_dir / "final_checkpoint.pt"

    for epoch in range(1, int(training_cfg["max_epochs"]) + 1):
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_cfg=loss_cfg,
            grad_clip=training_cfg.get("grad_clip"),
        )
        val_stats = _evaluate_loader(model, val_loader, device=device, loss_cfg=loss_cfg)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_nll": train_stats["nll"],
            "train_entropy": train_stats["entropy"],
            "train_diversity": train_stats["diversity"],
            "train_var_reg": train_stats["var_reg"],
            "val_loss": val_stats["loss"],
            "val_nll": val_stats["nll"],
            "val_entropy": val_stats["entropy"],
            "val_diversity": val_stats["diversity"],
            "val_var_reg": val_stats["var_reg"],
        }
        history_rows.append(row)

        logger.info(
            "epoch=%d train_nll=%.6f val_nll=%.6f",
            epoch,
            train_stats["nll"],
            val_stats["nll"],
        )

        improved = np.isfinite(val_stats["nll"]) and (val_stats["nll"] <= early_stopper.best)
        if improved or not best_ckpt_path.exists():
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_ckpt_path)

        should_stop = early_stopper.update(val_stats["nll"]) if np.isfinite(val_stats["nll"]) else False
        if should_stop:
            logger.info("Early stopping at epoch %d", epoch)
            break

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(run_dir / "training_log.csv", index=False)
    write_json(
        {
            "best_val_nll": float(history_df["val_nll"].min()),
            "final_val_nll": float(history_df["val_nll"].iloc[-1]),
            "epochs_trained": int(history_df["epoch"].iloc[-1]),
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
    ).to(device)

    checkpoint_path = run_path / checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    history_df = pd.read_csv(run_path / "training_log.csv")
    _finalize_outputs(model, prepared, config, run_path, history_df, device)
    return run_path

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from cytof_archetypes.evaluation.metrics import reconstruction_metrics_per_cell
from cytof_archetypes.experiments.common import BenchmarkRun


def run_rare_population_preservation(runs: list[BenchmarkRun], output_root: str | Path) -> pd.DataFrame:
    out = Path(output_root)
    tables_dir = out / "tables"
    plots_dir = out / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | int]] = []

    for run in runs:
        split = run.result.split_results["test"]
        labels = split.labels
        if labels is None:
            continue
        latent = split.weights if split.weights is not None else split.latent
        _, mse = reconstruction_metrics_per_cell(split.x_true, split.x_recon, split.logvar)

        class_rows = _per_class_metrics(labels=labels, mse=mse, latent=latent)
        for row in class_rows:
            row.update({"method": run.method, "seed": run.seed, "k": run.representation_dim})
            rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(tables_dir / "per_class_method_metrics.csv", index=False)

    if not metrics_df.empty:
        _make_rare_population_figure(runs, metrics_df, plots_dir / "rare_population_preservation.png")

    return metrics_df


def _per_class_metrics(labels: np.ndarray, mse: np.ndarray, latent: np.ndarray, knn_k: int = 15) -> list[dict[str, float | str | int]]:
    labels = labels.astype(str)
    frame = pd.DataFrame({"label": labels, "mse": mse})

    class_centroids: dict[str, np.ndarray] = {}
    class_within_var: dict[str, float] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        class_latent = latent[mask]
        class_centroids[label] = class_latent.mean(axis=0)
        class_within_var[label] = float(np.mean(np.var(class_latent, axis=0))) if len(class_latent) else float("nan")

    centroid_keys = sorted(class_centroids.keys())
    centroid_mat = np.vstack([class_centroids[k] for k in centroid_keys])
    centroid_dist = pairwise_distances(centroid_mat)

    neigh = _knn_indices(latent, knn_k=knn_k)
    per_cell_purity = np.zeros(len(labels), dtype=float)
    for i in range(len(labels)):
        if neigh.shape[1] == 0:
            per_cell_purity[i] = np.nan
        else:
            per_cell_purity[i] = np.mean(labels[neigh[i]] == labels[i])

    rows: list[dict[str, float | str | int]] = []
    for idx, label in enumerate(centroid_keys):
        mask = labels == label
        tri = centroid_dist[idx]
        sep = float(np.mean(tri[tri > 0])) if np.any(tri > 0) else float("nan")
        rows.append(
            {
                "label": label,
                "class_size": int(np.sum(mask)),
                "per_class_reconstruction_error": float(np.mean(mse[mask])),
                "per_class_neighborhood_purity": float(np.nanmean(per_cell_purity[mask])),
                "centroid_separation": sep,
                "within_class_variance": class_within_var[label],
            }
        )
    return rows


def _knn_indices(latent: np.ndarray, knn_k: int) -> np.ndarray:
    if len(latent) <= 1:
        return np.zeros((len(latent), 0), dtype=int)
    n_neighbors = min(knn_k + 1, len(latent))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(latent)
    indices = nn.kneighbors(latent, return_distance=False)
    return indices[:, 1:]


def _make_rare_population_figure(runs: list[BenchmarkRun], metrics_df: pd.DataFrame, out_path: Path) -> None:
    # Use the best run per method for cleaner visualization.
    best_runs: list[BenchmarkRun] = []
    seen_methods = sorted(set(metrics_df["method"].tolist()))
    for method in seen_methods:
        candidates = [run for run in runs if run.method == method and run.result.split_results["test"].labels is not None]
        if not candidates:
            continue
        best_runs.append(min(candidates, key=lambda run: run.val_metrics["val_mse"]))

    best_df = metrics_df[
        metrics_df.apply(lambda row: any(row.method == run.method and row.seed == run.seed and row.k == run.representation_dim for run in best_runs), axis=1)
    ].copy()

    pivot_err = best_df.pivot_table(index="label", columns="method", values="per_class_reconstruction_error", aggfunc="mean")
    pivot_purity = best_df.pivot_table(index="label", columns="method", values="per_class_neighborhood_purity", aggfunc="mean")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=False)

    im0 = axes[0, 0].imshow(pivot_err.to_numpy(), aspect="auto", cmap="magma")
    axes[0, 0].set_title("Per-class reconstruction error")
    axes[0, 0].set_xticks(np.arange(len(pivot_err.columns)))
    axes[0, 0].set_xticklabels(pivot_err.columns, rotation=45, ha="right", fontsize=8)
    axes[0, 0].set_yticks(np.arange(len(pivot_err.index)))
    axes[0, 0].set_yticklabels(pivot_err.index, fontsize=8)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.03, pad=0.02)

    im1 = axes[0, 1].imshow(pivot_purity.to_numpy(), aspect="auto", cmap="viridis")
    axes[0, 1].set_title("Per-class neighborhood purity")
    axes[0, 1].set_xticks(np.arange(len(pivot_purity.columns)))
    axes[0, 1].set_xticklabels(pivot_purity.columns, rotation=45, ha="right", fontsize=8)
    axes[0, 1].set_yticks(np.arange(len(pivot_purity.index)))
    axes[0, 1].set_yticklabels(pivot_purity.index, fontsize=8)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.03, pad=0.02)

    for method, sub_df in best_df.groupby("method"):
        axes[1, 0].scatter(sub_df["class_size"], sub_df["per_class_reconstruction_error"], s=35, alpha=0.8, label=method)
        axes[1, 1].scatter(sub_df["class_size"], sub_df["per_class_neighborhood_purity"], s=35, alpha=0.8, label=method)

    axes[1, 0].set_title("Class size vs reconstruction error")
    axes[1, 0].set_xlabel("Class size")
    axes[1, 0].set_ylabel("Per-class reconstruction error")
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].set_title("Class size vs neighborhood purity")
    axes[1, 1].set_xlabel("Class size")
    axes[1, 1].set_ylabel("Per-class neighborhood purity")
    axes[1, 1].grid(alpha=0.2)

    handles, labels = axes[1, 1].get_legend_handles_labels()
    if handles:
        axes[1, 1].legend(handles, labels, frameon=False, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)

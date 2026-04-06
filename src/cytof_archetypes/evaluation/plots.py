from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_figure(fig: plt.Figure, out: Path, save_pdf: bool = True) -> None:
    try:
        fig.tight_layout()
    except RuntimeError as exc:
        if "Colorbar layout of new layout engine not compatible" not in str(exc):
            raise
    fig.savefig(out, dpi=220)
    if save_pdf:
        fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_loss_curve(history: pd.DataFrame, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=False)
    y_train = "train_nll" if "train_nll" in history.columns else "train_loss"
    y_val = "val_nll" if "val_nll" in history.columns else "val_loss"
    ax.plot(history["epoch"], history[y_train], label=y_train)
    ax.plot(history["epoch"], history[y_val], label=y_val)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training curve")
    ax.legend()
    _save_figure(fig, out)


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    out_path: str | Path,
    cmap: str = "viridis",
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(max(6, len(col_labels) * 0.45), max(3, len(row_labels) * 0.35)),
        constrained_layout=False,
    )
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save_figure(fig, out)


def plot_class_weight_heatmap(class_df: pd.DataFrame, out_path: str | Path) -> None:
    weight_cols = [c for c in class_df.columns if c.startswith("w_") and not c.endswith("_std")]
    if not weight_cols:
        weight_cols = [c for c in class_df.columns if c.startswith("component_")]
    matrix = class_df[weight_cols].to_numpy()
    labels = class_df.index.tolist() if class_df.index.name else class_df.get("label", class_df.index).tolist()
    plot_heatmap(
        matrix=matrix,
        row_labels=[str(idx) for idx in labels],
        col_labels=weight_cols,
        title="Class mean component weights",
        out_path=out_path,
    )


def plot_scatter2d(
    coords: np.ndarray,
    labels: np.ndarray | None,
    out_path: str | Path,
    title: str,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(coords) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=False)
    if labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.8)
    else:
        labels = labels.astype(str)
        unique_labels = sorted(set(labels.tolist()))
        for label in unique_labels:
            mask = labels == label
            ax.scatter(coords[mask, 0], coords[mask, 1], s=7, alpha=0.7, label=label)
        if len(unique_labels) <= 25:
            ax.legend(loc="best", fontsize=7, frameon=False)
    ax.set_xlabel("dim_1")
    ax.set_ylabel("dim_2")
    ax.set_title(title)
    _save_figure(fig, out)


def plot_metric_vs_dim(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    method_col: str,
    out_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=False)
    for method, method_df in frame.groupby(method_col):
        grouped = method_df.groupby(x_col, as_index=False)[y_col].mean(numeric_only=True)
        ax.plot(grouped[x_col], grouped[y_col], marker="o", label=method)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    _save_figure(fig, out)


def plot_parameter_count_vs_error(
    frame: pd.DataFrame,
    param_col: str,
    error_col: str,
    method_col: str,
    out_path: str | Path,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=False)
    for method, method_df in frame.groupby(method_col):
        ax.scatter(method_df[param_col], method_df[error_col], s=35, alpha=0.8, label=method)
    ax.set_xlabel("Parameter count")
    ax.set_ylabel(error_col)
    ax.set_title("Fit vs model complexity")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    _save_figure(fig, out)


def plot_histogram(values: np.ndarray, out_path: str | Path, title: str, xlabel: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=False)
    ax.hist(values[np.isfinite(values)], bins=40, color="#3a7ca5", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cell count")
    _save_figure(fig, out)


def plot_boxplot_by_class(frame: pd.DataFrame, label_col: str, value_col: str, out_path: str | Path, title: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    groups = [grp[value_col].to_numpy() for _, grp in frame.groupby(label_col)]
    labels = [str(name) for name, _ in frame.groupby(label_col)]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.5), 4.5), constrained_layout=False)
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=65, labelsize=8)
    _save_figure(fig, out)


def plot_pareto(frame: pd.DataFrame, fit_col: str, interp_col: str, method_col: str, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=False)
    for method, sub in frame.groupby(method_col):
        ax.scatter(sub[fit_col], sub[interp_col], s=50, label=method, alpha=0.85)
    ax.set_xlabel(fit_col)
    ax.set_ylabel(interp_col)
    ax.set_title("Fit vs Interpretability")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    _save_figure(fig, out)


def plot_scatter(frame: pd.DataFrame, x_col: str, y_col: str, hue_col: str, out_path: str | Path, title: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=False)
    for hue, sub in frame.groupby(hue_col):
        ax.scatter(sub[x_col], sub[y_col], alpha=0.8, s=30, label=str(hue))
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    _save_figure(fig, out)

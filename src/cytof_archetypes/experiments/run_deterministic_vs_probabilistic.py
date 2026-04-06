from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from cytof_archetypes.evaluation.deconvolution_metrics import per_cell_weight_entropy
from cytof_archetypes.evaluation.metrics import reconstruction_metrics_per_cell
from cytof_archetypes.evaluation.plots import plot_heatmap, plot_scatter, plot_scatter2d
from cytof_archetypes.evaluation.statistics import benjamini_hochberg, paired_wilcoxon
from cytof_archetypes.experiments.common import BenchmarkRun


def run_deterministic_vs_probabilistic(runs: list[BenchmarkRun], output_root: str | Path) -> pd.DataFrame:
    out = Path(output_root)
    tables_dir = out / "tables"
    plots_dir = out / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    det = {(run.seed, run.representation_dim): run for run in runs if run.method == "deterministic_archetypal_ae"}
    prob = {(run.seed, run.representation_dim): run for run in runs if run.method == "probabilistic_archetypal_ae"}

    rows: list[dict[str, float | str | int]] = []
    p_values: list[float] = []

    for key in sorted(set(det) & set(prob)):
        det_run = det[key]
        prob_run = prob[key]
        seed, k = key

        det_test = det_run.result.split_results["test"]
        prob_test = prob_run.result.split_results["test"]

        det_nll, det_mse = reconstruction_metrics_per_cell(det_test.x_true, det_test.x_recon, det_test.logvar)
        prob_nll, prob_mse = reconstruction_metrics_per_cell(prob_test.x_true, prob_test.x_recon, prob_test.logvar)

        det_entropy = per_cell_weight_entropy(det_test.weights) if det_test.weights is not None else np.array([])
        prob_entropy = per_cell_weight_entropy(prob_test.weights) if prob_test.weights is not None else np.array([])

        wilcoxon = paired_wilcoxon(det_mse, prob_mse)
        p_values.append(wilcoxon["p_value"])

        rare_det, rare_prob = _rare_class_errors(det_test.labels, det_mse, prob_mse)
        dispersion_det, dispersion_prob = _within_class_dispersion(det_test.labels, det_test.x_true, det_test.x_recon, prob_test.x_recon)
        var_exp_det, var_exp_prob = _per_class_variance_explained(det_test.labels, det_test.x_true, det_test.x_recon, prob_test.x_recon)

        rows.append(
            {
                "seed": seed,
                "k": k,
                "det_val_mse": det_run.val_metrics["val_mse"],
                "prob_val_mse": prob_run.val_metrics["val_mse"],
                "det_test_mse": float(np.mean(det_mse)),
                "prob_test_mse": float(np.mean(prob_mse)),
                "det_test_nll": float(np.mean(det_nll)),
                "prob_test_nll": float(np.mean(prob_nll)),
                "det_weight_entropy": float(np.mean(det_entropy)) if len(det_entropy) else float("nan"),
                "prob_weight_entropy": float(np.mean(prob_entropy)) if len(prob_entropy) else float("nan"),
                "det_rare_class_error": rare_det,
                "prob_rare_class_error": rare_prob,
                "det_within_class_dispersion": dispersion_det,
                "prob_within_class_dispersion": dispersion_prob,
                "det_per_class_variance_explained": var_exp_det,
                "prob_per_class_variance_explained": var_exp_prob,
                "wilcoxon_stat": wilcoxon["stat"],
                "wilcoxon_p": wilcoxon["p_value"],
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["wilcoxon_q"] = benjamini_hochberg(summary["wilcoxon_p"].fillna(1.0).tolist())
    summary.to_csv(tables_dir / "deterministic_vs_probabilistic_summary.csv", index=False)

    if not summary.empty:
        plot_scatter(
            summary,
            x_col="det_test_mse",
            y_col="prob_test_mse",
            hue_col="k",
            out_path=plots_dir / "deterministic_vs_probabilistic_comparison.png",
            title="Deterministic vs probabilistic archetypal AE",
        )

    best = _best_probabilistic_run(runs)
    if best is not None:
        split = best.result.split_results["test"]
        _, mse = reconstruction_metrics_per_cell(split.x_true, split.x_recon, split.logvar)
        coords = _embed(split.weights if split.weights is not None else split.latent)
        if len(coords):
            error_labels = _error_bins(mse, n_bins=6)
            plot_scatter2d(
                coords=coords,
                labels=error_labels,
                out_path=plots_dir / "umap_colored_by_reconstruction_error.png",
                title="Embedding colored by reconstruction error bins",
            )

        means = best.result.components_mean
        vars_ = best.result.components_var
        if means is not None:
            plot_heatmap(
                matrix=means,
                row_labels=[f"arch_{i}" for i in range(means.shape[0])],
                col_labels=best.result.marker_names,
                title="Probabilistic archetype means",
                out_path=plots_dir / "archetype_mean_heatmap_probabilistic.png",
            )
        if vars_ is not None:
            plot_heatmap(
                matrix=vars_,
                row_labels=[f"arch_{i}" for i in range(vars_.shape[0])],
                col_labels=best.result.marker_names,
                title="Probabilistic archetype variances",
                out_path=plots_dir / "archetype_variance_heatmap_probabilistic.png",
            )

    return summary


def _rare_class_errors(labels: np.ndarray | None, det_mse: np.ndarray, prob_mse: np.ndarray) -> tuple[float, float]:
    if labels is None:
        return float("nan"), float("nan")
    series = pd.Series(labels)
    counts = series.value_counts()
    if counts.empty:
        return float("nan"), float("nan")
    cutoff = np.quantile(counts.to_numpy(), 0.2)
    rare = set(counts[counts <= cutoff].index.tolist())
    mask = np.array([lab in rare for lab in labels])
    if not np.any(mask):
        return float("nan"), float("nan")
    return float(np.mean(det_mse[mask])), float(np.mean(prob_mse[mask]))


def _within_class_dispersion(
    labels: np.ndarray | None,
    x_true: np.ndarray,
    det_recon: np.ndarray,
    prob_recon: np.ndarray,
) -> tuple[float, float]:
    if labels is None:
        return float("nan"), float("nan")
    det_residual = np.mean((x_true - det_recon) ** 2, axis=1)
    prob_residual = np.mean((x_true - prob_recon) ** 2, axis=1)
    frame = pd.DataFrame({"label": labels, "det": det_residual, "prob": prob_residual})
    det_disp = frame.groupby("label")["det"].std().mean()
    prob_disp = frame.groupby("label")["prob"].std().mean()
    return float(det_disp), float(prob_disp)


def _per_class_variance_explained(
    labels: np.ndarray | None,
    x_true: np.ndarray,
    det_recon: np.ndarray,
    prob_recon: np.ndarray,
) -> tuple[float, float]:
    if labels is None:
        return float("nan"), float("nan")
    frame = pd.DataFrame({"label": labels})
    det_scores = []
    prob_scores = []
    for label, idx in frame.groupby("label").groups.items():
        idx_array = np.asarray(list(idx), dtype=int)
        x_sub = x_true[idx_array]
        var_true = float(np.var(x_sub))
        if var_true <= 1e-10:
            continue
        det_var = float(np.var(x_sub - det_recon[idx_array]))
        prob_var = float(np.var(x_sub - prob_recon[idx_array]))
        det_scores.append(1.0 - det_var / var_true)
        prob_scores.append(1.0 - prob_var / var_true)
    if not det_scores:
        return float("nan"), float("nan")
    return float(np.mean(det_scores)), float(np.mean(prob_scores))


def _best_probabilistic_run(runs: list[BenchmarkRun]) -> BenchmarkRun | None:
    candidates = [run for run in runs if run.method == "probabilistic_archetypal_ae"]
    if not candidates:
        return None
    return min(candidates, key=lambda run: run.val_metrics["val_nll"])


def _embed(values: np.ndarray) -> np.ndarray:
    if values is None or len(values) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    try:
        import umap

        return umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.2, random_state=0).fit_transform(values)
    except Exception:
        pca = PCA(n_components=min(2, values.shape[1]), random_state=0)
        coords = pca.fit_transform(values)
        if coords.shape[1] == 1:
            coords = np.hstack([coords, np.zeros((len(coords), 1), dtype=np.float32)])
        return coords


def _error_bins(values: np.ndarray, n_bins: int = 6) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.array(["nan"] * len(values), dtype=object)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values[finite], quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.array(["bin_0"] * len(values), dtype=object)
    bins = np.digitize(values, edges[1:-1], right=False)
    return np.array([f"bin_{int(b)}" for b in bins], dtype=object)

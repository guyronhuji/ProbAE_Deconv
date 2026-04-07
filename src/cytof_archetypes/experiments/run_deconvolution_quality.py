from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from cytof_archetypes.evaluation.deconvolution_metrics import (
    class_component_means,
    class_profile_separation,
    class_purity_of_dominant,
    dominant_component_stats,
    per_cell_weight_entropy,
)
from cytof_archetypes.evaluation.plots import (
    plot_boxplot_by_class,
    plot_class_weight_heatmap,
    plot_histogram,
    plot_scatter2d,
)
from cytof_archetypes.evaluation.embeddings import umap_fit_transform_large
from cytof_archetypes.experiments.common import BenchmarkRun


def run_deconvolution_quality(runs: list[BenchmarkRun], output_root: str | Path) -> pd.DataFrame:
    out = Path(output_root)
    tables_dir = out / "tables"
    plots_dir = out / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    deconv_methods = {
        "nmf",
        "classical_archetypes",
        "deterministic_archetypal_ae",
        "probabilistic_archetypal_ae",
    }

    summary_rows: list[dict[str, float | str | int]] = []
    class_rows: list[pd.DataFrame] = []
    entropy_rows: list[dict[str, float | str | int]] = []

    filtered_runs = [run for run in runs if run.method in deconv_methods and run.result.split_results["test"].weights is not None]
    for run in filtered_runs:
        split = run.result.split_results["test"]
        weights = split.weights
        if weights is None:
            continue
        entropy = per_cell_weight_entropy(weights)
        dominant = dominant_component_stats(weights)
        class_means = class_component_means(weights, split.labels)
        purity = class_purity_of_dominant(weights, split.labels)
        separation = class_profile_separation(class_means)
        mean_entropy = float(np.mean(entropy)) if len(entropy) else float("nan")

        summary_rows.append(
            {
                "method": run.method,
                "seed": run.seed,
                "k": run.representation_dim,
                "weight_entropy_mean": mean_entropy,
                "dominant_fraction": dominant["dominant_fraction"],
                "mixed_fraction": dominant["mixed_fraction"],
                "class_purity_dominant_component": purity,
                "class_weight_profile_separation": separation,
            }
        )

        if not class_means.empty:
            class_means = class_means.copy()
            class_means.insert(0, "method", run.method)
            class_means.insert(1, "seed", run.seed)
            class_means.insert(2, "k", run.representation_dim)
            class_rows.append(class_means)

        for idx, value in enumerate(entropy):
            entropy_rows.append(
                {
                    "method": run.method,
                    "seed": run.seed,
                    "k": run.representation_dim,
                    "cell_id": str(split.cell_ids[idx]),
                    "label": "" if split.labels is None else str(split.labels[idx]),
                    "weight_entropy": float(value),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    class_df = pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame()
    entropy_df = pd.DataFrame(entropy_rows)

    summary_df.to_csv(tables_dir / "deconvolution_quality_summary.csv", index=False)
    class_df.to_csv(tables_dir / "class_component_means.csv", index=False)
    entropy_df.to_csv(tables_dir / "per_cell_weight_entropy.csv", index=False)

    # Required plots are generated from the best probabilistic run by validation NLL.
    best = _best_probabilistic_deconv_run(filtered_runs)
    if best is not None:
        split = best.result.split_results["test"]
        weights = split.weights
        assert weights is not None
        class_means = class_component_means(weights, split.labels)
        if not class_means.empty:
            class_means_indexed = class_means.set_index("label")
            plot_class_weight_heatmap(class_means_indexed, plots_dir / "class_component_mean_weight_heatmap.png")

        ent = per_cell_weight_entropy(weights)
        plot_histogram(
            ent,
            plots_dir / "weight_entropy_histogram.png",
            title="Per-cell component-weight entropy",
            xlabel="Entropy",
        )

        if split.labels is not None and len(split.labels):
            emb = _weight_embedding(weights)
            plot_scatter2d(
                emb,
                split.labels,
                plots_dir / "umap_weight_space_by_label.png",
                title="Weight-space embedding colored by label",
            )
            entropy_frame = pd.DataFrame({"label": split.labels, "weight_entropy": ent})
            plot_boxplot_by_class(
                entropy_frame,
                label_col="label",
                value_col="weight_entropy",
                out_path=plots_dir / "per_class_weight_entropy_boxplot.png",
                title="Per-class weight entropy",
            )

    return summary_df


def _best_probabilistic_deconv_run(runs: list[BenchmarkRun]) -> BenchmarkRun | None:
    candidates = [run for run in runs if run.method == "probabilistic_archetypal_ae"]
    if not candidates:
        return None
    return min(candidates, key=lambda run: run.val_metrics.get("val_nll", float("inf")))


def _weight_embedding(weights: np.ndarray) -> np.ndarray:
    try:
        return umap_fit_transform_large(
            weights, n_components=2, n_neighbors=20, min_dist=0.15, random_state=0
        )
    except Exception:
        pca = PCA(n_components=min(2, weights.shape[1]), random_state=0)
        coords = pca.fit_transform(weights)
        if coords.shape[1] == 1:
            coords = np.hstack([coords, np.zeros((len(coords), 1), dtype=np.float32)])
        return coords

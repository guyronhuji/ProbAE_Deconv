from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cytof_archetypes.evaluation.interpretability import combined_interpretability_score
from cytof_archetypes.evaluation.metrics import reconstruction_metrics_per_cell
from cytof_archetypes.evaluation.plots import plot_metric_vs_dim, plot_pareto
from cytof_archetypes.experiments.common import BenchmarkRun


def run_interpretability_tradeoff(runs: list[BenchmarkRun], output_root: str | Path) -> pd.DataFrame:
    out = Path(output_root)
    tables_dir = out / "tables"
    plots_dir = out / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    methods = {"ae", "vae", "deterministic_archetypal_ae", "probabilistic_archetypal_ae", "nmf"}

    rows: list[dict[str, float | str | int]] = []
    for run in runs:
        if run.method not in methods:
            continue
        split = run.result.split_results["test"]
        _, mse = reconstruction_metrics_per_cell(split.x_true, split.x_recon, split.logvar)
        interp = combined_interpretability_score(
            component_means=run.result.components_mean,
            weights=split.weights,
            labels=split.labels,
        )
        rows.append(
            {
                "method": run.method,
                "seed": run.seed,
                "k_or_latent_dim": run.representation_dim,
                "test_mse": float(np.mean(mse)) if len(mse) else float("nan"),
                "test_nll": run.test_metrics["test_nll"],
                "marker_coherence": interp["marker_coherence"],
                "class_specificity": interp["class_specificity"],
                "weight_sparsity": interp["weight_sparsity"],
                "entropy_sparsity": interp["entropy_sparsity"],
                "interpretability_score": interp["interpretability_score"],
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "fit_vs_interpretability.csv", index=False)

    if not table.empty:
        plot_pareto(
            frame=table,
            fit_col="test_mse",
            interp_col="interpretability_score",
            method_col="method",
            out_path=plots_dir / "pareto_fit_interpretability.png",
        )
        plot_metric_vs_dim(
            frame=table,
            x_col="k_or_latent_dim",
            y_col="class_specificity",
            method_col="method",
            out_path=plots_dir / "class_specificity_comparison.png",
            title="Class specificity across model capacity",
            ylabel="Class specificity",
        )

    return table

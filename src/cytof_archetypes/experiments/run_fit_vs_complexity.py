from __future__ import annotations

from pathlib import Path

import pandas as pd

from cytof_archetypes.evaluation.plots import (
    plot_metric_vs_dim,
    plot_parameter_count_vs_error,
)


def run_fit_vs_complexity(summary_df: pd.DataFrame, output_root: str | Path) -> pd.DataFrame:
    out = Path(output_root)
    tables_dir = out / "tables"
    plots_dir = out / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fit_df = summary_df.copy()
    fit_df = fit_df.rename(columns={"representation_dim": "k_or_latent_dim"})
    fit_df.to_csv(tables_dir / "fit_vs_complexity_summary.csv", index=False)

    plot_metric_vs_dim(
        frame=fit_df,
        x_col="k_or_latent_dim",
        y_col="test_mse",
        method_col="method",
        out_path=plots_dir / "reconstruction_vs_k.png",
        title="Reconstruction error vs K/latent dim",
        ylabel="Test MSE",
    )
    plot_metric_vs_dim(
        frame=fit_df,
        x_col="k_or_latent_dim",
        y_col="test_nll",
        method_col="method",
        out_path=plots_dir / "nll_vs_k.png",
        title="Gaussian NLL vs K/latent dim",
        ylabel="Test NLL",
    )
    plot_parameter_count_vs_error(
        frame=fit_df,
        param_col="param_count",
        error_col="val_mse",
        method_col="method",
        out_path=plots_dir / "parameter_count_vs_validation_error.png",
    )

    improvement_rows: list[dict[str, float | str | int]] = []
    for method, method_df in fit_df.groupby("method"):
        grouped = method_df.groupby("k_or_latent_dim", as_index=False)["val_mse"].mean(numeric_only=True).sort_values("k_or_latent_dim")
        grouped["delta_val_mse"] = grouped["val_mse"].diff()
        for row in grouped.itertuples(index=False):
            improvement_rows.append(
                {
                    "method": method,
                    "k_or_latent_dim": int(row.k_or_latent_dim),
                    "val_mse": float(row.val_mse),
                    "delta_val_mse": float(row.delta_val_mse) if pd.notna(row.delta_val_mse) else float("nan"),
                }
            )
    pd.DataFrame(improvement_rows).to_csv(tables_dir / "fit_vs_complexity_improvement.csv", index=False)
    return fit_df

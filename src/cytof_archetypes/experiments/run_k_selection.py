from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cytof_archetypes.evaluation.interpretability import combined_interpretability_score
from cytof_archetypes.evaluation.plots import plot_heatmap, plot_metric_vs_dim
from cytof_archetypes.evaluation.reporting import write_markdown
from cytof_archetypes.experiments.common import BenchmarkRun


def run_k_selection(runs: list[BenchmarkRun], output_root: str | Path) -> pd.DataFrame:
    out = Path(output_root)
    tables_dir = out / "tables"
    plots_dir = out / "plots"
    reports_dir = out / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    methods = {"nmf", "classical_archetypes", "deterministic_archetypal_ae", "probabilistic_archetypal_ae"}

    rows: list[dict[str, float | int | str]] = []
    for run in runs:
        if run.method not in methods:
            continue
        split = run.result.split_results["test"]
        if split.labels is not None:
            rare_err = _rare_class_error(split.labels, np.mean((split.x_true - split.x_recon) ** 2, axis=1))
        else:
            rare_err = float("nan")
        interp = combined_interpretability_score(
            component_means=run.result.components_mean,
            weights=split.weights,
            labels=split.labels,
        )
        redundancy = _component_redundancy(run.result.components_mean)
        rows.append(
            {
                "method": run.method,
                "seed": run.seed,
                "k": run.representation_dim,
                "val_mse": run.val_metrics["val_mse"],
                "test_mse": run.test_metrics["test_mse"],
                "rare_class_error": rare_err,
                "interpretability_score": interp["interpretability_score"],
                "component_redundancy": redundancy,
            }
        )

    raw = pd.DataFrame(rows)
    grouped = raw.groupby(["method", "k"], as_index=False).mean(numeric_only=True)

    scored = []
    for method, sub in grouped.groupby("method"):
        sub = sub.sort_values("k").copy()
        fit_score = _invert_minmax(sub["val_mse"].to_numpy())
        rare_score = _invert_minmax(sub["rare_class_error"].to_numpy())
        interp_score = _minmax(sub["interpretability_score"].to_numpy())
        redundancy_score = _invert_minmax(sub["component_redundancy"].to_numpy())
        sub["fit_score"] = fit_score
        sub["rare_score"] = rare_score
        sub["interp_score"] = interp_score
        sub["redundancy_score"] = redundancy_score
        sub["k_selection_score"] = np.nanmean(np.vstack([fit_score, rare_score, interp_score, redundancy_score]), axis=0)
        max_score = np.nanmax(sub["k_selection_score"].to_numpy())
        threshold = max_score - 0.02
        eligible = sub[sub["k_selection_score"] >= threshold]
        recommended_k = int(eligible["k"].min()) if not eligible.empty else int(sub.loc[sub["k_selection_score"].idxmax(), "k"])
        sub["recommended_k"] = recommended_k
        scored.append(sub)

    summary = pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()
    summary.to_csv(tables_dir / "k_selection_summary.csv", index=False)

    if not summary.empty:
        plot_metric_vs_dim(
            frame=summary,
            x_col="k",
            y_col="val_mse",
            method_col="method",
            out_path=plots_dir / "k_selection_elbow.png",
            title="Elbow curve by method",
            ylabel="Validation MSE",
        )
        heat = summary.pivot(index="method", columns="k", values="k_selection_score")
        plot_heatmap(
            matrix=heat.to_numpy(),
            row_labels=heat.index.astype(str).tolist(),
            col_labels=[str(col) for col in heat.columns.tolist()],
            title="K selection score heatmap",
            out_path=plots_dir / "k_selection_metric_heatmap.png",
        )
        _plot_per_class_vs_k(runs, plots_dir / "per_class_performance_across_k.png")

    report_text = _k_selection_report(summary)
    write_markdown(reports_dir / "k_selection_recommendation.md", report_text)

    return summary


def _rare_class_error(labels: np.ndarray, mse: np.ndarray) -> float:
    counts = pd.Series(labels).value_counts()
    cutoff = np.quantile(counts.to_numpy(), 0.2)
    rare = set(counts[counts <= cutoff].index.tolist())
    mask = np.array([label in rare for label in labels])
    return float(np.mean(mse[mask])) if np.any(mask) else float("nan")


def _component_redundancy(component_means: np.ndarray | None) -> float:
    if component_means is None or len(component_means) < 2:
        return float("nan")
    corr = np.corrcoef(component_means)
    tri = np.abs(corr[np.triu_indices_from(corr, k=1)])
    return float(np.mean(tri)) if len(tri) else float("nan")


def _minmax(values: np.ndarray) -> np.ndarray:
    vals = values.astype(float)
    if np.all(~np.isfinite(vals)):
        return np.full_like(vals, np.nan)
    lo = np.nanmin(vals)
    hi = np.nanmax(vals)
    if np.isclose(lo, hi):
        return np.ones_like(vals)
    return (vals - lo) / (hi - lo)


def _invert_minmax(values: np.ndarray) -> np.ndarray:
    return 1.0 - _minmax(values)


def _plot_per_class_vs_k(runs: list[BenchmarkRun], out_path: Path) -> None:
    # Focus on probabilistic archetypes for the per-class K trace.
    candidates = [run for run in runs if run.method == "probabilistic_archetypal_ae" and run.result.split_results["test"].labels is not None]
    if not candidates:
        return
    rows: list[dict[str, float | str | int]] = []
    for run in candidates:
        split = run.result.split_results["test"]
        labels = split.labels
        assert labels is not None
        mse = np.mean((split.x_true - split.x_recon) ** 2, axis=1)
        frame = pd.DataFrame({"label": labels, "mse": mse})
        agg = frame.groupby("label", as_index=False).mean(numeric_only=True)
        for row in agg.itertuples(index=False):
            rows.append({"label": row.label, "k": run.representation_dim, "mse": row.mse, "seed": run.seed})
    if not rows:
        return
    table = pd.DataFrame(rows).groupby(["label", "k"], as_index=False).mean(numeric_only=True)
    pivot = table.pivot(index="label", columns="k", values="mse")
    plot_heatmap(
        matrix=pivot.to_numpy(),
        row_labels=pivot.index.astype(str).tolist(),
        col_labels=[str(v) for v in pivot.columns.tolist()],
        title="Per-class reconstruction error across K (probabilistic)",
        out_path=out_path,
        cmap="magma",
    )


def _k_selection_report(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "# K Selection Recommendation\n\nNo K-selection records were generated."

    lines = ["# K Selection Recommendation", ""]
    for method, sub in summary.groupby("method"):
        rec = int(sub["recommended_k"].iloc[0])
        best = sub.loc[sub["k_selection_score"].idxmax()]
        lines.extend(
            [
                f"## {method}",
                f"- Recommended smallest near-optimal K: `{rec}`",
                f"- Best score observed at K={int(best['k'])}: `{float(best['k_selection_score']):.4f}`",
                f"- Selection score combines fit, rare-class error, interpretability, and redundancy.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation guidance",
            "- Prefer the smallest recommended K unless a larger K is required for specific biological hypotheses.",
            "- Check rare-class preservation and component redundancy before finalizing figures.",
        ]
    )
    return "\n".join(lines)

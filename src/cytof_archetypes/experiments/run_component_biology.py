from __future__ import annotations

from pathlib import Path

import pandas as pd

from cytof_archetypes.evaluation.interpretability import (
    component_marker_profile_table,
    marker_group_enrichment,
    top_markers_table,
)
from cytof_archetypes.evaluation.plots import plot_heatmap
from cytof_archetypes.experiments.common import BenchmarkRun


def run_component_biology(
    runs: list[BenchmarkRun],
    output_root: str | Path,
    marker_groups: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(output_root)
    tables_dir = out / "tables"
    plots_dir = out / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    methods = {"nmf", "classical_archetypes", "deterministic_archetypal_ae", "probabilistic_archetypal_ae"}

    profile_frames: list[pd.DataFrame] = []
    top_marker_frames: list[pd.DataFrame] = []
    for run in runs:
        if run.method not in methods:
            continue
        means = run.result.components_mean
        if means is None:
            continue
        profile_frames.append(
            component_marker_profile_table(
                method=run.method,
                k=run.representation_dim,
                seed=run.seed,
                marker_names=run.result.marker_names,
                component_means=means,
                component_vars=run.result.components_var,
            )
        )
        top_marker_frames.append(
            top_markers_table(
                method=run.method,
                k=run.representation_dim,
                seed=run.seed,
                marker_names=run.result.marker_names,
                component_means=means,
                top_n=8,
            )
        )

    profiles = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame()
    enrichment = marker_group_enrichment(profiles, marker_groups=marker_groups)
    top_markers = pd.concat(top_marker_frames, ignore_index=True) if top_marker_frames else pd.DataFrame()

    profiles.to_csv(tables_dir / "component_marker_profiles.csv", index=False)
    enrichment.to_csv(tables_dir / "component_marker_enrichment.csv", index=False)
    top_markers.to_csv(tables_dir / "component_top_markers.csv", index=False)

    best_prob = _best_probabilistic_run(runs)
    if best_prob is not None and best_prob.result.components_mean is not None:
        plot_heatmap(
            matrix=best_prob.result.components_mean,
            row_labels=[f"component_{i}" for i in range(best_prob.result.components_mean.shape[0])],
            col_labels=best_prob.result.marker_names,
            title="Component marker heatmap (best probabilistic run)",
            out_path=plots_dir / "component_marker_heatmap.png",
        )

    return profiles, enrichment


def _best_probabilistic_run(runs: list[BenchmarkRun]) -> BenchmarkRun | None:
    candidates = [run for run in runs if run.method == "probabilistic_archetypal_ae" and run.result.components_mean is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda run: run.val_metrics.get("val_nll", float("inf")))

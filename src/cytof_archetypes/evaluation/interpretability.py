from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from cytof_archetypes.evaluation.deconvolution_metrics import class_component_means, per_cell_weight_entropy


def marker_coherence_score(component_means: np.ndarray | None) -> float:
    if component_means is None or component_means.size == 0 or len(component_means) < 2:
        return float("nan")
    corr = np.corrcoef(component_means)
    if corr.ndim < 2:
        return float("nan")
    tri = np.abs(corr[np.triu_indices_from(corr, k=1)])
    if len(tri) == 0:
        return float("nan")
    # Lower inter-component correlation means better separation/coherence.
    return float(1.0 - np.mean(tri))


def class_specificity_score(weights: np.ndarray | None, labels: np.ndarray | None) -> float:
    if weights is None or labels is None or len(weights) == 0:
        return float("nan")
    means = class_component_means(weights, labels)
    if means.empty:
        return float("nan")
    cols = [c for c in means.columns if c.startswith("component_")]
    if len(cols) < 2:
        return float("nan")
    dist = pairwise_distances(means[cols].to_numpy(), metric="euclidean")
    tri = dist[np.triu_indices_from(dist, k=1)]
    return float(np.mean(tri)) if len(tri) else float("nan")


def sparsity_score(weights: np.ndarray | None) -> float:
    if weights is None or len(weights) == 0:
        return float("nan")
    normalized = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
    l1 = np.sum(np.abs(normalized), axis=1)
    l2 = np.sqrt(np.sum(normalized**2, axis=1))
    k = normalized.shape[1]
    sparsity = (np.sqrt(k) - (l1 / np.clip(l2, 1e-8, None))) / (np.sqrt(k) - 1.0)
    return float(np.mean(sparsity))


def entropy_sparsity_score(weights: np.ndarray | None) -> float:
    if weights is None or len(weights) == 0:
        return float("nan")
    entropy = per_cell_weight_entropy(weights)
    k = weights.shape[1]
    max_entropy = np.log(max(k, 2))
    return float(1.0 - np.mean(entropy) / max_entropy)


def combined_interpretability_score(
    component_means: np.ndarray | None,
    weights: np.ndarray | None,
    labels: np.ndarray | None,
) -> dict[str, float]:
    marker_coh = marker_coherence_score(component_means)
    class_spec = class_specificity_score(weights, labels)
    sparsity = sparsity_score(weights)
    ent_sparse = entropy_sparsity_score(weights)
    pieces = [v for v in [marker_coh, class_spec, sparsity, ent_sparse] if np.isfinite(v)]
    total = float(np.mean(pieces)) if pieces else float("nan")
    return {
        "marker_coherence": marker_coh,
        "class_specificity": class_spec,
        "weight_sparsity": sparsity,
        "entropy_sparsity": ent_sparse,
        "interpretability_score": total,
    }


def component_marker_profile_table(
    method: str,
    k: int,
    seed: int,
    marker_names: list[str],
    component_means: np.ndarray,
    component_vars: np.ndarray | None,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for comp_idx in range(component_means.shape[0]):
        for marker_idx, marker in enumerate(marker_names):
            row: dict[str, float | str | int] = {
                "method": method,
                "k": k,
                "seed": seed,
                "component": comp_idx,
                "marker": marker,
                "mean": float(component_means[comp_idx, marker_idx]),
            }
            if component_vars is not None and component_vars.shape == component_means.shape:
                row["variance"] = float(component_vars[comp_idx, marker_idx])
            rows.append(row)
    return pd.DataFrame(rows)


def top_markers_table(
    method: str,
    k: int,
    seed: int,
    marker_names: list[str],
    component_means: np.ndarray,
    top_n: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, str | int | float]] = []
    for comp_idx in range(component_means.shape[0]):
        scores = component_means[comp_idx]
        ranked = np.argsort(-np.abs(scores))[:top_n]
        for rank, idx in enumerate(ranked, start=1):
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "seed": seed,
                    "component": comp_idx,
                    "rank": rank,
                    "marker": marker_names[idx],
                    "score": float(scores[idx]),
                }
            )
    return pd.DataFrame(rows)


def marker_group_enrichment(
    component_profiles: pd.DataFrame,
    marker_groups: dict[str, list[str]] | None,
) -> pd.DataFrame:
    if component_profiles.empty:
        return pd.DataFrame()
    if not marker_groups:
        return pd.DataFrame(
            columns=[
                "method",
                "k",
                "seed",
                "component",
                "marker_group",
                "mean_abs_score",
                "n_markers_in_group",
            ]
        )

    rows: list[dict[str, float | int | str]] = []
    for (method, k, seed, comp), sub_df in component_profiles.groupby(["method", "k", "seed", "component"]):
        marker_to_score = {row.marker: abs(float(row.mean)) for row in sub_df.itertuples(index=False)}
        for group_name, markers in marker_groups.items():
            vals = [marker_to_score[m] for m in markers if m in marker_to_score]
            rows.append(
                {
                    "method": method,
                    "k": int(k),
                    "seed": int(seed),
                    "component": int(comp),
                    "marker_group": group_name,
                    "mean_abs_score": float(np.mean(vals)) if vals else float("nan"),
                    "n_markers_in_group": int(len(vals)),
                }
            )
    return pd.DataFrame(rows)

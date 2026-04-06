from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def per_cell_weight_entropy(weights: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if weights.size == 0:
        return np.zeros((0,), dtype=np.float32)
    normalized = _normalize_rows(weights, eps=eps)
    entropy = -(normalized * np.log(normalized + eps)).sum(axis=1)
    return entropy.astype(np.float32)


def dominant_component_stats(weights: np.ndarray, dominant_threshold: float = 0.7) -> dict[str, float]:
    if weights.size == 0:
        return {
            "dominant_fraction": float("nan"),
            "mixed_fraction": float("nan"),
        }
    normalized = _normalize_rows(weights)
    dominant = normalized.max(axis=1)
    dominant_fraction = float(np.mean(dominant >= dominant_threshold))
    return {
        "dominant_fraction": dominant_fraction,
        "mixed_fraction": float(1.0 - dominant_fraction),
    }


def class_component_means(weights: np.ndarray, labels: np.ndarray | None) -> pd.DataFrame:
    if labels is None or weights.size == 0:
        return pd.DataFrame()
    frame = pd.DataFrame(weights, columns=[f"component_{i}" for i in range(weights.shape[1])])
    frame["label"] = labels
    return frame.groupby("label", as_index=False).mean(numeric_only=True)


def class_purity_of_dominant(weights: np.ndarray, labels: np.ndarray | None) -> float:
    if labels is None or weights.size == 0:
        return float("nan")
    normalized = _normalize_rows(weights)
    dominant = normalized.argmax(axis=1)
    frame = pd.DataFrame({"label": labels.astype(str), "dom": dominant.astype(int)})
    purity = []
    for comp, comp_df in frame.groupby("dom"):
        if len(comp_df) == 0:
            continue
        purity.append(comp_df["label"].value_counts(normalize=True).iloc[0])
    if not purity:
        return float("nan")
    return float(np.mean(purity))


def class_profile_separation(class_component_mean_df: pd.DataFrame) -> float:
    if class_component_mean_df.empty or len(class_component_mean_df) < 2:
        return float("nan")
    cols = [c for c in class_component_mean_df.columns if c.startswith("component_")]
    if not cols:
        return float("nan")
    values = class_component_mean_df[cols].to_numpy()
    dist = pairwise_distances(values, metric="euclidean")
    tri = dist[np.triu_indices_from(dist, k=1)]
    return float(np.mean(tri)) if len(tri) else float("nan")


def per_class_entropy(weights: np.ndarray, labels: np.ndarray | None) -> pd.DataFrame:
    if labels is None or weights.size == 0:
        return pd.DataFrame(columns=["label", "weight_entropy"])
    entropy = per_cell_weight_entropy(weights)
    frame = pd.DataFrame({"label": labels, "weight_entropy": entropy})
    return frame


def _normalize_rows(weights: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.clip(weights.sum(axis=1, keepdims=True), eps, None)
    return weights / denom

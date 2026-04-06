from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


def compute_metrics(
    x: np.ndarray,
    mean: np.ndarray,
    logvar: np.ndarray,
    labels: np.ndarray | None = None,
) -> dict[str, float]:
    if len(x) == 0:
        return {"nll_mean": float("nan"), "mse_mean": float("nan")}

    nll_per_cell, mse_per_cell = reconstruction_metrics_per_cell(x, mean, logvar)
    metrics: dict[str, float] = {
        "nll_mean": float(np.mean(nll_per_cell)),
        "nll_std": float(np.std(nll_per_cell)),
        "mse_mean": float(np.mean(mse_per_cell)),
        "mse_std": float(np.std(mse_per_cell)),
    }

    per_marker_mse = np.mean((x - mean) ** 2, axis=0)
    for idx, marker_value in enumerate(per_marker_mse):
        metrics[f"marker_{idx:02d}_mse"] = float(marker_value)

    if labels is not None and len(labels) == len(x):
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            metrics[f"class_{label}_nll"] = float(np.mean(nll_per_cell[mask]))
            metrics[f"class_{label}_mse"] = float(np.mean(mse_per_cell[mask]))

    return metrics


def reconstruction_metrics_per_cell(
    x: np.ndarray,
    mean: np.ndarray,
    logvar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inv_var = np.exp(-logvar)
    nll_per_marker = 0.5 * (logvar + ((x - mean) ** 2) * inv_var + np.log(2.0 * np.pi))
    nll_per_cell = nll_per_marker.sum(axis=1)
    mse_per_cell = np.mean((x - mean) ** 2, axis=1)
    return nll_per_cell, mse_per_cell


def per_class_reconstruction_frame(
    labels: np.ndarray | None,
    nll_per_cell: np.ndarray,
    mse_per_cell: np.ndarray,
) -> pd.DataFrame:
    if labels is None or len(labels) == 0:
        return pd.DataFrame(columns=["label", "n_cells", "mse", "nll"])
    frame = pd.DataFrame({"label": labels, "mse": mse_per_cell, "nll": nll_per_cell})
    out = frame.groupby("label", as_index=False).agg(
        n_cells=("label", "size"),
        mse=("mse", "mean"),
        nll=("nll", "mean"),
    )
    return out


def representation_structure_metrics(
    latent: np.ndarray,
    labels: np.ndarray | None,
    n_clusters: int | None = None,
    knn_k: int = 15,
) -> dict[str, float]:
    if labels is None or len(labels) == 0 or len(latent) < 3:
        return {
            "ari": float("nan"),
            "nmi": float("nan"),
            "knn_purity": float("nan"),
            "silhouette": float("nan"),
        }

    labels = labels.astype(str)
    classes = sorted(set(labels.tolist()))
    k = n_clusters or max(2, len(classes))
    k = min(k, len(latent) - 1)
    if k < 2:
        return {
            "ari": float("nan"),
            "nmi": float("nan"),
            "knn_purity": float("nan"),
            "silhouette": float("nan"),
        }

    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
    pred = km.fit_predict(latent)

    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)

    n_neighbors = min(knn_k + 1, len(latent))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(latent)
    neigh = nn.kneighbors(latent, return_distance=False)
    neigh = neigh[:, 1:]
    purity = []
    for idx in range(len(latent)):
        if neigh.shape[1] == 0:
            purity.append(np.nan)
        else:
            purity.append(float(np.mean(labels[neigh[idx]] == labels[idx])))
    knn_purity = float(np.nanmean(purity))

    silhouette = float("nan")
    if len(set(labels.tolist())) > 1:
        try:
            silhouette = float(silhouette_score(latent, labels))
        except Exception:
            silhouette = float("nan")

    return {
        "ari": float(ari),
        "nmi": float(nmi),
        "knn_purity": knn_purity,
        "silhouette": silhouette,
    }

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def save_embeddings_npz(
    weights: np.ndarray,
    labels: np.ndarray | None,
    out_path: str | Path,
    random_state: int = 42,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}

    if len(weights) == 0:
        payload["pca_2d"] = np.zeros((0, 2), dtype=np.float32)
        payload["umap_2d"] = np.zeros((0, 2), dtype=np.float32)
    else:
        pca = PCA(n_components=min(2, weights.shape[1]), random_state=random_state)
        pca_2d = pca.fit_transform(weights)
        if pca_2d.shape[1] == 1:
            pca_2d = np.hstack([pca_2d, np.zeros((pca_2d.shape[0], 1))])
        payload["pca_2d"] = pca_2d.astype(np.float32)

        try:
            import umap

            umap_model = umap.UMAP(
                n_components=2,
                n_neighbors=20,
                min_dist=0.15,
                random_state=random_state,
            )
            payload["umap_2d"] = umap_model.fit_transform(weights).astype(np.float32)
        except Exception:
            payload["umap_2d"] = np.zeros((weights.shape[0], 2), dtype=np.float32)

    if labels is not None:
        payload["labels"] = labels.astype(str)
    np.savez(out, **payload)

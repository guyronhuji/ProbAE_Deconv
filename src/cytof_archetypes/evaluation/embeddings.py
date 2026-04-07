from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def umap_fit_transform_large(
    x: np.ndarray,
    fit_subsample: int = 50_000,
    chunk_size: int = 100_000,
    random_state: int = 42,
    **umap_kwargs,
) -> np.ndarray:
    """UMAP for large arrays: fit on a subsample, transform the rest in chunks.

    Fitting UMAP on >50k points is very slow. Instead:
    1. Fit on min(n, fit_subsample) randomly chosen rows.
    2. Transform the full array in chunks of chunk_size using the fitted model.

    Parameters
    ----------
    x:              Input array of shape (n_samples, n_features).
    fit_subsample:  Max rows to use for fitting. All rows used if n <= this.
    chunk_size:     Rows per transform chunk (controls peak memory).
    random_state:   Random seed for UMAP and subsampling.
    **umap_kwargs:  Passed to umap.UMAP (n_components, n_neighbors, etc.).
    """
    import umap as umap_lib

    n = len(x)
    rng = np.random.default_rng(random_state)

    if n <= fit_subsample:
        fit_x = x
        print(f"[umap] fitting on all {n:,} points ...", flush=True)
    else:
        idx = rng.choice(n, size=fit_subsample, replace=False)
        fit_x = x[idx]
        print(f"[umap] fitting on {fit_subsample:,} / {n:,} points (subsample) ...", flush=True)

    model = umap_lib.UMAP(random_state=random_state, verbose=True, **umap_kwargs)
    model.fit(fit_x)

    n_chunks = (n + chunk_size - 1) // chunk_size
    print(f"[umap] transforming {n:,} points in {n_chunks} chunk(s) of {chunk_size:,} ...", flush=True)
    coords = np.empty((n, model.n_components), dtype=np.float32)
    for i, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        print(f"[umap]   chunk {i+1}/{n_chunks}: rows {start:,}–{end:,}", flush=True)
        coords[start:end] = model.transform(x[start:end]).astype(np.float32)

    print(f"[umap] done. Output shape: {coords.shape}", flush=True)
    return coords


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
            payload["umap_2d"] = umap_fit_transform_large(
                weights,
                n_components=2,
                n_neighbors=20,
                min_dist=0.15,
                random_state=random_state,
            )
        except Exception:
            payload["umap_2d"] = np.zeros((weights.shape[0], 2), dtype=np.float32)

    if labels is not None:
        payload["labels"] = labels.astype(str)
    np.savez(out, **payload)

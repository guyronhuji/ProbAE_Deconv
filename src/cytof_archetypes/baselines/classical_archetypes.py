from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from cytof_archetypes.baselines.base import BaseMethod, MethodRunResult, SplitResult, make_unit_logvar


class ClassicalArchetypeMethod(BaseMethod):
    method_name = "classical_archetypes"

    def run(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        cell_ids_train: np.ndarray,
        labels_val: np.ndarray | None,
        labels_test: np.ndarray | None,
        cell_ids_val: np.ndarray,
        cell_ids_test: np.ndarray,
        marker_names: list[str],
        seed: int,
        representation_dim: int,
        config: dict,
    ) -> MethodRunResult:
        n_iters = int(config.get("n_iters", 40))
        lr = float(config.get("lr", 0.1))
        pg_steps = int(config.get("pg_steps", 80))

        kmeans = KMeans(n_clusters=representation_dim, random_state=seed, n_init="auto")
        archetypes = kmeans.fit(x_train).cluster_centers_.astype(np.float32)

        for _ in range(n_iters):
            weights = _solve_simplex_weights(x_train, archetypes, steps=pg_steps, lr=lr)
            denom = np.clip(weights.sum(axis=0)[:, None], 1e-8, None)
            archetypes = (weights.T @ x_train) / denom

        w_train = _solve_simplex_weights(x_train, archetypes, steps=pg_steps, lr=lr)
        w_val = _solve_simplex_weights(x_val, archetypes, steps=pg_steps, lr=lr) if len(x_val) else np.zeros((0, representation_dim), dtype=np.float32)
        w_test = _solve_simplex_weights(x_test, archetypes, steps=pg_steps, lr=lr) if len(x_test) else np.zeros((0, representation_dim), dtype=np.float32)

        recon_train = w_train @ archetypes
        recon_val = w_val @ archetypes
        recon_test = w_test @ archetypes

        return MethodRunResult(
            method=self.method_name,
            seed=seed,
            representation_dim=representation_dim,
            params={"n_iters": n_iters, "pg_steps": pg_steps, "param_count": int(archetypes.size)},
            components_mean=archetypes.astype(np.float32),
            components_var=None,
            marker_names=marker_names,
            split_results={
                "train": SplitResult(
                    split="train",
                    x_true=x_train,
                    x_recon=recon_train.astype(np.float32),
                    logvar=make_unit_logvar(x_train),
                    latent=w_train.astype(np.float32),
                    weights=w_train.astype(np.float32),
                    labels=None,
                    cell_ids=cell_ids_train,
                ),
                "val": SplitResult(
                    split="val",
                    x_true=x_val,
                    x_recon=recon_val.astype(np.float32),
                    logvar=make_unit_logvar(x_val),
                    latent=w_val.astype(np.float32),
                    weights=w_val.astype(np.float32),
                    labels=labels_val,
                    cell_ids=cell_ids_val,
                ),
                "test": SplitResult(
                    split="test",
                    x_true=x_test,
                    x_recon=recon_test.astype(np.float32),
                    logvar=make_unit_logvar(x_test),
                    latent=w_test.astype(np.float32),
                    weights=w_test.astype(np.float32),
                    labels=labels_test,
                    cell_ids=cell_ids_test,
                ),
            },
            training_history=None,
        )


def _project_rows_simplex(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    sorted_m = -np.sort(-matrix, axis=1)
    cumsum = np.cumsum(sorted_m, axis=1)
    ks = np.arange(1, matrix.shape[1] + 1)
    cond = sorted_m + (1.0 - cumsum) / ks > 0
    rho = cond.sum(axis=1) - 1
    theta = (cumsum[np.arange(matrix.shape[0]), rho] - 1.0) / (rho + 1)
    projected = np.maximum(matrix - theta[:, None], 0.0)
    return projected.astype(np.float32)


def _solve_simplex_weights(x: np.ndarray, archetypes: np.ndarray, steps: int = 80, lr: float = 0.1) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((0, archetypes.shape[0]), dtype=np.float32)
    weights = np.full((x.shape[0], archetypes.shape[0]), 1.0 / archetypes.shape[0], dtype=np.float32)
    zzt = archetypes @ archetypes.T
    xzt = x @ archetypes.T
    for _ in range(steps):
        grad = weights @ zzt - xzt
        weights = weights - lr * grad
        weights = _project_rows_simplex(weights)
    return weights.astype(np.float32)

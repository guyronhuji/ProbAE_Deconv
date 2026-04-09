from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF

from cytof_archetypes.baselines.base import BaseMethod, MethodRunResult, SplitResult, make_unit_logvar


class NMFMethod(BaseMethod):
    method_name = "nmf"

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
        eps = float(config.get("eps", 1e-6))
        min_value = float(min(x_train.min(), x_val.min() if len(x_val) else 0.0, x_test.min() if len(x_test) else 0.0))
        shift = -min_value + eps if min_value <= 0 else 0.0

        train_shifted = x_train + shift
        val_shifted = x_val + shift
        test_shifted = x_test + shift

        model = NMF(
            n_components=representation_dim,
            init="nndsvda",
            random_state=seed,
            max_iter=int(config.get("max_iter", 1200)),
            alpha_W=float(config.get("alpha_w", 0.0)),
            alpha_H=float(config.get("alpha_h", 0.0)),
            l1_ratio=float(config.get("l1_ratio", 0.0)),
        )

        w_train = model.fit_transform(train_shifted)
        h = model.components_

        w_val = model.transform(val_shifted) if len(val_shifted) else np.zeros((0, representation_dim), dtype=np.float32)
        w_test = model.transform(test_shifted) if len(test_shifted) else np.zeros((0, representation_dim), dtype=np.float32)

        recon_train = w_train @ h - shift
        recon_val = w_val @ h - shift
        recon_test = w_test @ h - shift

        return MethodRunResult(
            method=self.method_name,
            seed=seed,
            representation_dim=representation_dim,
            params={
                "shift": shift,
                "max_iter": int(config.get("max_iter", 1200)),
                "param_count": int(h.size),
            },
            components_mean=h.astype(np.float32),
            components_var=None,
            marker_names=marker_names,
            split_results={
                "train": SplitResult(
                    split="train",
                    x_true=x_train,
                    x_recon=recon_train.astype(np.float32),
                    logvar=make_unit_logvar(x_train),
                    latent=w_train.astype(np.float32),
                    weights=_normalize_rows(w_train),
                    labels=None,
                    cell_ids=cell_ids_train,
                ),
                "val": SplitResult(
                    split="val",
                    x_true=x_val,
                    x_recon=recon_val.astype(np.float32),
                    logvar=make_unit_logvar(x_val),
                    latent=w_val.astype(np.float32),
                    weights=_normalize_rows(w_val),
                    labels=labels_val,
                    cell_ids=cell_ids_val,
                ),
                "test": SplitResult(
                    split="test",
                    x_true=x_test,
                    x_recon=recon_test.astype(np.float32),
                    logvar=make_unit_logvar(x_test),
                    latent=w_test.astype(np.float32),
                    weights=_normalize_rows(w_test),
                    labels=labels_test,
                    cell_ids=cell_ids_test,
                ),
            },
            training_history=None,
        )


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.clip(x.sum(axis=1, keepdims=True), eps, None)
    return (x / denom).astype(np.float32)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MarkerPreprocessor:
    transform: str = "none"
    arcsinh_cofactor: float = 5.0
    normalization: str = "zscore"
    clip_min: float | None = None
    clip_max: float | None = None

    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def _apply_transform(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=np.float32)
        if self.transform == "arcsinh":
            out = np.arcsinh(out / float(self.arcsinh_cofactor))
        elif self.transform != "none":
            raise ValueError(f"Unsupported transform: {self.transform}")
        return out

    def fit(self, x: np.ndarray) -> "MarkerPreprocessor":
        transformed = self._apply_transform(x)
        if self.normalization == "zscore":
            center = transformed.mean(axis=0)
            scale = transformed.std(axis=0)
        elif self.normalization == "robust_zscore":
            center = np.median(transformed, axis=0)
            q1 = np.percentile(transformed, 25, axis=0)
            q3 = np.percentile(transformed, 75, axis=0)
            scale = q3 - q1
        elif self.normalization == "none":
            center = np.zeros(transformed.shape[1], dtype=np.float32)
            scale = np.ones(transformed.shape[1], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported normalization: {self.normalization}")

        scale = np.where(scale <= 1e-8, 1.0, scale)
        self.center_ = center.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        return self

    def transform_array(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Preprocessor must be fit before transform.")
        out = self._apply_transform(x)
        out = (out - self.center_) / self.scale_
        if self.clip_min is not None or self.clip_max is not None:
            out = np.clip(out, self.clip_min, self.clip_max)
        return out.astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform_array(x)

    def state_dict(self) -> dict:
        return {
            "transform": self.transform,
            "arcsinh_cofactor": self.arcsinh_cofactor,
            "normalization": self.normalization,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "center": None if self.center_ is None else self.center_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "MarkerPreprocessor":
        pre = cls(
            transform=state["transform"],
            arcsinh_cofactor=state["arcsinh_cofactor"],
            normalization=state["normalization"],
            clip_min=state["clip_min"],
            clip_max=state["clip_max"],
        )
        if state.get("center") is not None:
            pre.center_ = np.asarray(state["center"], dtype=np.float32)
            pre.scale_ = np.asarray(state["scale"], dtype=np.float32)
        return pre

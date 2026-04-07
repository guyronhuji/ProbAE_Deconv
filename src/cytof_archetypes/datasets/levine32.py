from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    x: np.ndarray
    cell_ids: np.ndarray
    labels: np.ndarray | None
    frame: pd.DataFrame


@dataclass
class Levine32Bundle:
    markers: list[str]
    label_column: str | None
    cell_id_column: str
    train: SplitData
    val: SplitData
    test: SplitData
    raw_frame: pd.DataFrame


def _load_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _load_h5ad(path: Path, label_column: str, cell_id_column: str) -> pd.DataFrame:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "Reading .h5ad files requires optional dependency 'anndata'."
        ) from exc

    adata = ad.read_h5ad(path)
    x_df = adata.to_df()
    meta_df = adata.obs.copy()
    meta_df[cell_id_column] = meta_df.index.astype(str)
    if label_column not in meta_df.columns:
        meta_df[label_column] = np.nan
    frame = x_df.join(meta_df[[cell_id_column, label_column]], how="left")
    return frame.reset_index(drop=True)


def _detect_markers(
    frame: pd.DataFrame,
    marker_columns: list[str] | None,
    label_column: str | None,
    cell_id_column: str,
) -> list[str]:
    if marker_columns:
        missing = [col for col in marker_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Marker columns missing from dataset: {missing}")
        return marker_columns

    excluded = {cell_id_column}
    if label_column is not None:
        excluded.add(label_column)
    numeric_cols = [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if not numeric_cols:
        raise ValueError("Could not infer marker columns from numeric columns.")
    return numeric_cols


def _build_split(
    frame: pd.DataFrame,
    markers: list[str],
    label_column: str | None,
    cell_id_column: str,
    indices: np.ndarray,
) -> SplitData:
    split_frame = frame.iloc[indices].reset_index(drop=True)
    labels = None
    if label_column is not None and label_column in split_frame.columns:
        raw_labels = split_frame[label_column]
        labels = raw_labels.astype(str).to_numpy()
        if np.all(pd.isna(raw_labels)):
            labels = None
    return SplitData(
        x=split_frame[markers].to_numpy(dtype=np.float32),
        cell_ids=split_frame[cell_id_column].astype(str).to_numpy(),
        labels=labels,
        frame=split_frame,
    )


def _safe_stratify(labels: np.ndarray | None) -> np.ndarray | None:
    if labels is None:
        return None
    series = pd.Series(labels)
    counts = series.value_counts(dropna=False)
    if counts.min() < 2:
        return None
    return labels


def load_levine32_bundle(
    input_path: str | Path,
    marker_columns: list[str] | None = None,
    label_column: str | None = "label",
    cell_id_column: str = "cell_id",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Levine32Bundle:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.suffix.lower() == ".h5ad":
        frame = _load_h5ad(path, label_column or "label", cell_id_column)
    else:
        frame = _load_tabular(path)

    if cell_id_column not in frame.columns:
        frame[cell_id_column] = [f"cell_{idx}" for idx in range(len(frame))]

    effective_label_col = label_column or ("label" if path.suffix.lower() == ".h5ad" else None)
    markers = _detect_markers(frame, marker_columns, effective_label_col, cell_id_column)
    if len(markers) == 0:
        raise ValueError("No marker columns available after validation.")

    indices = np.arange(len(frame))
    labels_for_split = None
    if label_column is not None and label_column in frame.columns:
        labels_for_split = frame[label_column].astype(str).to_numpy()

    stratify_full = _safe_stratify(labels_for_split)
    if test_fraction > 0:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=seed,
            stratify=stratify_full,
        )
    else:
        train_val_idx, test_idx = indices, np.array([], dtype=int)

    val_ratio = 0.0
    if (1.0 - test_fraction) > 0:
        val_ratio = val_fraction / (1.0 - test_fraction)

    if val_ratio > 0:
        train_labels = labels_for_split[train_val_idx] if labels_for_split is not None else None
        stratify_train = _safe_stratify(train_labels)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio,
            random_state=seed,
            stratify=stratify_train,
        )
    else:
        train_idx, val_idx = train_val_idx, np.array([], dtype=int)

    train_split = _build_split(frame, markers, label_column, cell_id_column, train_idx)
    val_split = _build_split(frame, markers, label_column, cell_id_column, val_idx)
    test_split = _build_split(frame, markers, label_column, cell_id_column, test_idx)
    return Levine32Bundle(
        markers=markers,
        label_column=label_column,
        cell_id_column=cell_id_column,
        train=train_split,
        val=val_split,
        test=test_split,
        raw_frame=frame,
    )

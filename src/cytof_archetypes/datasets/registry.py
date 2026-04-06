from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cytof_archetypes.datasets.levine32 import SplitData, load_levine32_bundle


@dataclass
class DatasetBundle:
    name: str
    markers: list[str]
    label_column: str | None
    cell_id_column: str
    train: SplitData
    val: SplitData
    test: SplitData
    raw_frame: pd.DataFrame


def _to_dataset_bundle(name: str, source_bundle: Any) -> DatasetBundle:
    return DatasetBundle(
        name=name,
        markers=list(source_bundle.markers),
        label_column=source_bundle.label_column,
        cell_id_column=source_bundle.cell_id_column,
        train=source_bundle.train,
        val=source_bundle.val,
        test=source_bundle.test,
        raw_frame=source_bundle.raw_frame,
    )


def load_dataset_bundle(dataset_cfg: dict[str, Any], seed: int) -> DatasetBundle:
    name = str(dataset_cfg.get("name", "levine32")).lower()

    if name == "levine32":
        bundle = load_levine32_bundle(
            input_path=dataset_cfg["input_path"],
            marker_columns=dataset_cfg.get("marker_columns"),
            label_column=dataset_cfg.get("label_column"),
            cell_id_column=dataset_cfg.get("cell_id_column", "cell_id"),
            val_fraction=float(dataset_cfg.get("val_fraction", 0.15)),
            test_fraction=float(dataset_cfg.get("test_fraction", 0.15)),
            seed=seed,
        )
        return _to_dataset_bundle(name=name, source_bundle=bundle)

    # Secondary dataset support: any tabular/h5ad CyTOF-style file can be loaded
    # with the same interface by declaring its name and paths in config.
    if name in {"secondary", "custom_cytof", "cytof_secondary"}:
        input_path = Path(dataset_cfg["input_path"])
        if not input_path.exists():
            raise FileNotFoundError(f"Secondary dataset path does not exist: {input_path}")
        bundle = load_levine32_bundle(
            input_path=input_path,
            marker_columns=dataset_cfg.get("marker_columns"),
            label_column=dataset_cfg.get("label_column"),
            cell_id_column=dataset_cfg.get("cell_id_column", "cell_id"),
            val_fraction=float(dataset_cfg.get("val_fraction", 0.15)),
            test_fraction=float(dataset_cfg.get("test_fraction", 0.15)),
            seed=seed,
        )
        return _to_dataset_bundle(name=name, source_bundle=bundle)

    raise ValueError(
        f"Unsupported dataset '{name}'. Supported names: levine32, secondary/custom_cytof/cytof_secondary"
    )


def split_manifest_frame(bundle: DatasetBundle) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for split_name, split in (("train", bundle.train), ("val", bundle.val), ("test", bundle.test)):
        labels = split.labels if split.labels is not None else np.array([""] * len(split.cell_ids), dtype=object)
        for cell_id, label in zip(split.cell_ids, labels):
            rows.append({"cell_id": str(cell_id), "label": str(label), "split": split_name})
    return pd.DataFrame(rows)

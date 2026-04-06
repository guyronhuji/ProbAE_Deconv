#!/usr/bin/env python3
"""Download and prepare Levine32 into a local .h5ad file.

This script uses PyCytoData to fetch Levine32, then writes an .h5ad file
compatible with this repository's loaders.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Levine32 dataset as .h5ad")
    p.add_argument("--output", default="data/levine32_processed.h5ad", help="Output .h5ad path")
    p.add_argument("--label-column", default="label", help="Label column name in .obs")
    p.add_argument("--cell-id-column", default="cell_id", help="Cell ID column name in .obs")
    p.add_argument("--log-path", default="data/levine32_preprocessing_log.json", help="Output JSON metadata log")
    p.add_argument("--force-download", action="store_true", help="Force PyCytoData re-download")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is not None:
        p.add_argument(
            "--use-lineage-only",
            action=bool_action,
            default=True,
            help="Restrict to lineage channels when available (default: true)",
        )
        p.add_argument(
            "--install-missing",
            action=bool_action,
            default=True,
            help="pip install missing dependencies if needed (default: true)",
        )
    else:
        p.add_argument(
            "--use-lineage-only",
            action="store_true",
            default=True,
            help="Restrict to lineage channels when available",
        )
        p.add_argument("--no-use-lineage-only", dest="use_lineage_only", action="store_false")
        p.add_argument(
            "--install-missing",
            action="store_true",
            default=True,
            help="pip install missing dependencies if needed",
        )
        p.add_argument("--no-install-missing", dest="install_missing", action="store_false")
    return p.parse_args()


def _ensure_import(module: str, pip_pkg: str, install_missing: bool) -> Any:
    try:
        return __import__(module, fromlist=["*"])
    except ImportError:
        if not install_missing:
            raise
        print(f"Installing missing dependency: {pip_pkg}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_pkg])
        return __import__(module, fromlist=["*"])


def _clean_channel_name(name: str) -> str:
    return re.sub(r"\([^)]*\)Di?$", "", str(name)).strip()


def _make_unique(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        c = counts.get(name, 0)
        if c == 0:
            out.append(name)
        else:
            out.append(f"{name}_{c+1}")
        counts[name] = c + 1
    return out


def main() -> None:
    args = _parse_args()

    out_path = Path(args.output).expanduser().resolve()
    log_path = Path(args.log_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        print(f"Output exists, skipping (use --overwrite to replace): {out_path}", flush=True)
        return

    # NumPy 2 compatibility patch for fcsparser usage inside PyCytoData.
    if not hasattr(np.ndarray, "newbyteorder"):
        def _ndarray_newbyteorder(self, order="S"):
            return self.view(self.dtype.newbyteorder(order))
        np.ndarray.newbyteorder = _ndarray_newbyteorder  # type: ignore[attr-defined]

    _ensure_import("anndata", "anndata", args.install_missing)
    pycyto = _ensure_import("PyCytoData", "PyCytoData", args.install_missing)
    ad = _ensure_import("anndata", "anndata", args.install_missing)

    print("Downloading/loading Levine32 via PyCytoData ...", flush=True)
    try:
        exprs = pycyto.DataLoader.load_dataset(dataset="levine32", force_download=bool(args.force_download))
    except Exception as exc:  # pragma: no cover - runtime/network dependent
        raise RuntimeError(
            "Failed to load Levine32 via PyCytoData. "
            "Check internet access in the pod and package compatibility."
        ) from exc

    X = exprs.expression_matrix
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    X = np.asarray(X, dtype=np.float32)

    raw_channels: list[str]
    if hasattr(exprs, "channels") and exprs.channels is not None:
        raw_channels = [str(c) for c in exprs.channels]
    else:
        raw_channels = [f"marker_{i}" for i in range(X.shape[1])]

    if args.use_lineage_only and hasattr(exprs, "lineage_channels") and exprs.lineage_channels is not None:
        lineage = [str(c) for c in exprs.lineage_channels]
        lineage_set = set(lineage)
        keep_idx = [i for i, c in enumerate(raw_channels) if c in lineage_set]
        if keep_idx:
            X = X[:, keep_idx]
            raw_channels = [raw_channels[i] for i in keep_idx]

    marker_names = _make_unique([_clean_channel_name(c) for c in raw_channels])

    labels: np.ndarray
    if hasattr(exprs, "cell_types") and exprs.cell_types is not None:
        labels_series = pd.Series(exprs.cell_types, dtype="object").astype(str)
        labels_series = labels_series.where(~labels_series.str.lower().isin({"unassigned", "nan", "none", ""}), other=np.nan)
        labels = labels_series.to_numpy()
    else:
        labels = np.full(shape=(X.shape[0],), fill_value=np.nan, dtype=object)

    cell_ids = np.asarray([f"cell_{i}" for i in range(X.shape[0])], dtype=object)
    obs = pd.DataFrame(
        {
            args.cell_id_column: cell_ids,
            args.label_column: labels,
        },
        index=cell_ids,
    )

    adata = ad.AnnData(X=X.astype(np.float32), obs=obs)
    adata.var_names = marker_names
    adata.write_h5ad(out_path)

    summary = {
        "dataset": "levine32",
        "source": "PyCytoData.DataLoader.load_dataset(dataset='levine32')",
        "output_path": str(out_path),
        "n_cells": int(X.shape[0]),
        "n_markers": int(X.shape[1]),
        "label_column": args.label_column,
        "cell_id_column": args.cell_id_column,
        "force_download": bool(args.force_download),
        "use_lineage_only": bool(args.use_lineage_only),
        "marker_names_sample": marker_names[:10],
    }
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote dataset: {out_path}", flush=True)
    print(f"Wrote log: {log_path}", flush=True)


if __name__ == "__main__":
    main()

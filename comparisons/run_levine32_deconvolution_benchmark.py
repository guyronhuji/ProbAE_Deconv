#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy optional fallback
    linear_sum_assignment = None

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Ensure matplotlib cache is writable before importing plot modules.
os.environ.setdefault("MPLCONFIGDIR", str((REPO_ROOT / "comparisons" / ".mplconfig").resolve()))
os.environ.setdefault("NUMBA_CACHE_DIR", str((REPO_ROOT / "comparisons" / ".numba_cache").resolve()))

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import Normalize

import cytof_archetypes.experiments.common as exp_common
from cytof_archetypes.evaluation.deconvolution_metrics import (
    class_purity_of_dominant,
    per_cell_weight_entropy,
)
from cytof_archetypes.evaluation.embeddings import umap_fit_transform_large
from cytof_archetypes.evaluation.metrics import reconstruction_metrics_per_cell, representation_structure_metrics
from cytof_archetypes.evaluation.reporting import save_environment_log, write_json, write_markdown
from cytof_archetypes.experiments.common import BenchmarkRun, prepare_data
from cytof_archetypes.experiments.run_fit_vs_complexity import run_fit_vs_complexity
from cytof_archetypes.baselines.base import SplitResult
from cytof_archetypes.utils import set_seed


DEFAULT_METHODS = [
    "nmf",
    "classical_archetypes",
    "deterministic_archetypal_ae",
    "probabilistic_archetypal_ae",
]


def _write_compact_method_artifacts(result: Any, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if result.components_mean is not None:
        pd.DataFrame(
            result.components_mean,
            index=[f"component_{i}" for i in range(result.components_mean.shape[0])],
            columns=result.marker_names,
        ).to_csv(out / "component_means.csv")
    if result.components_var is not None:
        pd.DataFrame(
            result.components_var,
            index=[f"component_{i}" for i in range(result.components_var.shape[0])],
            columns=result.marker_names,
        ).to_csv(out / "component_vars.csv")
    if result.training_history is not None:
        result.training_history.to_csv(out / "training_history.csv", index=False)

    # Save compact per-split metrics and test weights only.
    metrics_rows: list[dict[str, float | str | int]] = []
    for split_name, split in result.split_results.items():
        nll, mse = reconstruction_metrics_per_cell(split.x_true, split.x_recon, split.logvar)
        metrics_rows.append(
            {
                "split": split_name,
                "n_cells": int(len(split.x_true)),
                "mse_mean": float(np.mean(mse)) if len(mse) else float("nan"),
                "nll_mean": float(np.mean(nll)) if len(nll) else float("nan"),
            }
        )
    pd.DataFrame(metrics_rows).to_csv(out / "split_metrics.csv", index=False)

    test = result.split_results.get("test")
    if test is not None:
        test_dir = out / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        payload = pd.DataFrame({"cell_id": test.cell_ids.astype(str)})
        if test.labels is not None:
            payload["label"] = test.labels.astype(str)
        if test.weights is not None:
            for idx in range(test.weights.shape[1]):
                payload[f"w_{idx}"] = test.weights[:, idx]
        payload.to_csv(test_dir / "weights.csv", index=False)


@dataclass
class PreparedComparisonDataset:
    labeled_h5ad_path: Path
    metadata_path: Path
    split_manifest_path: Path
    processed_matrix_path: Path
    representations_dir: Path
    n_cells_raw: int
    n_cells_labeled: int
    n_unlabeled_removed: int
    markers: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Levine32 deconvolution benchmark in comparisons/.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML/JSON config path. When provided, keys override CLI defaults.",
    )
    parser.add_argument(
        "--input-h5ad",
        default=str(REPO_ROOT / "data" / "levine32_processed.h5ad"),
        help="Source Levine32 .h5ad path.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "comparisons" / "outputs"),
        help="Benchmark output root directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Fixed split seed.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[13, 17, 23],
        help="Method random seeds.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[4, 6, 8, 10],
        help="K sweep values.",
    )
    parser.add_argument(
        "--method-order",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Ordered methods to run.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation split fraction.")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Test split fraction.")
    parser.add_argument("--max-epochs", type=int, default=12, help="Neural-model max epochs.")
    parser.add_argument("--patience", type=int, default=4, help="Neural-model early stopping patience.")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Neural hidden layer dimensions.",
    )
    parser.add_argument("--batch-size", type=int, default=8192, help="Neural batch size.")
    parser.add_argument("--nmf-max-iter", type=int, default=250, help="NMF max iterations.")
    parser.add_argument("--classical-n-iters", type=int, default=20, help="Classical AA outer iterations.")
    parser.add_argument("--classical-pg-steps", type=int, default=50, help="Classical AA projected-gradient steps.")
    parser.add_argument("--classical-lr", type=float, default=0.1, help="Classical AA projected-gradient learning rate.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for neural models (auto, cpu, mps, cuda).",
    )
    parser.add_argument("--cpu-workers", type=int, default=8, help="CPU multiprocessing workers.")
    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is not None:
        parser.add_argument(
            "--show-progress",
            action=bool_action,
            default=True,
            help="Show progress bars for core benchmark runs.",
        )
    else:
        parser.add_argument("--show-progress", action="store_true", default=True, help="Show progress bars.")
        parser.add_argument("--no-show-progress", dest="show_progress", action="store_false")
    if bool_action is not None:
        parser.add_argument(
            "--show-training-progress",
            action=bool_action,
            default=True,
            help="Show periodic NN training progress logs (det/prob archetypal AEs).",
        )
        parser.add_argument(
            "--training-progress-leave",
            action=bool_action,
            default=False,
            help="Keep completed training progress bars in output.",
        )
    else:
        parser.add_argument(
            "--show-training-progress",
            action="store_true",
            default=True,
            help="Show periodic NN training progress logs (det/prob archetypal AEs).",
        )
        parser.add_argument("--no-show-training-progress", dest="show_training_progress", action="store_false")
        parser.add_argument(
            "--training-progress-leave",
            action="store_true",
            default=False,
            help="Keep completed training progress bars in output.",
        )
        parser.add_argument("--no-training-progress-leave", dest="training_progress_leave", action="store_false")
    parser.add_argument(
        "--training-progress-level",
        default="epoch",
        choices=["epoch", "batch"],
        help="Compatibility flag (progress bars removed; kept for config compatibility).",
    )
    parser.add_argument(
        "--training-progress-log-every-epochs",
        type=int,
        default=500,
        help="Print NN training status every N epochs.",
    )
    parser.add_argument("--run-benchmark", action="store_true", help="Run training benchmark stage.")
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip training and only regenerate tables/figures from existing outputs.",
    )
    return parser.parse_args()


def _load_source_preprocessing_log(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _load_runtime_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config path does not exist: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML config files. Install `pyyaml` or use JSON config.")
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _apply_runtime_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        args.method_cfg_from_config = {}
        return args
    cfg_path = Path(args.config).expanduser().resolve()
    cfg_dir = cfg_path.parent
    cfg = _load_runtime_config(cfg_path)
    # 1) Flat-key compatibility
    for key, value in cfg.items():
        attr = str(key).strip().replace("-", "_")
        if hasattr(args, attr):
            setattr(args, attr, value)

    # 2) Nested suite-style compatibility (matches configs/*.yaml patterns)
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset", {}), dict) else {}
    sweeps_cfg = cfg.get("sweeps", {}) if isinstance(cfg.get("sweeps", {}), dict) else {}
    methods_cfg = cfg.get("methods", {}) if isinstance(cfg.get("methods", {}), dict) else {}

    if "output_dir" in cfg and hasattr(args, "output_root"):
        out_raw = str(cfg["output_dir"])
        out_path = Path(out_raw)
        if not out_path.is_absolute():
            out_path = (REPO_ROOT / out_path).resolve()
        args.output_root = str(out_path)
    if "show_progress" in cfg:
        args.show_progress = bool(cfg["show_progress"])
    if "show_training_progress" in cfg:
        args.show_training_progress = bool(cfg["show_training_progress"])
    if "training_progress_level" in cfg:
        args.training_progress_level = str(cfg["training_progress_level"]).lower()
    if "training_progress_leave" in cfg:
        args.training_progress_leave = bool(cfg["training_progress_leave"])
    if "training_progress_log_every_epochs" in cfg:
        args.training_progress_log_every_epochs = int(cfg["training_progress_log_every_epochs"])
    if "cpu_multiprocessing_workers" in cfg:
        args.cpu_workers = int(cfg["cpu_multiprocessing_workers"])
    if "seed" in cfg:
        args.seed = int(cfg["seed"])
    if "seeds" in cfg and isinstance(cfg["seeds"], list):
        args.seeds = [int(v) for v in cfg["seeds"]]
    if "method_order" in cfg and isinstance(cfg["method_order"], list):
        args.method_order = [str(v) for v in cfg["method_order"]]

    if "input_path" in dataset_cfg:
        raw = Path(str(dataset_cfg["input_path"])).expanduser()
        if raw.is_absolute():
            resolved = raw
        else:
            repo_candidate = (REPO_ROOT / raw)
            cfg_candidate = (cfg_dir / raw)
            if repo_candidate.exists():
                resolved = repo_candidate
            elif cfg_candidate.exists():
                resolved = cfg_candidate
            else:
                # Default to repo-relative semantics used by existing project configs.
                resolved = repo_candidate
        args.input_h5ad = str(resolved.resolve())
    if "val_fraction" in dataset_cfg:
        args.val_fraction = float(dataset_cfg["val_fraction"])
    if "test_fraction" in dataset_cfg:
        args.test_fraction = float(dataset_cfg["test_fraction"])

    if "k_values" in sweeps_cfg and isinstance(sweeps_cfg["k_values"], list):
        args.k_values = [int(v) for v in sweeps_cfg["k_values"]]

    # Method-specific extraction for CLI-level fields.
    nmf_cfg = methods_cfg.get("nmf", {}) if isinstance(methods_cfg.get("nmf", {}), dict) else {}
    caa_cfg = (
        methods_cfg.get("classical_archetypes", {})
        if isinstance(methods_cfg.get("classical_archetypes", {}), dict)
        else {}
    )
    det_cfg = (
        methods_cfg.get("deterministic_archetypal_ae", {})
        if isinstance(methods_cfg.get("deterministic_archetypal_ae", {}), dict)
        else {}
    )
    prob_cfg = (
        methods_cfg.get("probabilistic_archetypal_ae", {})
        if isinstance(methods_cfg.get("probabilistic_archetypal_ae", {}), dict)
        else {}
    )

    if "max_iter" in nmf_cfg:
        args.nmf_max_iter = int(nmf_cfg["max_iter"])
    if "n_iters" in caa_cfg:
        args.classical_n_iters = int(caa_cfg["n_iters"])
    if "pg_steps" in caa_cfg:
        args.classical_pg_steps = int(caa_cfg["pg_steps"])
    if "lr" in caa_cfg:
        args.classical_lr = float(caa_cfg["lr"])

    base_neural_cfg = prob_cfg if prob_cfg else det_cfg
    if "device" in base_neural_cfg:
        args.device = str(base_neural_cfg["device"])
    if "hidden_dims" in base_neural_cfg and isinstance(base_neural_cfg["hidden_dims"], list):
        args.hidden_dims = [int(v) for v in base_neural_cfg["hidden_dims"]]
    if "batch_size" in base_neural_cfg:
        args.batch_size = int(base_neural_cfg["batch_size"])
    if "max_epochs" in base_neural_cfg:
        args.max_epochs = int(base_neural_cfg["max_epochs"])
    if "patience" in base_neural_cfg:
        args.patience = int(base_neural_cfg["patience"])

    # Preserve full method configs for direct merge in runner.
    args.method_cfg_from_config = methods_cfg
    return args


def _extract_dense_matrix(adata: ad.AnnData) -> np.ndarray:
    matrix = adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _valid_label_mask(labels: pd.Series) -> np.ndarray:
    label_obj = labels.astype("object")
    as_str = label_obj.astype(str).str.strip()
    lowered = as_str.str.lower()
    invalid = lowered.isin({"", "nan", "none", "unassigned", "unknown", "unlabeled"})
    return (~invalid & label_obj.notna()).to_numpy()


def _find_optional_sample_col(obs: pd.DataFrame) -> str | None:
    candidates = ["sample_id", "sample", "donor", "patient", "batch", "file_id", "fcs_file"]
    for col in candidates:
        if col in obs.columns:
            return col
    return None


def prepare_labeled_levine32_dataset(
    source_h5ad_path: Path,
    output_root: Path,
    split_seed: int,
    val_fraction: float,
    test_fraction: float,
) -> PreparedComparisonDataset:
    datasets_dir = output_root / "datasets"
    reports_dir = output_root / "reports"
    repr_dir = datasets_dir / "representations"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    repr_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(source_h5ad_path)
    obs = adata.obs.copy()
    if "label" not in obs.columns:
        raise ValueError("Source Levine32 dataset is missing required 'label' column.")

    mask = _valid_label_mask(obs["label"])
    adata_labeled = adata[mask].copy()
    obs_labeled = adata_labeled.obs.copy()
    cell_ids = obs_labeled.index.astype(str).to_numpy()

    marker_names = [str(name) for name in adata_labeled.var_names.tolist()]
    x_labeled = _extract_dense_matrix(adata_labeled)

    sample_col = _find_optional_sample_col(obs_labeled)
    metadata = pd.DataFrame({"cell_id": cell_ids, "label": obs_labeled["label"].astype(str).to_numpy()})
    if sample_col is not None:
        metadata["sample_id"] = obs_labeled[sample_col].astype(str).to_numpy()

    labeled_obs = pd.DataFrame(index=cell_ids)
    labeled_obs["cell_id"] = cell_ids
    labeled_obs["label"] = metadata["label"].to_numpy()
    if sample_col is not None:
        labeled_obs["sample_id"] = metadata["sample_id"].to_numpy()

    adata_clean = ad.AnnData(X=x_labeled, obs=labeled_obs)
    adata_clean.var_names = marker_names

    labeled_h5ad_path = datasets_dir / "levine32_labeled_processed.h5ad"
    metadata_path = datasets_dir / "levine32_labeled_metadata.csv"
    matrix_path = datasets_dir / "levine32_labeled_matrix.npz"
    split_manifest_path = reports_dir / "split_manifest.csv"
    preprocess_log_path = reports_dir / "preprocessing_details.json"

    adata_clean.write_h5ad(labeled_h5ad_path)
    metadata.to_csv(metadata_path, index=False)
    np.savez_compressed(matrix_path, x=x_labeled, marker_names=np.asarray(marker_names, dtype=object))

    indices = np.arange(len(metadata))
    labels = metadata["label"].to_numpy()
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=split_seed,
        stratify=labels,
    )
    val_ratio = val_fraction / (1.0 - test_fraction)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        random_state=split_seed,
        stratify=labels[train_val_idx],
    )
    split_map = {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }

    manifest_rows: list[dict[str, str]] = []
    for split_name, idx in split_map.items():
        split_meta = metadata.iloc[idx]
        for row in split_meta.itertuples(index=False):
            payload = {"cell_id": str(row.cell_id), "label": str(row.label), "split": split_name}
            if sample_col is not None:
                payload["sample_id"] = str(getattr(row, "sample_id"))
            manifest_rows.append(payload)
    split_manifest = pd.DataFrame(manifest_rows)
    split_manifest.to_csv(split_manifest_path, index=False)

    x_train = x_labeled[split_map["train"]]
    x_val = x_labeled[split_map["val"]]
    x_test = x_labeled[split_map["test"]]

    nmf_shift = float(max(0.0, -(float(np.min(x_train))) + 1e-6))
    np.savez_compressed(
        repr_dir / "shared_zscore_representation.npz",
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
    )
    np.savez_compressed(
        repr_dir / "nmf_nonnegative_representation.npz",
        x_train=x_train + nmf_shift,
        x_val=x_val + nmf_shift,
        x_test=x_test + nmf_shift,
        shift=nmf_shift,
    )

    source_log = _load_source_preprocessing_log(REPO_ROOT / "data" / "levine32_preprocessing_log.json")
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_h5ad_path": str(source_h5ad_path.resolve()),
        "source_preprocessing_log_path": str((REPO_ROOT / "data" / "levine32_preprocessing_log.json").resolve()),
        "source_preprocessing_log": source_log,
        "notes": [
            "Source data appears already transformed with arcsinh(cofactor=5) and z-score normalization.",
            "No extra clipping was applied in this benchmark pipeline.",
            "Unlabeled cells were removed before split creation.",
            "NMF nonnegative representation is shared_zscore + global train-min shift.",
        ],
        "split_seed": split_seed,
        "split_val_fraction": val_fraction,
        "split_test_fraction": test_fraction,
        "n_cells_raw": int(adata.n_obs),
        "n_cells_labeled": int(adata_labeled.n_obs),
        "n_unlabeled_removed": int(adata.n_obs - adata_labeled.n_obs),
        "n_markers": int(adata_labeled.n_vars),
        "marker_names": marker_names,
        "label_counts": metadata["label"].value_counts().to_dict(),
        "optional_sample_id_column": sample_col,
        "representations": {
            "shared_zscore": str((repr_dir / "shared_zscore_representation.npz").resolve()),
            "nmf_nonnegative": str((repr_dir / "nmf_nonnegative_representation.npz").resolve()),
            "nmf_shift": nmf_shift,
        },
    }
    write_json(preprocess_log_path, payload)

    return PreparedComparisonDataset(
        labeled_h5ad_path=labeled_h5ad_path,
        metadata_path=metadata_path,
        split_manifest_path=split_manifest_path,
        processed_matrix_path=matrix_path,
        representations_dir=repr_dir,
        n_cells_raw=int(adata.n_obs),
        n_cells_labeled=int(adata_labeled.n_obs),
        n_unlabeled_removed=int(adata.n_obs - adata_labeled.n_obs),
        markers=marker_names,
    )


def run_benchmark_core(
    dataset_bundle: PreparedComparisonDataset,
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[list[BenchmarkRun], pd.DataFrame]:
    # Replace large artifact dumps with compact outputs (weights/components/metrics),
    # otherwise Levine32 train/val/test reconstruction CSVs dominate runtime and disk.
    exp_common.write_method_artifacts = _write_compact_method_artifacts

    dataset_cfg = {
        "name": "levine32",
        "input_path": str(dataset_bundle.labeled_h5ad_path),
        "marker_columns": None,
        "label_column": "label",
        "cell_id_column": "cell_id",
        "val_fraction": float(args.val_fraction),
        "test_fraction": float(args.test_fraction),
    }
    preprocessing_cfg = {
        "transform": "none",
        "arcsinh_cofactor": 5.0,
        "normalization": "none",
        "clip_min": None,
        "clip_max": None,
    }
    prepared = prepare_data(dataset_cfg=dataset_cfg, preprocessing_cfg=preprocessing_cfg, seed=int(args.seed))

    det_defaults = {
        "device": str(args.device),
        "hidden_dims": [int(v) for v in args.hidden_dims],
        "dropout": 0.1,
        "batch_size": int(args.batch_size),
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "grad_clip": 1.0,
        "entropy_reg_weight": 1.0e-3,
        "diversity_reg_weight": 1.0e-3,
        "show_training_progress": bool(args.show_training_progress),
        "training_progress_level": str(args.training_progress_level),
        "training_progress_leave": bool(args.training_progress_leave),
        "training_progress_log_every_epochs": int(args.training_progress_log_every_epochs),
    }
    prob_defaults = {
        "device": str(args.device),
        "hidden_dims": [int(v) for v in args.hidden_dims],
        "dropout": 0.1,
        "batch_size": int(args.batch_size),
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "grad_clip": 1.0,
        "entropy_reg_weight": 1.0e-3,
        "diversity_reg_weight": 1.0e-3,
        "variance_reg_weight": 1.0e-5,
        "show_training_progress": bool(args.show_training_progress),
        "training_progress_level": str(args.training_progress_level),
        "training_progress_leave": bool(args.training_progress_leave),
        "training_progress_log_every_epochs": int(args.training_progress_log_every_epochs),
    }

    # Merge in nested config method overrides when provided.
    method_cfg_overrides = getattr(args, "method_cfg_from_config", {}) or {}
    nmf_cfg = {"max_iter": int(args.nmf_max_iter), **dict(method_cfg_overrides.get("nmf", {}))}
    caa_cfg = {
        "n_iters": int(args.classical_n_iters),
        "pg_steps": int(args.classical_pg_steps),
        "lr": float(args.classical_lr),
        **dict(method_cfg_overrides.get("classical_archetypes", {})),
    }
    det_cfg = {**det_defaults, **dict(method_cfg_overrides.get("deterministic_archetypal_ae", {}))}
    prob_cfg = {**prob_defaults, **dict(method_cfg_overrides.get("probabilistic_archetypal_ae", {}))}

    suite_cfg: dict[str, Any] = {
        "seeds": [int(seed) for seed in args.seeds],
        "sweeps": {"k_values": [int(k) for k in args.k_values]},
        "methods": {
            "nmf": nmf_cfg,
            "classical_archetypes": caa_cfg,
            "deterministic_archetypal_ae": det_cfg,
            "probabilistic_archetypal_ae": prob_cfg,
        },
        "method_order": [str(m) for m in getattr(args, "method_order", DEFAULT_METHODS)],
        "show_progress": bool(args.show_progress),
        "show_run_logs": True,
        "show_training_progress": bool(args.show_training_progress),
        "cpu_multiprocessing_workers": int(max(1, args.cpu_workers)),
        "cpu_parallel_methods": ["nmf", "classical_archetypes"],
        "gpu_multiprocessing_workers": 1,
        "gpu_parallel_methods": ["deterministic_archetypal_ae", "probabilistic_archetypal_ae"],
    }

    out = Path(output_root)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)

    save_environment_log(
        out / "reports" / "environment_log_core.json",
        extra={"dataset": "levine32_labeled"},
    )
    write_json(out / "reports" / "preprocessor.json", prepared.preprocessor.state_dict())

    reg = exp_common.method_registry()
    method_cfg_all = suite_cfg["methods"]
    methods = [m for m in suite_cfg["method_order"] if m in reg]
    k_values = [int(v) for v in suite_cfg["sweeps"]["k_values"]]
    seeds = [int(v) for v in suite_cfg["seeds"]]

    runs: list[BenchmarkRun] = []
    summary_rows: list[dict[str, float | str | int]] = []

    total_jobs = len(methods) * len(k_values) * len(seeds)
    progress = None
    if bool(args.show_progress) and tqdm is not None:
        progress = tqdm(total=total_jobs, desc="Core benchmark runs", unit="run")
    job_i = 0
    try:
        for method in methods:
            for dim in k_values:
                for seed in seeds:
                    job_i += 1
                    method_cfg = dict(method_cfg_all[method])
                    print(
                        f"[core-run {job_i}/{total_jobs}] START method={method} dim={dim} seed={seed}",
                        flush=True,
                    )

                    set_seed(seed)
                    result = reg[method].run(
                        x_train=prepared.train_x,
                        x_val=prepared.val_x,
                        x_test=prepared.test_x,
                        cell_ids_train=prepared.dataset.train.cell_ids,
                        labels_val=prepared.dataset.val.labels,
                        labels_test=prepared.dataset.test.labels,
                        cell_ids_val=prepared.dataset.val.cell_ids,
                        cell_ids_test=prepared.dataset.test.cell_ids,
                        marker_names=prepared.dataset.markers,
                        seed=seed,
                        representation_dim=dim,
                        config=method_cfg,
                    )

                    run_dir = out / "runs" / method / f"dim_{dim:02d}" / f"seed_{seed}"
                    _write_compact_method_artifacts(result, run_dir)

                    val_split = result.split_results["val"]
                    test_split = result.split_results["test"]
                    val_nll, val_mse = reconstruction_metrics_per_cell(val_split.x_true, val_split.x_recon, val_split.logvar)
                    test_nll, test_mse = reconstruction_metrics_per_cell(test_split.x_true, test_split.x_recon, test_split.logvar)
                    val_metrics = {
                        "val_mse": float(np.mean(val_mse)) if len(val_mse) else float("nan"),
                        "val_nll": float(np.mean(val_nll)) if len(val_nll) else float("nan"),
                    }
                    test_metrics = {
                        "test_mse": float(np.mean(test_mse)) if len(test_mse) else float("nan"),
                        "test_nll": float(np.mean(test_nll)) if len(test_nll) else float("nan"),
                    }
                    structure = representation_structure_metrics(
                        latent=test_split.latent,
                        labels=test_split.labels,
                    )

                    summary_rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "representation_dim": dim,
                            "val_mse": val_metrics["val_mse"],
                            "val_nll": val_metrics["val_nll"],
                            "test_mse": test_metrics["test_mse"],
                            "test_nll": test_metrics["test_nll"],
                            "ari": structure["ari"],
                            "nmi": structure["nmi"],
                            "knn_purity": structure["knn_purity"],
                            "silhouette": structure["silhouette"],
                            "param_count": int(result.params.get("param_count", 0)),
                        }
                    )

                    # Keep only test split in-memory to avoid huge run-object memory usage.
                    n_markers = test_split.x_true.shape[1]
                    empty = SplitResult(
                        split="empty",
                        x_true=np.zeros((0, n_markers), dtype=np.float32),
                        x_recon=np.zeros((0, n_markers), dtype=np.float32),
                        logvar=np.zeros((0, n_markers), dtype=np.float32),
                        latent=np.zeros((0, max(1, dim)), dtype=np.float32),
                        weights=None,
                        labels=None,
                        cell_ids=np.zeros((0,), dtype=object),
                    )
                    result.split_results = {"train": empty, "val": empty, "test": test_split}

                    runs.append(
                        BenchmarkRun(
                            method=method,
                            seed=seed,
                            representation_dim=dim,
                            run_dir=run_dir,
                            result=result,
                            val_metrics=val_metrics,
                            test_metrics=test_metrics,
                            structure_metrics=structure,
                        )
                    )
                    print(
                        f"[core-run {job_i}/{total_jobs}] DONE method={method} dim={dim} seed={seed} "
                        f"val_mse={val_metrics['val_mse']:.6f} test_mse={test_metrics['test_mse']:.6f}",
                        flush=True,
                    )
                    if progress is not None:
                        progress.set_postfix({"method": method, "k": dim, "seed": seed})
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "reports" / "core_run_summary.csv", index=False)
    return runs, summary_df


def _pairwise_component_similarity(a: np.ndarray, b: np.ndarray) -> float:
    sim = cosine_similarity(a, b)
    if linear_sum_assignment is not None:
        row_idx, col_idx = linear_sum_assignment(-sim)
        return float(np.mean(sim[row_idx, col_idx]))

    # Greedy fallback if scipy is unavailable.
    sim_copy = sim.copy()
    score = 0.0
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for _ in range(min(sim.shape[0], sim.shape[1])):
        best = np.unravel_index(np.argmax(sim_copy), sim_copy.shape)
        r, c = int(best[0]), int(best[1])
        if r in used_rows or c in used_cols:
            sim_copy[r, c] = -np.inf
            continue
        used_rows.add(r)
        used_cols.add(c)
        score += float(sim[r, c])
        sim_copy[r, :] = -np.inf
        sim_copy[:, c] = -np.inf
    matched = min(len(used_rows), len(used_cols))
    return float(score / matched) if matched > 0 else float("nan")


def compute_stability_table(runs: list[BenchmarkRun]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    grouped: dict[tuple[str, int], list[BenchmarkRun]] = {}
    for run in runs:
        grouped.setdefault((run.method, run.representation_dim), []).append(run)

    for (method, k), method_runs in sorted(grouped.items()):
        for run_a, run_b in combinations(sorted(method_runs, key=lambda r: r.seed), 2):
            if run_a.result.components_mean is None or run_b.result.components_mean is None:
                continue
            sim = _pairwise_component_similarity(run_a.result.components_mean, run_b.result.components_mean)
            rows.append(
                {
                    "method": method,
                    "k": int(k),
                    "seed_a": int(run_a.seed),
                    "seed_b": int(run_b.seed),
                    "component_similarity": float(sim),
                }
            )
    pairwise_df = pd.DataFrame(rows)
    if pairwise_df.empty:
        return pd.DataFrame(
            columns=["method", "k", "component_similarity_mean", "component_similarity_std", "n_seed_pairs"]
        )
    agg = pairwise_df.groupby(["method", "k"], as_index=False).agg(
        component_similarity_mean=("component_similarity", "mean"),
        component_similarity_std=("component_similarity", "std"),
        n_seed_pairs=("component_similarity", "size"),
    )
    return agg


def compute_deconvolution_summary(runs: list[BenchmarkRun], output_root: Path) -> pd.DataFrame:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str | int]] = []
    for run in runs:
        split = run.result.split_results["test"]
        weights = split.weights
        if weights is None or len(weights) == 0:
            continue
        entropy = per_cell_weight_entropy(weights)
        dominant = np.max(weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None), axis=1)
        rows.append(
            {
                "method": run.method,
                "seed": int(run.seed),
                "k": int(run.representation_dim),
                "weight_entropy_mean": float(np.mean(entropy)),
                "dominant_fraction": float(np.mean(dominant >= 0.7)),
                "mixed_fraction": float(np.mean(dominant < 0.7)),
                "class_purity_dominant_component": float(class_purity_of_dominant(weights, split.labels)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "deconvolution_quality_summary.csv", index=False)
    return df


def compute_k_selection_table(summary_df: pd.DataFrame, deconv_df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame = summary_df.rename(columns={"representation_dim": "k"}).copy()
    if not deconv_df.empty:
        frame = frame.merge(
            deconv_df[["method", "seed", "k", "weight_entropy_mean", "class_purity_dominant_component"]],
            on=["method", "seed", "k"],
            how="left",
        )
    grouped = frame.groupby(["method", "k"], as_index=False).mean(numeric_only=True)
    scored_parts: list[pd.DataFrame] = []
    for method, sub in grouped.groupby("method"):
        sub = sub.sort_values("k").copy()
        fit_score = _invert_minmax(sub["test_mse"].to_numpy())
        ari_score = _minmax(sub["ari"].to_numpy())
        nmi_score = _minmax(sub["nmi"].to_numpy())
        purity_score = _minmax(sub["class_purity_dominant_component"].to_numpy()) if "class_purity_dominant_component" in sub else np.full(len(sub), np.nan)
        entropy_score = _invert_minmax(sub["weight_entropy_mean"].to_numpy()) if "weight_entropy_mean" in sub else np.full(len(sub), np.nan)
        score_stack = np.vstack([fit_score, ari_score, nmi_score, purity_score, entropy_score])
        sub["k_selection_score"] = np.nanmean(score_stack, axis=0)
        max_score = float(np.nanmax(sub["k_selection_score"]))
        threshold = max_score - 0.02
        eligible = sub[sub["k_selection_score"] >= threshold]
        recommended = int(eligible["k"].min()) if not eligible.empty else int(sub.loc[sub["k_selection_score"].idxmax(), "k"])
        sub["recommended_k"] = recommended
        scored_parts.append(sub)
    out = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    out.to_csv(tables_dir / "k_selection_summary.csv", index=False)
    return out


def _minmax(values: np.ndarray) -> np.ndarray:
    vals = values.astype(float)
    if np.all(~np.isfinite(vals)):
        return np.full_like(vals, np.nan)
    lo = np.nanmin(vals)
    hi = np.nanmax(vals)
    if np.isclose(lo, hi):
        return np.ones_like(vals)
    return (vals - lo) / (hi - lo)


def _invert_minmax(values: np.ndarray) -> np.ndarray:
    return 1.0 - _minmax(values)


def compute_component_profiles(runs: list[BenchmarkRun], output_root: Path) -> pd.DataFrame:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str | int]] = []
    for run in runs:
        means = run.result.components_mean
        if means is None:
            continue
        vars_ = run.result.components_var
        for comp_idx in range(means.shape[0]):
            for marker_idx, marker in enumerate(run.result.marker_names):
                row: dict[str, float | str | int] = {
                    "method": run.method,
                    "k": int(run.representation_dim),
                    "seed": int(run.seed),
                    "component": int(comp_idx),
                    "marker": str(marker),
                    "mean": float(means[comp_idx, marker_idx]),
                }
                if vars_ is not None and vars_.shape == means.shape:
                    row["variance"] = float(vars_[comp_idx, marker_idx])
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "component_marker_profiles.csv", index=False)
    return df


def build_required_tables(
    runs: list[BenchmarkRun],
    summary_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    deconv_df: pd.DataFrame,
    k_df: pd.DataFrame,
    component_profiles: pd.DataFrame,
    output_root: Path,
) -> dict[str, Path]:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    stability_df = compute_stability_table(runs)
    stability_df.to_csv(tables_dir / "levine32_stability_summary.csv", index=False)

    deconv_aug = deconv_df.rename(columns={"k": "representation_dim"})
    method_summary = summary_df.merge(
        deconv_aug,
        on=["method", "seed", "representation_dim"],
        how="left",
        suffixes=("", "_deconv"),
    )
    method_summary = method_summary.merge(
        stability_df.rename(columns={"k": "representation_dim"}),
        on=["method", "representation_dim"],
        how="left",
    )
    method_summary = method_summary.rename(columns={"representation_dim": "k"})
    method_summary.to_csv(tables_dir / "levine32_method_summary.csv", index=False)

    k_sweep = method_summary.groupby(["method", "k"], as_index=False).agg(
        val_mse_mean=("val_mse", "mean"),
        val_mse_std=("val_mse", "std"),
        test_mse_mean=("test_mse", "mean"),
        test_mse_std=("test_mse", "std"),
        test_nll_mean=("test_nll", "mean"),
        ari_mean=("ari", "mean"),
        nmi_mean=("nmi", "mean"),
        knn_purity_mean=("knn_purity", "mean"),
        silhouette_mean=("silhouette", "mean"),
        weight_entropy_mean=("weight_entropy_mean", "mean"),
        dominant_purity_mean=("class_purity_dominant_component", "mean"),
        component_similarity_mean=("component_similarity_mean", "mean"),
    )
    if not k_df.empty and {"method", "k", "recommended_k"}.issubset(k_df.columns):
        k_sweep = k_sweep.merge(
            k_df[["method", "k", "recommended_k"]],
            on=["method", "k"],
            how="left",
        )
    k_sweep.to_csv(tables_dir / "levine32_k_sweep_summary.csv", index=False)

    per_class_rows: list[dict[str, float | str | int]] = []
    entropy_rows: list[dict[str, float | str | int]] = []
    for run in runs:
        split = run.result.split_results["test"]
        labels = split.labels
        if labels is None:
            continue
        nll, mse = reconstruction_metrics_per_cell(split.x_true, split.x_recon, split.logvar)
        weights = split.weights
        entropy = per_cell_weight_entropy(weights) if weights is not None else np.full((len(labels),), np.nan)
        dom = (
            np.argmax(weights, axis=1).astype(int)
            if weights is not None and len(weights)
            else np.full((len(labels),), -1, dtype=int)
        )
        per_cell = pd.DataFrame(
            {
                "label": labels.astype(str),
                "mse": mse,
                "nll": nll,
                "entropy": entropy,
                "dominant_component": dom,
            }
        )
        for row in per_cell.groupby("label", as_index=False).agg(
            n_cells=("label", "size"),
            mse=("mse", "mean"),
            nll=("nll", "mean"),
            entropy=("entropy", "mean"),
        ).itertuples(index=False):
            per_class_rows.append(
                {
                    "method": run.method,
                    "seed": int(run.seed),
                    "k": int(run.representation_dim),
                    "label": str(row.label),
                    "n_cells": int(row.n_cells),
                    "mse": float(row.mse),
                    "nll": float(row.nll),
                    "entropy": float(row.entropy),
                }
            )
        for idx, ent in enumerate(entropy):
            entropy_rows.append(
                {
                    "method": run.method,
                    "seed": int(run.seed),
                    "k": int(run.representation_dim),
                    "cell_id": str(split.cell_ids[idx]),
                    "label": str(labels[idx]),
                    "weight_entropy": float(ent),
                }
            )

    per_class_df = pd.DataFrame(
        per_class_rows,
        columns=["method", "seed", "k", "label", "n_cells", "mse", "nll", "entropy"],
    )
    per_class_df.to_csv(tables_dir / "levine32_per_class_metrics.csv", index=False)

    profiles = component_profiles.copy()
    profiles.to_csv(tables_dir / "levine32_component_marker_profiles.csv", index=False)

    entropy_df = pd.DataFrame(
        entropy_rows,
        columns=["method", "seed", "k", "cell_id", "label", "weight_entropy"],
    )
    entropy_df.to_csv(tables_dir / "levine32_weight_entropy.csv", index=False)

    return {
        "method_summary": tables_dir / "levine32_method_summary.csv",
        "k_sweep": tables_dir / "levine32_k_sweep_summary.csv",
        "per_class": tables_dir / "levine32_per_class_metrics.csv",
        "component_profiles": tables_dir / "levine32_component_marker_profiles.csv",
        "weight_entropy": tables_dir / "levine32_weight_entropy.csv",
        "stability": tables_dir / "levine32_stability_summary.csv",
    }


def _compute_embedding(x: np.ndarray, random_state: int = 42) -> np.ndarray:
    # Use PCA-only embedding for deterministic, low-memory figure generation in this environment.
    pca = PCA(n_components=min(2, x.shape[1]), random_state=random_state)
    coords = pca.fit_transform(x)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((len(coords), 1), dtype=np.float32)])
    return coords.astype(np.float32)


def _compute_marker_umap(x: np.ndarray, random_state: int = 42) -> np.ndarray:
    try:
        return umap_fit_transform_large(
            x,
            n_components=2,
            n_neighbors=25,
            min_dist=0.2,
            random_state=random_state,
        )
    except Exception:
        # Graceful fallback if UMAP backend fails in a given environment.
        return _compute_embedding(x, random_state=random_state)


def _method_display_name(method: str) -> str:
    mapping = {
        "nmf": "NMF",
        "classical_archetypes": "Classical Archetypal Analysis",
        "deterministic_archetypal_ae": "Deterministic Archetypal AE",
        "probabilistic_archetypal_ae": "Probabilistic Archetypal AE",
    }
    return mapping.get(method, method)


def _representative_runs(
    runs: list[BenchmarkRun],
    k_df: pd.DataFrame,
) -> dict[str, BenchmarkRun]:
    recommended_k: dict[str, int] = {}
    if not k_df.empty and {"method", "recommended_k"}.issubset(k_df.columns):
        rec = k_df.groupby("method", as_index=False)["recommended_k"].first()
        recommended_k = {str(row.method): int(row.recommended_k) for row in rec.itertuples(index=False)}

    out: dict[str, BenchmarkRun] = {}
    for method in DEFAULT_METHODS:
        subset = [run for run in runs if run.method == method]
        if not subset:
            continue
        target_k = recommended_k.get(method)
        if target_k is not None:
            subset_k = [run for run in subset if run.representation_dim == target_k]
            if subset_k:
                subset = subset_k
        best = min(subset, key=lambda run: (run.val_metrics.get("val_mse", float("inf")), run.seed))
        out[method] = best
    return out


def generate_figures(
    runs: list[BenchmarkRun],
    method_summary: pd.DataFrame,
    k_sweep: pd.DataFrame,
    per_class: pd.DataFrame,
    k_df: pd.DataFrame,
    dataset_bundle: PreparedComparisonDataset,
    output_root: Path,
) -> list[Path]:
    fig_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    figures: list[Path] = []
    rep_runs = _representative_runs(runs, k_df)
    if not rep_runs:
        return figures

    # Figure 1: overview.
    adata_labeled = ad.read_h5ad(dataset_bundle.labeled_h5ad_path)
    x_all = _extract_dense_matrix(adata_labeled)
    labels_all = adata_labeled.obs["label"].astype(str).to_numpy()
    emb_all = _compute_embedding(x_all, random_state=42)
    marker_df = pd.DataFrame(x_all, columns=[str(m) for m in adata_labeled.var_names])
    label_counts = pd.Series(labels_all).value_counts().sort_values(ascending=False)
    top_markers = marker_df.var(axis=0).sort_values(ascending=False).head(10).index.tolist()
    marker_summary = marker_df[top_markers].describe().loc[["mean", "std"]]

    fig1, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=False)
    uniq_labels = sorted(set(labels_all.tolist()))
    cmap = plt.cm.get_cmap("tab20", len(uniq_labels))
    for idx, label in enumerate(uniq_labels):
        mask = labels_all == label
        axes[0].scatter(emb_all[mask, 0], emb_all[mask, 1], s=2.0, alpha=0.6, color=cmap(idx), label=label)
    axes[0].set_title("UMAP/PCA of labeled Levine32 cells")
    axes[0].set_xlabel("dim_1")
    axes[0].set_ylabel("dim_2")
    if len(uniq_labels) <= 20:
        axes[0].legend(fontsize=6, frameon=False, markerscale=4)

    axes[1].bar(label_counts.index.astype(str), label_counts.to_numpy(), color="#2f6690")
    axes[1].set_title("Class counts (labeled cells only)")
    axes[1].tick_params(axis="x", rotation=75, labelsize=8)
    axes[1].set_ylabel("cells")

    im = axes[2].imshow(marker_summary.to_numpy(), aspect="auto", cmap="viridis")
    axes[2].set_title("Marker summary (mean/std, top-variance markers)")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["mean", "std"])
    axes[2].set_xticks(np.arange(len(top_markers)))
    axes[2].set_xticklabels(top_markers, rotation=75, fontsize=8)
    fig1.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig1.tight_layout()
    fig1_path = fig_dir / "figure1_levine32_benchmark_overview.png"
    fig1.savefig(fig1_path, dpi=220)
    plt.close(fig1)
    figures.append(fig1_path)

    # Figure 2: mixtures vs discrete assignments.
    base_split = next(iter(rep_runs.values())).result.split_results["test"]
    emb_test = _compute_embedding(base_split.x_true, random_state=123)
    fig2, axes2 = plt.subplots(len(DEFAULT_METHODS), 3, figsize=(18, 4.4 * len(DEFAULT_METHODS)), constrained_layout=False)
    for row_idx, method in enumerate(DEFAULT_METHODS):
        run = rep_runs.get(method)
        if run is None:
            continue
        split = run.result.split_results["test"]
        weights = split.weights
        if weights is None:
            continue
        dom = np.argmax(weights, axis=1)
        entropy = per_cell_weight_entropy(weights)

        ax_dom = axes2[row_idx, 0]
        ax_ent = axes2[row_idx, 1]
        ax_hist = axes2[row_idx, 2]

        sc_dom = ax_dom.scatter(emb_test[:, 0], emb_test[:, 1], c=dom, s=5, alpha=0.7, cmap="tab10")
        ax_dom.set_title(f"{_method_display_name(method)} | dominant component (K={run.representation_dim})")
        ax_dom.set_xlabel("dim_1")
        ax_dom.set_ylabel("dim_2")
        fig2.colorbar(sc_dom, ax=ax_dom, fraction=0.046, pad=0.04)

        sc_ent = ax_ent.scatter(
            emb_test[:, 0],
            emb_test[:, 1],
            c=entropy,
            s=5,
            alpha=0.7,
            cmap="magma",
            norm=Normalize(vmin=float(np.nanmin(entropy)), vmax=float(np.nanmax(entropy))),
        )
        ax_ent.set_title(f"{_method_display_name(method)} | weight entropy")
        ax_ent.set_xlabel("dim_1")
        ax_ent.set_ylabel("dim_2")
        fig2.colorbar(sc_ent, ax=ax_ent, fraction=0.046, pad=0.04)

        ax_hist.hist(entropy[np.isfinite(entropy)], bins=35, color="#6c9a8b", alpha=0.85)
        ax_hist.set_title(f"{_method_display_name(method)} | entropy histogram")
        ax_hist.set_xlabel("entropy")
        ax_hist.set_ylabel("cells")
    fig2.tight_layout()
    fig2_path = fig_dir / "figure2_cells_are_mixtures_not_discrete.png"
    fig2.savefig(fig2_path, dpi=220)
    plt.close(fig2)
    figures.append(fig2_path)

    # Figure 3: interpretability profiles.
    fig3, axes3 = plt.subplots(len(DEFAULT_METHODS), 2, figsize=(18, 4.8 * len(DEFAULT_METHODS)), constrained_layout=False)
    for row_idx, method in enumerate(DEFAULT_METHODS):
        run = rep_runs.get(method)
        if run is None or run.result.components_mean is None:
            continue
        means = run.result.components_mean
        markers = run.result.marker_names

        ax_hm = axes3[row_idx, 0]
        hm = ax_hm.imshow(means, aspect="auto", cmap="coolwarm")
        ax_hm.set_title(f"{_method_display_name(method)} | component x marker heatmap")
        ax_hm.set_yticks(np.arange(means.shape[0]))
        ax_hm.set_yticklabels([f"C{i}" for i in range(means.shape[0])], fontsize=7)
        ax_hm.set_xticks(np.arange(len(markers)))
        ax_hm.set_xticklabels(markers, rotation=90, fontsize=7)
        fig3.colorbar(hm, ax=ax_hm, fraction=0.046, pad=0.04)

        ax_txt = axes3[row_idx, 1]
        ax_txt.axis("off")
        lines = [f"{_method_display_name(method)} | top markers per component"]
        for comp_idx in range(means.shape[0]):
            ranked = np.argsort(-np.abs(means[comp_idx]))[:5]
            summary = ", ".join([f"{markers[i]} ({means[comp_idx, i]:.2f})" for i in ranked])
            lines.append(f"C{comp_idx}: {summary}")
        ax_txt.text(0.01, 0.98, "\n".join(lines), va="top", fontsize=9, family="monospace")
    fig3.tight_layout()
    fig3_path = fig_dir / "figure3_component_interpretability.png"
    fig3.savefig(fig3_path, dpi=220)
    plt.close(fig3)
    figures.append(fig3_path)

    # Figure 4: deterministic vs probabilistic.
    det_prob = method_summary[method_summary["method"].isin(["deterministic_archetypal_ae", "probabilistic_archetypal_ae"])]
    det_prob_k = (
        det_prob.groupby(["method", "k"], as_index=False)["test_mse"].mean()
        if not det_prob.empty
        else pd.DataFrame()
    )
    fig4, axes4 = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=False)
    if not det_prob.empty:
        det_runs = method_summary[method_summary["method"] == "deterministic_archetypal_ae"]
        prob_runs = method_summary[method_summary["method"] == "probabilistic_archetypal_ae"]
        merged = det_runs.merge(
            prob_runs,
            on=["seed", "k"],
            suffixes=("_det", "_prob"),
        )
        if not merged.empty:
            axes4[0].scatter(merged["test_mse_det"], merged["test_mse_prob"], c=merged["k"], cmap="viridis", s=45)
            lo = min(float(merged["test_mse_det"].min()), float(merged["test_mse_prob"].min()))
            hi = max(float(merged["test_mse_det"].max()), float(merged["test_mse_prob"].max()))
            axes4[0].plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
        axes4[0].set_title("Reconstruction MSE: deterministic vs probabilistic")
        axes4[0].set_xlabel("Deterministic test MSE")
        axes4[0].set_ylabel("Probabilistic test MSE")

    if not per_class.empty:
        class_cmp = per_class[per_class["method"].isin(["deterministic_archetypal_ae", "probabilistic_archetypal_ae"])]
        class_plot = class_cmp.groupby(["method", "label"], as_index=False)["mse"].mean()
        labels_order = class_plot.groupby("label")["mse"].mean().sort_values().index.tolist()
        width = 0.42
        xs = np.arange(len(labels_order))
        det_vals = [
            float(class_plot[(class_plot["method"] == "deterministic_archetypal_ae") & (class_plot["label"] == label)]["mse"].mean())
            for label in labels_order
        ]
        prob_vals = [
            float(class_plot[(class_plot["method"] == "probabilistic_archetypal_ae") & (class_plot["label"] == label)]["mse"].mean())
            for label in labels_order
        ]
        axes4[1].bar(xs - width / 2, det_vals, width=width, label="Deterministic", color="#b24a4a")
        axes4[1].bar(xs + width / 2, prob_vals, width=width, label="Probabilistic", color="#3f8f6b")
        axes4[1].set_xticks(xs)
        axes4[1].set_xticklabels(labels_order, rotation=75, fontsize=7)
        axes4[1].set_title("Per-class reconstruction MSE")
        axes4[1].set_ylabel("MSE")
        axes4[1].legend(frameon=False)

    prob_run = rep_runs.get("probabilistic_archetypal_ae")
    if prob_run is not None and prob_run.result.components_var is not None:
        var_hm = axes4[2].imshow(prob_run.result.components_var, aspect="auto", cmap="magma")
        axes4[2].set_title("Probabilistic archetype variance heatmap")
        axes4[2].set_yticks(np.arange(prob_run.result.components_var.shape[0]))
        axes4[2].set_yticklabels([f"A{i}" for i in range(prob_run.result.components_var.shape[0])], fontsize=8)
        axes4[2].set_xticks(np.arange(len(prob_run.result.marker_names)))
        axes4[2].set_xticklabels(prob_run.result.marker_names, rotation=90, fontsize=7)
        fig4.colorbar(var_hm, ax=axes4[2], fraction=0.046, pad=0.04)
    fig4.tight_layout()
    fig4_path = fig_dir / "figure4_deterministic_vs_probabilistic.png"
    fig4.savefig(fig4_path, dpi=220)
    plt.close(fig4)
    figures.append(fig4_path)

    # Figure 5: method comparison summary.
    summary_best = (
        method_summary.groupby(["method", "k"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["method", "test_mse"])
    )
    if not k_df.empty and {"method", "recommended_k"}.issubset(k_df.columns):
        rec = k_df.groupby("method", as_index=False)["recommended_k"].first()
        summary_best = summary_best.merge(rec, on="method", how="left")
        summary_best = summary_best[summary_best["k"] == summary_best["recommended_k"]]
    else:
        summary_best = summary_best.loc[summary_best.groupby("method")["test_mse"].idxmin()]

    fig5, axes5 = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=False)
    metrics = [
        ("test_mse", "Reconstruction MSE"),
        ("ari", "ARI"),
        ("nmi", "NMI"),
        ("class_purity_dominant_component", "Dominant-component purity"),
    ]
    for ax, (col, title) in zip(axes5.flatten(), metrics):
        vals = summary_best[col].to_numpy() if col in summary_best.columns else np.array([])
        methods = summary_best["method"].tolist()
        ax.bar([_method_display_name(m) for m in methods], vals, color="#3a7ca5")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig5.tight_layout()
    fig5_path = fig_dir / "figure5_method_comparison_summary.png"
    fig5.savefig(fig5_path, dpi=220)
    plt.close(fig5)
    figures.append(fig5_path)

    # Figure 6: K selection.
    fig6, axes6 = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=False)
    for method, sub in k_sweep.groupby("method"):
        sub = sub.sort_values("k")
        axes6[0, 0].plot(sub["k"], sub["test_mse_mean"], marker="o", label=_method_display_name(method))
    axes6[0, 0].set_title("Reconstruction vs K")
    axes6[0, 0].set_xlabel("K")
    axes6[0, 0].set_ylabel("Test MSE (mean)")
    axes6[0, 0].legend(fontsize=8, frameon=False)

    for method, sub in k_sweep.groupby("method"):
        sub = sub.sort_values("k")
        axes6[0, 1].plot(sub["k"], sub["ari_mean"], marker="o", label=f"{_method_display_name(method)} ARI")
        axes6[0, 1].plot(sub["k"], sub["nmi_mean"], marker="x", linestyle="--", label=f"{_method_display_name(method)} NMI")
    axes6[0, 1].set_title("ARI / NMI vs K")
    axes6[0, 1].set_xlabel("K")
    axes6[0, 1].set_ylabel("Score")
    axes6[0, 1].legend(fontsize=6, frameon=False, ncol=2)

    if not per_class.empty:
        per_class_k = per_class.groupby(["method", "k"], as_index=False)["mse"].mean()
        for method, sub in per_class_k.groupby("method"):
            sub = sub.sort_values("k")
            axes6[1, 0].plot(sub["k"], sub["mse"], marker="o", label=_method_display_name(method))
    axes6[1, 0].set_title("Per-class preservation (mean class MSE) vs K")
    axes6[1, 0].set_xlabel("K")
    axes6[1, 0].set_ylabel("Mean class MSE")
    axes6[1, 0].legend(fontsize=8, frameon=False)

    axes6[1, 1].axis("off")
    rec_lines = ["Recommended K (smallest near-optimal):"]
    if not k_df.empty and {"method", "recommended_k"}.issubset(k_df.columns):
        rec = k_df.groupby("method", as_index=False)["recommended_k"].first()
        for row in rec.itertuples(index=False):
            rec_lines.append(f"- {_method_display_name(str(row.method))}: K={int(row.recommended_k)}")
    axes6[1, 1].text(0.02, 0.98, "\n".join(rec_lines), va="top", fontsize=11)
    fig6.tight_layout()
    fig6_path = fig_dir / "figure6_k_selection.png"
    fig6.savefig(fig6_path, dpi=220)
    plt.close(fig6)
    figures.append(fig6_path)

    # Figure 7: marker-space UMAP colored by dominant archetype/component and effective K for all methods.
    marker_umap = _compute_marker_umap(base_split.x_true, random_state=321)
    method_rows = [method for method in DEFAULT_METHODS if method in rep_runs]
    if not method_rows:
        return figures
    fig7, axes7 = plt.subplots(len(method_rows), 2, figsize=(14, 4.2 * len(method_rows)), constrained_layout=False)
    if len(method_rows) == 1:
        axes7 = np.expand_dims(axes7, axis=0)
    overlay_rows: list[dict[str, float | str | int]] = []
    for row_idx, method in enumerate(method_rows):
        run = rep_runs.get(method)
        if run is None:
            continue
        split = run.result.split_results["test"]
        weights = split.weights
        if weights is None or len(weights) == 0:
            continue
        norm = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
        dominant = np.argmax(norm, axis=1).astype(int)
        eff_k = np.exp(per_cell_weight_entropy(norm)).astype(np.float32)

        ax_dom = axes7[row_idx, 0]
        ax_eff = axes7[row_idx, 1]

        sc_dom = ax_dom.scatter(marker_umap[:, 0], marker_umap[:, 1], c=dominant, s=5, alpha=0.75, cmap="tab10")
        ax_dom.set_title(f"{_method_display_name(method)} | marker UMAP: dominant component/archetype")
        ax_dom.set_xlabel("UMAP-1")
        ax_dom.set_ylabel("UMAP-2")
        fig7.colorbar(sc_dom, ax=ax_dom, fraction=0.046, pad=0.04)

        sc_eff = ax_eff.scatter(
            marker_umap[:, 0],
            marker_umap[:, 1],
            c=eff_k,
            s=5,
            alpha=0.75,
            cmap="viridis",
            norm=Normalize(vmin=1.0, vmax=float(run.representation_dim)),
        )
        ax_eff.set_title(f"{_method_display_name(method)} | marker UMAP: effective K")
        ax_eff.set_xlabel("UMAP-1")
        ax_eff.set_ylabel("UMAP-2")
        fig7.colorbar(sc_eff, ax=ax_eff, fraction=0.046, pad=0.04)

        labels = split.labels if split.labels is not None else np.asarray([""] * len(split.cell_ids), dtype=object)
        for i in range(len(split.cell_ids)):
            overlay_rows.append(
                {
                    "method": method,
                    "seed": int(run.seed),
                    "k": int(run.representation_dim),
                    "cell_id": str(split.cell_ids[i]),
                    "label": str(labels[i]),
                    "umap_1": float(marker_umap[i, 0]),
                    "umap_2": float(marker_umap[i, 1]),
                    "dominant_archetype": int(dominant[i]),
                    "effective_k": float(eff_k[i]),
                }
            )
    fig7.tight_layout()
    fig7_path = fig_dir / "figure7_marker_umap_effective_k_and_dominant_archetypes.png"
    fig7.savefig(fig7_path, dpi=220)
    plt.close(fig7)
    figures.append(fig7_path)

    overlay_df = pd.DataFrame(
        overlay_rows,
        columns=[
            "method",
            "seed",
            "k",
            "cell_id",
            "label",
            "umap_1",
            "umap_2",
            "dominant_archetype",
            "effective_k",
        ],
    )
    overlay_df.to_csv(tables_dir / "levine32_marker_umap_dominant_effective_k.csv", index=False)

    return figures


def write_interpretation_report(
    method_summary: pd.DataFrame,
    k_sweep: pd.DataFrame,
    per_class: pd.DataFrame,
    output_root: Path,
) -> Path:
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Levine32 Deconvolution Benchmark Report",
        "",
        "## Scope",
        "- Dataset: public Levine32 CyTOF (`cells x 32 markers`).",
        "- Cells with missing/unassigned labels were removed before splitting.",
        "- Methods benchmarked: NMF, classical archetypal analysis, deterministic archetypal AE, probabilistic archetypal AE.",
        "- K sweep: 4, 6, 8, 10.",
        "- Seeds: 13, 17, 23.",
        "",
        "## Interpretation Goals",
        "### 1) Do cells look one-hot or mixed under each method?",
    ]
    if "weight_entropy_mean" in method_summary.columns:
        entropy = (
            method_summary.groupby("method", as_index=False)["weight_entropy_mean"]
            .mean()
            .sort_values("weight_entropy_mean")
        )
        for row in entropy.itertuples(index=False):
            lines.append(f"- {_method_display_name(str(row.method))}: mean entropy={float(row.weight_entropy_mean):.4f}")
    lines.extend(
        [
            "",
            "### 2) Are inferred components biologically interpretable?",
            "- See `tables/levine32_component_marker_profiles.csv` and Figure 3 for marker-level component signatures.",
            "- Interpretability is judged by component-marker contrast and class-associated weight structure.",
            "",
            "### 3) Does probabilistic modeling improve over deterministic archetypes?",
        ]
    )
    det = method_summary[method_summary["method"] == "deterministic_archetypal_ae"]
    prob = method_summary[method_summary["method"] == "probabilistic_archetypal_ae"]
    merged = det.merge(prob, on=["seed", "k"], suffixes=("_det", "_prob"))
    if not merged.empty:
        win_rate = float(np.mean(merged["test_mse_prob"] < merged["test_mse_det"]))
        lines.append(f"- Probabilistic model has lower test MSE in {win_rate * 100:.1f}% of paired runs.")
        lines.append(
            f"- Mean test MSE deterministic={float(merged['test_mse_det'].mean()):.5f}, "
            f"probabilistic={float(merged['test_mse_prob'].mean()):.5f}."
        )
    lines.extend(
        [
            "",
            "### 4) What K balances fit and interpretability?",
            "- See `tables/levine32_k_sweep_summary.csv` and Figure 6.",
            "- Recommendation follows smallest near-optimal K per method from the K-selection analysis.",
            "",
            "## Focused Conclusions",
            "- The benchmark is centered on deconvolution mixtures, not trajectory/pseudotime analysis.",
            "- Use component profiles and class-wise errors jointly for final biological interpretation.",
            "",
            "## Reproducibility Artifacts",
            "- `reports/config_resolved.json`",
            "- `reports/split_manifest.csv`",
            "- `reports/environment_log.json`",
            "- `reports/preprocessing_details.json`",
            "- Per-run artifacts in `runs/<method>/dim_##/seed_##/`",
        ]
    )

    report_path = reports_dir / "levine32_benchmark_report.md"
    write_markdown(report_path, "\n".join(lines))
    return report_path


def write_experiment_readme(
    dataset_info: PreparedComparisonDataset,
    output_root: Path,
    args: argparse.Namespace,
) -> Path:
    readme_path = REPO_ROOT / "comparisons" / "README.md"
    text = f"""# Levine32 Deconvolution Benchmark

This directory contains a reproducible benchmark comparing deconvolution methods on Levine32:

1. NMF
2. Classical archetypal analysis
3. Deterministic archetypal autoencoder
4. Probabilistic archetypal autoencoder

## Dataset
- Source file: `{dataset_info.labeled_h5ad_path}`
- Raw cells in source: `{dataset_info.n_cells_raw}`
- Labeled cells kept: `{dataset_info.n_cells_labeled}`
- Unlabeled cells removed: `{dataset_info.n_unlabeled_removed}`
- Markers: `{len(dataset_info.markers)}`

## Preprocessing
- Input already appears arcsinh(cofactor=5) + z-score normalized in source preprocessing logs.
- No extra clipping was applied in this benchmark.
- Shared representation for all methods: labeled marker matrix as provided in source.
- NMF nonnegative requirement: train-min shift saved in `outputs/datasets/representations/nmf_nonnegative_representation.npz`.

## Splits
- Train/val/test: `{1.0 - args.val_fraction - args.test_fraction:.2f}/{args.val_fraction:.2f}/{args.test_fraction:.2f}`
- Stratified by label
- Fixed split seed: `{args.seed}`
- Split manifest: `outputs/reports/split_manifest.csv`

## Run command
```bash
python3 comparisons/run_levine32_deconvolution_benchmark.py --config comparisons/configs/levine32_benchmark.yaml
```

## Notebook-first workflow
- Main notebook: `comparisons/00_levine32_full_benchmark_from_config.ipynb`
- Config file: `comparisons/configs/levine32_benchmark.yaml`
- NN training progress bars are controlled by:
  - `show_training_progress`
  - `training_progress_level` (`epoch` or `batch`)
  - `training_progress_leave`

To regenerate figures/tables from existing run artifacts:
```bash
python3 comparisons/run_levine32_deconvolution_benchmark.py --config comparisons/configs/levine32_benchmark.yaml --skip-benchmark
```

## Key deliverables
- Processed dataset + metadata: `outputs/datasets/`
- Per-method run artifacts: `outputs/runs/`
- Required summary tables: `outputs/tables/levine32_*.csv`
- Main figures: `outputs/figures/figure*.png`
- Benchmark report: `outputs/reports/levine32_benchmark_report.md`
"""
    readme_path.write_text(text, encoding="utf-8")
    return readme_path


def write_comparison_notebooks(output_root: Path) -> list[Path]:
    notebook_dir = REPO_ROOT / "comparisons"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    run_nb = notebook_dir / "01_run_levine32_benchmark.ipynb"
    report_nb = notebook_dir / "02_levine32_benchmark_report.ipynb"

    run_cells: list[dict[str, Any]] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Experiment: Run Levine32 deconvolution benchmark\n",
                "\n",
                "This notebook runs the benchmark pipeline and writes all outputs under `comparisons/outputs`.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import subprocess\n",
                "import sys\n",
                "\n",
                "REPO_ROOT = Path.cwd()\n",
                "if not (REPO_ROOT / 'comparisons').exists():\n",
                "    REPO_ROOT = Path('/Users/ronguy/Dropbox/Work/CyTOF/Experiments/ProbAE_Deconv')\n",
                "REPO_ROOT\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cmd = [\n",
                "    sys.executable,\n",
                "    str(REPO_ROOT / 'comparisons' / 'run_levine32_deconvolution_benchmark.py'),\n",
                "    '--run-benchmark',\n",
                "]\n",
                "print('Running:', ' '.join(cmd))\n",
                "subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)\n",
            ],
        },
    ]

    report_cells: list[dict[str, Any]] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Experiment: Levine32 benchmark summary\n",
                "\n",
                "This notebook loads the benchmark outputs and summarizes the key findings.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "from IPython.display import Image, display\n",
                "\n",
                "REPO_ROOT = Path.cwd()\n",
                "if not (REPO_ROOT / 'comparisons').exists():\n",
                "    REPO_ROOT = Path('/Users/ronguy/Dropbox/Work/CyTOF/Experiments/ProbAE_Deconv')\n",
                "OUT = REPO_ROOT / 'comparisons' / 'outputs'\n",
                "OUT\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "method_summary = pd.read_csv(OUT / 'tables' / 'levine32_method_summary.csv')\n",
                "k_sweep = pd.read_csv(OUT / 'tables' / 'levine32_k_sweep_summary.csv')\n",
                "per_class = pd.read_csv(OUT / 'tables' / 'levine32_per_class_metrics.csv')\n",
                "method_summary.head()\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(k_sweep.sort_values(['method', 'k']).head(20))\n",
                "display(per_class.sort_values(['method', 'k', 'label']).head(20))\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for fig_name in [\n",
                "    'figure1_levine32_benchmark_overview.png',\n",
                "    'figure2_cells_are_mixtures_not_discrete.png',\n",
                "    'figure3_component_interpretability.png',\n",
                "    'figure4_deterministic_vs_probabilistic.png',\n",
                "    'figure5_method_comparison_summary.png',\n",
                "    'figure6_k_selection.png',\n",
                "    'figure7_marker_umap_effective_k_and_dominant_archetypes.png',\n",
                "]:\n",
                "    p = OUT / 'figures' / fig_name\n",
                "    print(p, 'exists=', p.exists())\n",
                "    if p.exists():\n",
                "        display(Image(filename=str(p), width=1100))\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "report_path = OUT / 'reports' / 'levine32_benchmark_report.md'\n",
                "print(report_path)\n",
                "print(report_path.read_text()[:4000])\n",
            ],
        },
    ]

    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }

    for path, cells in ((run_nb, run_cells), (report_nb, report_cells)):
        notebook = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
        path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")

    return [run_nb, report_nb]


def save_extended_environment_log(output_root: Path, args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": getattr(torch, "__version__", "unknown"),
        "seed_for_split": int(args.seed),
        "method_seeds": [int(v) for v in args.seeds],
        "k_values": [int(v) for v in args.k_values],
    }
    try:
        import sklearn

        payload["scikit_learn"] = sklearn.__version__
    except Exception:
        payload["scikit_learn"] = "unknown"
    write_json(output_root / "reports" / "environment_log_extended.json", payload)


def _load_table(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def main() -> None:
    args = parse_args()
    args = _apply_runtime_config(args)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    write_json(
        output_root / "reports" / "config_resolved.json",
        {
            "config_path": None if not args.config else str(Path(args.config).expanduser().resolve()),
            "input_h5ad": str(Path(args.input_h5ad).expanduser().resolve()),
            "output_root": str(output_root),
            "split_seed": int(args.seed),
            "method_seeds": [int(v) for v in args.seeds],
            "k_values": [int(v) for v in args.k_values],
            "val_fraction": float(args.val_fraction),
            "test_fraction": float(args.test_fraction),
            "device": str(args.device),
            "cpu_workers": int(args.cpu_workers),
            "show_progress": bool(args.show_progress),
            "show_training_progress": bool(args.show_training_progress),
            "training_progress_level": str(args.training_progress_level),
            "training_progress_leave": bool(args.training_progress_leave),
            "training_progress_log_every_epochs": int(args.training_progress_log_every_epochs),
            "optimization": {
                "nmf_max_iter": int(args.nmf_max_iter),
                "classical_n_iters": int(args.classical_n_iters),
                "classical_pg_steps": int(args.classical_pg_steps),
                "classical_lr": float(args.classical_lr),
                "neural_hidden_dims": [int(v) for v in args.hidden_dims],
                "neural_batch_size": int(args.batch_size),
                "neural_max_epochs": int(args.max_epochs),
                "neural_patience": int(args.patience),
            },
        },
    )

    save_environment_log(output_root / "reports" / "environment_log.json", extra={"benchmark": "levine32_comparisons"})
    save_extended_environment_log(output_root, args)

    dataset_bundle = prepare_labeled_levine32_dataset(
        source_h5ad_path=Path(args.input_h5ad).expanduser().resolve(),
        output_root=output_root,
        split_seed=int(args.seed),
        val_fraction=float(args.val_fraction),
        test_fraction=float(args.test_fraction),
    )
    write_experiment_readme(dataset_bundle, output_root, args)

    runs: list[BenchmarkRun] = []
    summary_df = pd.DataFrame()
    if args.skip_benchmark and not args.run_benchmark:
        summary_df = _load_table(output_root / "reports" / "core_run_summary.csv")
    else:
        runs, summary_df = run_benchmark_core(dataset_bundle, output_root, args)

    if summary_df.empty:
        summary_df = _load_table(output_root / "reports" / "core_run_summary.csv")
    if summary_df.empty:
        raise RuntimeError("No core benchmark summary is available. Run with --run-benchmark first.")

    # If runs are not in memory (skip mode), rebuild lightweight run metadata from artifacts where possible.
    if not runs:
        print("No in-memory run objects available; figures requiring run internals may be limited.", flush=True)

    fit_df = run_fit_vs_complexity(summary_df=summary_df, output_root=output_root)
    deconv_df = compute_deconvolution_summary(runs=runs, output_root=output_root) if runs else _load_table(output_root / "tables" / "deconvolution_quality_summary.csv")
    k_df = compute_k_selection_table(summary_df=summary_df, deconv_df=deconv_df, output_root=output_root) if runs else _load_table(output_root / "tables" / "k_selection_summary.csv")
    component_profiles = compute_component_profiles(runs=runs, output_root=output_root) if runs else _load_table(output_root / "tables" / "component_marker_profiles.csv")

    required = build_required_tables(
        runs=runs,
        summary_df=summary_df,
        fit_df=fit_df,
        deconv_df=deconv_df,
        k_df=k_df,
        component_profiles=component_profiles,
        output_root=output_root,
    )

    method_summary = _load_table(required["method_summary"])
    k_sweep = _load_table(required["k_sweep"])
    per_class = _load_table(required["per_class"])

    figures = generate_figures(
        runs=runs,
        method_summary=method_summary,
        k_sweep=k_sweep,
        per_class=per_class,
        k_df=k_df,
        dataset_bundle=dataset_bundle,
        output_root=output_root,
    ) if runs else []

    report_path = write_interpretation_report(
        method_summary=method_summary,
        k_sweep=k_sweep,
        per_class=per_class,
        output_root=output_root,
    )
    notebooks = write_comparison_notebooks(output_root=output_root)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "dataset": {
            "labeled_h5ad_path": str(dataset_bundle.labeled_h5ad_path),
            "metadata_path": str(dataset_bundle.metadata_path),
            "split_manifest_path": str(dataset_bundle.split_manifest_path),
            "processed_matrix_path": str(dataset_bundle.processed_matrix_path),
            "representations_dir": str(dataset_bundle.representations_dir),
            "n_cells_raw": dataset_bundle.n_cells_raw,
            "n_cells_labeled": dataset_bundle.n_cells_labeled,
            "n_unlabeled_removed": dataset_bundle.n_unlabeled_removed,
            "n_markers": len(dataset_bundle.markers),
        },
        "required_tables": {key: str(path) for key, path in required.items()},
        "figures": [str(path) for path in figures],
        "report_path": str(report_path),
        "notebooks": [str(path) for path in notebooks],
    }
    write_json(output_root / "reports" / "benchmark_manifest.json", manifest)

    print("Benchmark completed.", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

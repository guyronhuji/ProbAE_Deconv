from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Keep plotting/cache writes inside the repo workspace.
os.environ.setdefault("MPLCONFIGDIR", str((REPO_ROOT / "tmp" / "mplconfig").resolve()))
os.environ.setdefault("NUMBA_CACHE_DIR", str((REPO_ROOT / "tmp" / "numba_cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

from cytof_archetypes.experiments.common import prepare_data, run_core_benchmark
from cytof_archetypes.experiments.run_k_selection import run_k_selection


DEFAULT_CONFIG_PATH = REPO_ROOT / "PDAC" / "configs" / "pae_k_sweep.yaml"
PAE_METHOD = "probabilistic_archetypal_ae"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PDAC PAE K sweep across random seeds.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to sweep YAML config (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    cfg_path = path.expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    if "dataset" not in cfg:
        raise ValueError("Missing required 'dataset' section in config.")
    if "methods" not in cfg or PAE_METHOD not in cfg["methods"]:
        raise ValueError(f"Missing required methods.{PAE_METHOD} section in config.")
    if "sweeps" not in cfg or "k_values" not in cfg["sweeps"]:
        raise ValueError("Missing required sweeps.k_values in config.")
    if "seeds" not in cfg:
        raise ValueError("Missing required 'seeds' list in config.")

    dataset_cfg = dict(cfg["dataset"])
    if "input_path" not in dataset_cfg:
        raise ValueError("dataset.input_path is required.")
    input_path = Path(dataset_cfg["input_path"])
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    dataset_cfg["input_path"] = str(input_path.resolve())
    cfg["dataset"] = dataset_cfg

    out_dir = Path(cfg.get("output_dir", "outputs/PDAC/pae_k_sweep"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    cfg["output_dir"] = str(out_dir.resolve())
    return cfg


def _build_core_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "seeds": [int(v) for v in cfg.get("seeds", [42, 123, 456])],
        "sweeps": {
            "k_values": [int(v) for v in cfg["sweeps"].get("k_values", [])],
            "latent_dims": [int(v) for v in cfg["sweeps"].get("latent_dims", cfg["sweeps"].get("k_values", []))],
        },
        "methods": {PAE_METHOD: dict(cfg["methods"][PAE_METHOD])},
        "method_order": [PAE_METHOD],
        "show_progress": bool(cfg.get("show_progress", True)),
        "show_run_logs": bool(cfg.get("show_run_logs", True)),
        "show_training_progress": bool(cfg.get("show_training_progress", True)),
        "training_progress_level": str(cfg.get("training_progress_level", "epoch")),
        "training_progress_leave": bool(cfg.get("training_progress_leave", False)),
        "cpu_multiprocessing_workers": int(cfg.get("cpu_multiprocessing_workers", 1)),
        "cpu_parallel_methods": list(cfg.get("cpu_parallel_methods", [])),
        "gpu_multiprocessing_workers": int(cfg.get("gpu_multiprocessing_workers", 1)),
        "gpu_parallel_methods": list(cfg.get("gpu_parallel_methods", [PAE_METHOD])),
    }


def _write_pae_sweep_tables(summary_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    per_run = summary_df.loc[summary_df["method"] == PAE_METHOD].copy()
    per_run = per_run.rename(columns={"representation_dim": "k"})
    per_run = per_run.sort_values(["k", "seed"]).reset_index(drop=True)
    per_run_path = tables_dir / "pae_k_sweep_seed_level.csv"
    per_run.to_csv(per_run_path, index=False)

    metric_cols = ["val_mse", "val_nll", "test_mse", "test_nll", "ari", "nmi", "knn_purity", "silhouette"]
    agg = (
        per_run.groupby("k")
        .agg(
            n_runs=("seed", "count"),
            **{f"{col}_mean": (col, "mean") for col in metric_cols},
            **{f"{col}_std": (col, "std") for col in metric_cols},
        )
        .reset_index()
        .sort_values("k")
    )
    agg_path = tables_dir / "pae_k_sweep_k_aggregate.csv"
    agg.to_csv(agg_path, index=False)
    return per_run_path, agg_path


def run_pae_k_sweep(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = _load_config(Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_data(
        dataset_cfg=cfg["dataset"],
        preprocessing_cfg=dict(cfg.get("preprocessing", {})),
        seed=int(cfg.get("seed", 42)),
    )

    core_cfg = _build_core_cfg(cfg)
    runs, summary_df = run_core_benchmark(prepared=prepared, suite_cfg=core_cfg, output_root=output_dir)

    per_run_path, agg_path = _write_pae_sweep_tables(summary_df=summary_df, out_dir=output_dir)

    k_summary = run_k_selection(runs=runs, output_root=output_dir)
    k_summary = k_summary.loc[k_summary["method"] == PAE_METHOD].copy()
    k_summary_path = output_dir / "tables" / "pae_k_selection_summary.csv"
    k_summary.to_csv(k_summary_path, index=False)

    recommended = "n/a"
    if not k_summary.empty and "recommended_k" in k_summary.columns:
        recommended = str(int(k_summary["recommended_k"].iloc[0]))

    return {
        "output_dir": str(output_dir),
        "seed_level_csv": str(per_run_path),
        "k_aggregate_csv": str(agg_path),
        "k_selection_csv": str(k_summary_path),
        "recommended_k": recommended,
    }


def main() -> None:
    args = _parse_args()
    result = run_pae_k_sweep(config_path=args.config)

    print(f"Output directory: {result['output_dir']}")
    print(f"Seed-level sweep table: {result['seed_level_csv']}")
    print(f"K-aggregate sweep table: {result['k_aggregate_csv']}")
    print(f"K-selection table: {result['k_selection_csv']}")
    print(f"Recommended K (PAE): {result['recommended_k']}")


if __name__ == "__main__":
    main()

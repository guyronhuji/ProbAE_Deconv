from __future__ import annotations

import copy
import multiprocessing as mp
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cytof_archetypes.evaluation.reporting import save_environment_log, write_json
from cytof_archetypes.multimodal import DEFAULT_MULTIMODAL_CONFIG, resolve_multimodal_paths, train_multimodal_from_config


MULTIMODAL_SUITE_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "output_dir": "outputs/multimodal_suite",
    "show_progress": True,
    "multiprocessing_workers": 1,
    "seeds": [42, 123, 456],
    "sweeps": {
        "k_values": [4, 6, 8, 10],
    },
    "base_config": copy.deepcopy(DEFAULT_MULTIMODAL_CONFIG),
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_multimodal_suite_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(MULTIMODAL_SUITE_DEFAULTS)

    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}
    merged = _deep_update(MULTIMODAL_SUITE_DEFAULTS, user_cfg)

    base_config = merged.get("base_config", {})
    merged["base_config"] = resolve_multimodal_paths(base_config, config_dir=cfg_path.parent)

    out_dir = Path(merged.get("output_dir", "outputs/multimodal_suite")).expanduser()
    if not out_dir.is_absolute():
        out_dir = (cfg_path.parent / out_dir).resolve()
    merged["output_dir"] = str(out_dir)
    return merged


def _run_single_multimodal_job(job: dict[str, Any]) -> dict[str, Any]:
    k = int(job["k"])
    seed = int(job["seed"])
    run_cfg = copy.deepcopy(job["run_config"])

    run_dir = train_multimodal_from_config(run_cfg)
    summary_path = Path(run_dir) / "metrics" / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = yaml.safe_load(handle) or {}

    recon_csv = Path(run_dir) / "metrics" / "reconstruction_metrics.csv"
    recon_df = pd.read_csv(recon_csv) if recon_csv.exists() else pd.DataFrame()

    row: dict[str, Any] = {
        "k": k,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_epoch": summary.get("best_epoch"),
        "best_val_loss_total": summary.get("best_val_loss_total"),
    }
    if not recon_df.empty:
        val_df = recon_df[recon_df["split"] == "val"]
        test_df = recon_df[recon_df["split"] == "test"]
        row["val_nll_mean"] = float(val_df["nll"].mean()) if len(val_df) else float("nan")
        row["test_nll_mean"] = float(test_df["nll"].mean()) if len(test_df) else float("nan")
        row["val_mse_mean"] = float(val_df["mse"].mean()) if len(val_df) else float("nan")
        row["test_mse_mean"] = float(test_df["mse"].mean()) if len(test_df) else float("nan")
    else:
        row["val_nll_mean"] = float("nan")
        row["test_nll_mean"] = float("nan")
        row["val_mse_mean"] = float("nan")
        row["test_mse_mean"] = float("nan")
    return row


def run_multimodal_experiment_suite(config: dict[str, Any]) -> Path:
    resolved = copy.deepcopy(config)
    output_root = Path(resolved.get("output_dir", "outputs/multimodal_suite"))
    runs_root = output_root / "runs"
    tables_dir = output_root / "tables"
    reports_dir = output_root / "reports"

    runs_root.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    save_environment_log(reports_dir / "environment_log.json", extra={"suite": "multimodal"})
    write_json(reports_dir / "config_resolved.json", resolved)

    seeds = [int(v) for v in resolved.get("seeds", [42, 123, 456])]
    k_values = [int(v) for v in resolved.get("sweeps", {}).get("k_values", [4, 6, 8, 10])]
    base_config = copy.deepcopy(resolved.get("base_config", DEFAULT_MULTIMODAL_CONFIG))

    jobs: list[dict[str, Any]] = []
    for k in k_values:
        for seed in seeds:
            run_cfg = copy.deepcopy(base_config)
            run_cfg["seed"] = int(seed)
            run_cfg.setdefault("shared_latent", {})
            run_cfg["shared_latent"]["n_archetypes"] = int(k)
            run_cfg.setdefault("output", {})
            run_cfg["output"]["base_dir"] = str(runs_root)
            run_cfg["output"]["run_name"] = f"k_{k:02d}_seed_{seed}"
            jobs.append({"k": k, "seed": seed, "run_config": run_cfg})

    workers = int(resolved.get("multiprocessing_workers", 1))
    if workers > 1 and len(jobs) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            rows = pool.map(_run_single_multimodal_job, jobs)
    else:
        rows = [_run_single_multimodal_job(job) for job in jobs]

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(tables_dir / "multimodal_k_sweep_summary.csv", index=False)

    grouped = summary_df.groupby("k", dropna=False)["best_val_loss_total"].mean().reset_index()
    grouped = grouped.sort_values("best_val_loss_total", ascending=True)
    grouped.to_csv(tables_dir / "multimodal_k_ranked.csv", index=False)

    best_k = int(grouped.iloc[0]["k"]) if len(grouped) else None
    best_row = {}
    if best_k is not None:
        best_group = summary_df[summary_df["k"] == best_k].sort_values("best_val_loss_total", ascending=True)
        if len(best_group):
            best_row = best_group.iloc[0].to_dict()

    write_json(
        reports_dir / "selection_summary.json",
        {
            "recommended_k": best_k,
            "criterion": "lowest mean best_val_loss_total across seeds",
            "best_run": best_row,
        },
    )
    return output_root

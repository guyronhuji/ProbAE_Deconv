from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None

from cytof_archetypes.evaluation.reporting import save_environment_log, write_json, write_markdown
from cytof_archetypes.evaluation.statistics import bootstrap_mean_ci
from cytof_archetypes.experiments.common import prepare_data, run_core_benchmark
from cytof_archetypes.experiments.run_auxiliary_representation_models import run_auxiliary_representation_models
from cytof_archetypes.experiments.run_component_biology import run_component_biology
from cytof_archetypes.experiments.run_deconvolution_quality import run_deconvolution_quality
from cytof_archetypes.experiments.run_deterministic_vs_probabilistic import run_deterministic_vs_probabilistic
from cytof_archetypes.experiments.run_fit_vs_complexity import run_fit_vs_complexity
from cytof_archetypes.experiments.run_interpretability_tradeoff import run_interpretability_tradeoff
from cytof_archetypes.experiments.run_k_selection import run_k_selection
from cytof_archetypes.experiments.run_rare_population_preservation import run_rare_population_preservation
from cytof_archetypes.experiments.run_secondary_dataset_validation import run_secondary_dataset_validation
from cytof_archetypes.notebook_generation.writer import generate_benchmark_notebooks

REPO_ROOT = Path(__file__).resolve().parents[3]


SUITE_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "output_dir": "outputs/experiment_suite",
    "notebook_output_dir": "notebooks/experiment_suite",
    "show_progress": True,
    "show_run_logs": True,
    "show_training_progress": True,
    "training_progress_level": "epoch",
    "training_progress_leave": False,
    "cpu_multiprocessing_workers": 1,
    "cpu_parallel_methods": ["nmf", "classical_archetypes"],
    "device": "auto",
    "dataset": {
        "name": "levine32",
        "input_path": "data/levine32_processed.h5ad",
        "marker_columns": None,
        "label_column": "label",
        "cell_id_column": "cell_id",
        "val_fraction": 0.15,
        "test_fraction": 0.15,
    },
    "preprocessing": {
        "transform": "none",
        "arcsinh_cofactor": 5.0,
        "normalization": "zscore",
        "clip_min": None,
        "clip_max": None,
    },
    "seeds": [13, 17, 23],
    "sweeps": {
        "k_values": [4, 6, 8, 10],
        "latent_dims": [4, 6, 8, 10],
    },
    "methods": {
        "nmf": {"max_iter": 1200},
        "classical_archetypes": {"n_iters": 40, "pg_steps": 80, "lr": 0.1},
        "deterministic_archetypal_ae": {
            "device": "auto",
            "hidden_dims": [128, 64],
            "dropout": 0.1,
            "batch_size": 256,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "max_epochs": 80,
            "patience": 12,
            "grad_clip": 1.0,
            "entropy_reg_weight": 1e-3,
            "diversity_reg_weight": 1e-3,
        },
        "probabilistic_archetypal_ae": {
            "device": "auto",
            "hidden_dims": [128, 64],
            "dropout": 0.1,
            "batch_size": 256,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "max_epochs": 80,
            "patience": 12,
            "grad_clip": 1.0,
            "entropy_reg_weight": 1e-3,
            "diversity_reg_weight": 1e-3,
            "variance_reg_weight": 1e-5,
        },
        "ae": {
            "device": "auto",
            "hidden_dims": [128, 64],
            "dropout": 0.1,
            "batch_size": 256,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "max_epochs": 80,
            "patience": 12,
            "grad_clip": 1.0,
        },
        "vae": {
            "device": "auto",
            "hidden_dims": [128, 64],
            "dropout": 0.1,
            "batch_size": 256,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "max_epochs": 80,
            "patience": 12,
            "grad_clip": 1.0,
            "beta": 1.0,
        },
    },
    "secondary_dataset": {
        "enabled": False,
        "dataset": {
            "name": "secondary",
            "input_path": "data/secondary_dataset_placeholder.csv",
            "marker_columns": None,
            "label_column": "label",
            "cell_id_column": "cell_id",
            "val_fraction": 0.15,
            "test_fraction": 0.15,
        },
        "method_order": ["nmf", "classical_archetypes", "deterministic_archetypal_ae", "probabilistic_archetypal_ae"],
    },
    "auxiliary_models": {"enabled": False},
    "marker_groups": {},
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_suite_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        merged = _deep_update({}, SUITE_DEFAULTS)
        return _resolve_config_paths(merged, config_dir=REPO_ROOT)

    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}
    merged = _deep_update(SUITE_DEFAULTS, user_cfg)
    return _resolve_config_paths(merged, config_dir=cfg_path.parent)


def run_experiment_suite(config: dict[str, Any]) -> Path:
    resolved_cfg = _resolve_config_paths(copy.deepcopy(config), config_dir=REPO_ROOT)
    out = Path(resolved_cfg.get("output_dir", str(REPO_ROOT / "outputs" / "experiment_suite")))
    notebook_dir = Path(resolved_cfg.get("notebook_output_dir", str(REPO_ROOT / "notebooks" / "experiment_suite")))
    show_progress = bool(resolved_cfg.get("show_progress", True))
    show_run_logs = bool(resolved_cfg.get("show_run_logs", True))
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    notebook_dir.mkdir(parents=True, exist_ok=True)

    with (out / "reports" / "config_resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved_cfg, handle, indent=2, sort_keys=True)

    save_environment_log(
        out / "reports" / "suite_environment_log.json",
        extra={"suite": "probabilistic_archetypal_deconvolution"},
    )

    phase_labels = [
        "data prepared",
        "core methods done",
        "fit vs complexity",
        "deconvolution quality",
        "det vs prob",
        "k selection",
        "component biology",
        "rare populations",
        "secondary dataset",
        "auxiliary models",
        "interpretability",
        "notebooks",
    ]
    phase_bar = None
    if show_progress and tqdm is not None:
        phase_bar = tqdm(total=len(phase_labels), desc="Suite phases", unit="phase")

    def _phase_done(label: str) -> None:
        if phase_bar is not None:
            phase_bar.set_postfix_str(label)
            phase_bar.update(1)

    def _phase_start(label: str, detail: str) -> None:
        if show_run_logs:
            print(f"[suite-phase] {label}: {detail}", flush=True)

    try:
        # 1) Data loading and split system.
        _phase_start("1/12", "prepare_data")
        prepared = prepare_data(
            resolved_cfg["dataset"],
            resolved_cfg.get("preprocessing", {}),
            seed=int(resolved_cfg.get("seed", 42)),
        )
        _phase_done("data prepared")

        # 2-7) Core methods in order (NMF, classical AA, deterministic AAE, probabilistic AAE, AE, VAE).
        _phase_start("2/12", "run_core_benchmark (all methods × dims × seeds)")
        method_order = [
            "nmf",
            "classical_archetypes",
            "deterministic_archetypal_ae",
            "probabilistic_archetypal_ae",
            "ae",
            "vae",
        ]
        core_cfg = {
            "seeds": resolved_cfg.get("seeds", [13, 17, 23]),
            "sweeps": resolved_cfg.get("sweeps", {}),
            "methods": resolved_cfg.get("methods", {}),
            "method_order": method_order,
            "show_progress": show_progress,
            "show_run_logs": show_run_logs,
            "show_training_progress": bool(resolved_cfg.get("show_training_progress", show_progress)),
            "training_progress_level": str(resolved_cfg.get("training_progress_level", "epoch")),
            "training_progress_leave": bool(resolved_cfg.get("training_progress_leave", False)),
            "cpu_multiprocessing_workers": int(resolved_cfg.get("cpu_multiprocessing_workers", 1)),
            "cpu_parallel_methods": resolved_cfg.get("cpu_parallel_methods", ["nmf", "classical_archetypes"]),
        }
        runs, summary_df = run_core_benchmark(prepared=prepared, suite_cfg=core_cfg, output_root=out)
        _phase_done("core methods done")

        # 8) Fit vs complexity.
        _phase_start("3/12", "run_fit_vs_complexity")
        fit_df = run_fit_vs_complexity(summary_df, out)
        _phase_done("fit vs complexity")

        # 9) Deconvolution quality.
        _phase_start("4/12", "run_deconvolution_quality")
        deconv_df = run_deconvolution_quality(runs, out)
        _phase_done("deconvolution quality")

        # 10) Deterministic vs probabilistic comparison.
        _phase_start("5/12", "run_deterministic_vs_probabilistic")
        det_prob_df = run_deterministic_vs_probabilistic(runs, out)
        _phase_done("det vs prob")

        # 11) K selection.
        _phase_start("6/12", "run_k_selection")
        k_df = run_k_selection(runs, out)
        _phase_done("k selection")

        # 12) Component biology analysis.
        _phase_start("7/12", "run_component_biology")
        component_profiles, enrichment_df = run_component_biology(
            runs,
            out,
            marker_groups=resolved_cfg.get("marker_groups", {}),
        )
        _phase_done("component biology")

        # 13) Rare/transitional population preservation.
        _phase_start("8/12", "run_rare_population_preservation")
        rare_df = run_rare_population_preservation(runs, out)
        _phase_done("rare populations")

        # 14) Optional second dataset.
        _phase_start("9/12", "run_secondary_dataset_validation")
        secondary_df = run_secondary_dataset_validation(resolved_cfg, out)
        _phase_done("secondary dataset")

        # 15) Optional auxiliary representation models.
        _phase_start("10/12", "run_auxiliary_representation_models")
        aux_df = run_auxiliary_representation_models(runs, resolved_cfg, out)
        _phase_done("auxiliary models")

        # 16) Interpretability vs flexibility (paper-wide summary).
        _phase_start("11/12", "run_interpretability_tradeoff")
        interp_df = run_interpretability_tradeoff(runs, out)
        _phase_done("interpretability")

        # 17) Notebook generation.
        _phase_start("12/12", f"generate_benchmark_notebooks -> {notebook_dir}")
        created_notebooks = generate_benchmark_notebooks(notebook_dir)
        _phase_done("notebooks")
    finally:
        if phase_bar is not None:
            phase_bar.close()

    # 18) Analysis-LLM docs and concise runbook.
    _write_suite_docs(out)

    _write_statistics_summary(out, fit_df, det_prob_df, interp_df)
    _write_runbook(out, resolved_cfg)

    # High-level manifest of generated artifacts for quick auditing.
    write_json(
        out / "reports" / "artifact_manifest.json",
        {
            "tables": sorted([p.name for p in (out / "tables").glob("*.csv")]),
            "plots": sorted([p.name for p in (out / "plots").glob("*.png")]),
            "notebook_output_dir": str(notebook_dir),
            "notebooks": sorted([path.name for path in created_notebooks]),
            "has_secondary_dataset_results": bool(len(secondary_df)),
            "has_auxiliary_results": bool(len(aux_df)),
            "n_component_profile_rows": int(len(component_profiles)),
            "n_enrichment_rows": int(len(enrichment_df)),
            "n_rare_rows": int(len(rare_df)),
            "n_deconvolution_rows": int(len(deconv_df)),
            "n_k_selection_rows": int(len(k_df)),
        },
    )

    return out


def _write_statistics_summary(out: Path, fit_df: pd.DataFrame, det_prob_df: pd.DataFrame, interp_df: pd.DataFrame) -> None:
    rows: list[dict[str, float | str]] = []
    if not fit_df.empty:
        ci = bootstrap_mean_ci(fit_df["test_mse"].to_numpy(), n_bootstrap=1000, ci=0.95, seed=0)
        rows.append({"metric": "test_mse_global", **ci})
    if not det_prob_df.empty:
        ci = bootstrap_mean_ci((det_prob_df["det_test_mse"] - det_prob_df["prob_test_mse"]).to_numpy(), n_bootstrap=1000, ci=0.95, seed=1)
        rows.append({"metric": "det_minus_prob_test_mse", **ci})
    if not interp_df.empty:
        ci = bootstrap_mean_ci(interp_df["interpretability_score"].to_numpy(), n_bootstrap=1000, ci=0.95, seed=2)
        rows.append({"metric": "interpretability_score_global", **ci})
    pd.DataFrame(rows).to_csv(out / "tables" / "statistical_summary.csv", index=False)


def _write_suite_docs(out: Path) -> None:
    readme_text = """# README for Analysis LLMs

## Purpose
This suite evaluates whether single-cell CyTOF profiles are better modeled as mixtures of probabilistic archetypal programs than by deterministic archetypes, classical deconvolution, and generic latent-variable models.

## Primary comparisons
1. Deterministic vs probabilistic archetypal autoencoders.
2. Deconvolution baselines (NMF, classical archetypal analysis) vs archetypal autoencoders.
3. Interpretability-fit tradeoff vs AE/VAE.
4. K-selection under fit, rare-class preservation, and interpretability.

## Key outputs
- `tables/k_selection_summary.csv`
- `tables/deterministic_vs_probabilistic_summary.csv`
- `tables/deconvolution_quality_summary.csv`
- `tables/fit_vs_interpretability.csv`
- `plots/deterministic_vs_probabilistic_comparison.png`
- `plots/class_component_mean_weight_heatmap.png`
- `plots/component_marker_heatmap.png`
- `plots/rare_population_preservation.png`

## Best K definition
Best K is the smallest K within 0.02 of the method-wise maximum K-selection score. The score combines fit, rare-class reconstruction, interpretability, and component non-redundancy.
"""

    instructions_text = """# ANALYSIS LLM INSTRUCTIONS

1. Inspect `tables/k_selection_summary.csv` first to identify recommended K per deconvolution method.
2. Compare deterministic and probabilistic archetypes using `tables/deterministic_vs_probabilistic_summary.csv` and `plots/deterministic_vs_probabilistic_comparison.png`.
3. Inspect class-component usage via `tables/class_component_means.csv` and `plots/class_component_mean_weight_heatmap.png`.
4. Interpret archetypes as latent biological programs, not strict one-to-one cell-type labels.
5. Use dataset-provided labels; never assume external cell-type names.
6. Avoid overinterpreting small metric differences that fall inside confidence intervals.
7. Keep auxiliary representation models (if present) supplementary; do not treat them as the core story.
"""

    write_markdown(out / "docs" / "README_for_analysis_llms.md", readme_text)
    write_markdown(out / "docs" / "ANALYSIS_LLM_INSTRUCTIONS.md", instructions_text)


def _write_runbook(out: Path, config: dict[str, Any]) -> None:
    runbook = f"""# Experiment Suite Runbook

## Entry point
```bash
python scripts/run_experiment_suite.py --config configs/experiment_suite.yaml
```

## Execution order
1. Data loading and split manifest
2. NMF baseline
3. Classical archetypal baseline
4. Deterministic archetypal autoencoder
5. Probabilistic archetypal autoencoder
6. AE baseline
7. VAE baseline
8. Fit vs complexity
9. Deconvolution quality
10. Deterministic vs probabilistic
11. K selection
12. Component biology
13. Rare/transitional preservation
14. Optional secondary dataset
15. Optional auxiliary representation models
16. Notebooks + analysis docs

## Reproducibility
- Fixed seeds: `{config.get('seeds', [])}`
- Resolved config: `reports/config_resolved.json`
- Split manifest: `reports/split_manifest.csv`
- Environment log: `reports/suite_environment_log.json`

## Primary deliverables
- Tables: `tables/*.csv`
- Plots: `plots/*.png` and `plots/*.pdf`
- Notebooks: `{config.get('notebook_output_dir', 'notebooks/experiment_suite')}/*.ipynb`
- Analysis docs: `docs/*.md`
"""
    write_markdown(out / "README.md", runbook)


def _resolve_notebook_dir(path_value: str | Path) -> Path:
    notebook_path = Path(path_value)
    if notebook_path.is_absolute():
        return notebook_path
    return REPO_ROOT / notebook_path


def _resolve_config_paths(config: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["output_dir"] = str(_resolve_path_for_output(cfg.get("output_dir", "outputs/experiment_suite"), config_dir))
    cfg["notebook_output_dir"] = str(
        _resolve_path_for_output(cfg.get("notebook_output_dir", "notebooks/experiment_suite"), config_dir)
    )

    dataset_cfg = cfg.get("dataset", {})
    if "input_path" in dataset_cfg:
        dataset_cfg["input_path"] = str(_resolve_existing_path(dataset_cfg["input_path"], config_dir))
    cfg["dataset"] = dataset_cfg

    secondary_cfg = cfg.get("secondary_dataset", {})
    secondary_dataset_cfg = secondary_cfg.get("dataset", {})
    if "input_path" in secondary_dataset_cfg:
        secondary_dataset_cfg["input_path"] = str(_resolve_existing_path(secondary_dataset_cfg["input_path"], config_dir))
    secondary_cfg["dataset"] = secondary_dataset_cfg
    cfg["secondary_dataset"] = secondary_cfg
    cfg = _resolve_method_devices(cfg)
    return cfg


def _resolve_existing_path(path_value: str | Path, config_dir: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate

    repo_candidate = (REPO_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    config_candidate = (config_dir / candidate).resolve()
    if config_candidate.exists():
        return config_candidate

    cwd_candidate = candidate.resolve()
    return cwd_candidate


def _resolve_path_for_output(path_value: str | Path, config_dir: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate

    # Prefer repository-root-relative outputs for consistency across launch locations.
    repo_candidate = (REPO_ROOT / candidate).resolve()
    return repo_candidate


def _resolve_method_devices(cfg: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(cfg)
    global_request = str(resolved.get("device", "auto"))
    global_device = _resolve_device_label(global_request)
    resolved["device"] = global_request
    resolved["resolved_device"] = global_device

    methods = resolved.setdefault("methods", {})
    neural_methods = [
        "deterministic_archetypal_ae",
        "probabilistic_archetypal_ae",
        "ae",
        "vae",
    ]
    for method_name in neural_methods:
        method_cfg = methods.setdefault(method_name, {})
        request = str(method_cfg.get("device", global_request))
        method_cfg["device"] = _resolve_device_label(request, fallback=global_device)
        methods[method_name] = method_cfg
    resolved["methods"] = methods
    return resolved


def _resolve_device_label(requested: str, fallback: str | None = None) -> str:
    req = str(requested).lower()
    available = _available_devices()
    default = fallback or _best_available_device(available)

    if req in {"auto", "mps_if_available"}:
        return _best_available_device(available)
    if req == "mps":
        return "mps" if available["mps"] else default
    if req == "cuda":
        return "cuda" if available["cuda"] else default
    if req == "cpu":
        return "cpu"
    return default


def _best_available_device(available: dict[str, bool]) -> str:
    if available["mps"]:
        return "mps"
    if available["cuda"]:
        return "cuda"
    return "cpu"


def _available_devices() -> dict[str, bool]:
    try:
        import torch
    except Exception:
        return {"mps": False, "cuda": False}

    mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    cuda_ok = bool(torch.cuda.is_available())
    return {"mps": mps_ok, "cuda": cuda_ok}

from __future__ import annotations

import copy
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None

from cytof_archetypes.baselines import (
    AEMethod,
    ClassicalArchetypeMethod,
    DeterministicArchetypalMethod,
    MethodRunResult,
    NMFMethod,
    ProbabilisticArchetypalMethod,
    VAEMethod,
    gaussian_nll_per_cell,
    write_method_artifacts,
)
from cytof_archetypes.datasets import DatasetBundle, load_dataset_bundle, split_manifest_frame
from cytof_archetypes.evaluation.metrics import (
    per_class_reconstruction_frame,
    reconstruction_metrics_per_cell,
    representation_structure_metrics,
)
from cytof_archetypes.evaluation.reporting import save_environment_log, write_json
from cytof_archetypes.preprocessing import MarkerPreprocessor
from cytof_archetypes.utils import set_seed

_MP_SHARED: dict[str, Any] | None = None


@dataclass
class PreparedData:
    dataset: DatasetBundle
    preprocessor: MarkerPreprocessor
    train_x: np.ndarray
    val_x: np.ndarray
    test_x: np.ndarray


@dataclass
class BenchmarkRun:
    method: str
    seed: int
    representation_dim: int
    run_dir: Path
    result: MethodRunResult
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    structure_metrics: dict[str, float]


def prepare_data(dataset_cfg: dict[str, Any], preprocessing_cfg: dict[str, Any], seed: int) -> PreparedData:
    bundle = load_dataset_bundle(dataset_cfg, seed=seed)
    bundle = _downsample_bundle_if_requested(bundle, dataset_cfg=dataset_cfg, seed=seed)
    preprocessor = MarkerPreprocessor(
        transform=preprocessing_cfg.get("transform", "none"),
        arcsinh_cofactor=float(preprocessing_cfg.get("arcsinh_cofactor", 5.0)),
        normalization=preprocessing_cfg.get("normalization", "zscore"),
        clip_min=preprocessing_cfg.get("clip_min"),
        clip_max=preprocessing_cfg.get("clip_max"),
    )
    train_x = preprocessor.fit_transform(bundle.train.x)
    val_x = preprocessor.transform_array(bundle.val.x) if len(bundle.val.x) else bundle.val.x
    test_x = preprocessor.transform_array(bundle.test.x) if len(bundle.test.x) else bundle.test.x
    return PreparedData(dataset=bundle, preprocessor=preprocessor, train_x=train_x, val_x=val_x, test_x=test_x)


def method_registry() -> dict[str, Any]:
    return {
        "nmf": NMFMethod(),
        "classical_archetypes": ClassicalArchetypeMethod(),
        "deterministic_archetypal_ae": DeterministicArchetypalMethod(),
        "probabilistic_archetypal_ae": ProbabilisticArchetypalMethod(),
        "ae": AEMethod(),
        "vae": VAEMethod(),
    }


def run_core_benchmark(
    prepared: PreparedData,
    suite_cfg: dict[str, Any],
    output_root: str | Path,
) -> tuple[list[BenchmarkRun], pd.DataFrame]:
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)

    save_environment_log(
        out / "reports" / "environment_log.json",
        extra={"dataset": prepared.dataset.name},
    )
    split_manifest_frame(prepared.dataset).to_csv(out / "reports" / "split_manifest.csv", index=False)
    write_json(out / "reports" / "preprocessor.json", prepared.preprocessor.state_dict())

    reg = method_registry()
    methods_cfg = copy.deepcopy(suite_cfg.get("methods", {}))
    sweep_cfg = suite_cfg.get("sweeps", {})
    k_values = [int(v) for v in sweep_cfg.get("k_values", [4, 6, 8, 10])]
    latent_values = [int(v) for v in sweep_cfg.get("latent_dims", [4, 6, 8, 10])]
    seeds = [int(v) for v in suite_cfg.get("seeds", [13, 17, 23])]

    runs: list[BenchmarkRun] = []
    summary_rows: list[dict[str, float | str | int]] = []

    method_order = suite_cfg.get(
        "method_order",
        [
            "nmf",
            "classical_archetypes",
            "deterministic_archetypal_ae",
            "probabilistic_archetypal_ae",
            "ae",
            "vae",
        ],
    )
    show_progress = bool(suite_cfg.get("show_progress", True))
    show_run_logs = bool(suite_cfg.get("show_run_logs", True))
    show_training_progress = bool(suite_cfg.get("show_training_progress", show_progress))
    training_progress_level = str(suite_cfg.get("training_progress_level", "epoch")).lower()
    if training_progress_level not in {"epoch", "batch"}:
        training_progress_level = "epoch"
    training_progress_leave = bool(suite_cfg.get("training_progress_leave", False))
    cpu_mp_workers = int(suite_cfg.get("cpu_multiprocessing_workers", 1))
    gpu_mp_workers = int(suite_cfg.get("gpu_multiprocessing_workers", 1))
    cpu_parallel_methods = set(
        suite_cfg.get("cpu_parallel_methods", ["nmf", "classical_archetypes"])
    )
    gpu_parallel_methods = set(
        suite_cfg.get(
            "gpu_parallel_methods",
            ["deterministic_archetypal_ae", "probabilistic_archetypal_ae", "ae", "vae"],
        )
    )
    neural_methods = {"deterministic_archetypal_ae", "probabilistic_archetypal_ae", "ae", "vae"}
    job_specs: list[tuple[str, int, int]] = []
    for method in method_order:
        if method not in reg:
            continue
        dims = (
            k_values
            if method in {"nmf", "classical_archetypes", "deterministic_archetypal_ae", "probabilistic_archetypal_ae"}
            else latent_values
        )
        for dim in dims:
            for seed in seeds:
                job_specs.append((method, dim, seed))

    cpu_methods = set(cpu_parallel_methods)
    cpu_job_total = sum(1 for method, _, _ in job_specs if method in cpu_methods)
    accel_job_total = len(job_specs) - cpu_job_total
    cpu_progress = None
    accel_progress = None
    if show_progress and tqdm is not None and len(job_specs) > 0:
        next_pos = 0
        if cpu_job_total > 0:
            cpu_progress = tqdm(total=cpu_job_total, desc="CPU benchmark runs", unit="run", position=next_pos)
            next_pos += 1
        if accel_job_total > 0:
            accel_progress = tqdm(total=accel_job_total, desc="MPS/GPU benchmark runs", unit="run", position=next_pos)

    try:
        run_index_map = {(m, d, s): idx for idx, (m, d, s) in enumerate(job_specs, start=1)}
        pending_cpu_mp_jobs: list[dict[str, Any]] = []
        pending_gpu_mp_jobs: list[dict[str, Any]] = []
        serial_jobs: list[dict[str, Any]] = []

        def _consume_completed(payload: dict[str, Any]) -> None:
            method = str(payload["method"])
            dim = int(payload["dim"])
            seed = int(payload["seed"])
            run_idx = int(payload["run_idx"])
            run_t0 = payload.get("run_t0")
            run_dir = Path(payload["run_dir"])
            result = payload["result"]
            write_method_artifacts(result, run_dir)

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
                    "param_count": int(result.params.get("param_count", _fallback_param_count(result))),
                }
            )
            _save_per_class_tables(run_dir=run_dir, labels=test_split.labels, mse=test_mse, nll=test_nll)
            if show_run_logs:
                run_elapsed = payload.get("run_elapsed")
                if isinstance(run_elapsed, (float, int)):
                    elapsed = float(run_elapsed)
                elif isinstance(run_t0, (float, int)):
                    elapsed = time.time() - float(run_t0)
                else:
                    elapsed = float("nan")
                print(
                    f"[core-run {run_idx}/{len(job_specs)}] DONE  "
                    f"method={method} dim={dim} seed={seed} "
                    f"val_mse={val_metrics['val_mse']:.6f} test_mse={test_metrics['test_mse']:.6f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
            target_progress = cpu_progress if method in cpu_methods else accel_progress
            if target_progress is not None:
                target_progress.set_postfix({"method": method, "dim": dim, "seed": seed})
                target_progress.update(1)

        for method, dim, seed in job_specs:
            run_idx = run_index_map[(method, dim, seed)]
            method_cfg = copy.deepcopy(methods_cfg.get(method, {}))
            if method in neural_methods:
                method_cfg.setdefault("show_training_progress", show_training_progress)
                method_cfg.setdefault("training_progress_level", training_progress_level)
                method_cfg.setdefault("training_progress_leave", training_progress_leave)
                method_cfg.setdefault("training_progress_desc", f"{method} dim={dim} seed={seed}")
            run_dir = out / "runs" / method / f"dim_{dim:02d}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            device_label = _display_device_label(method, method_cfg)

            use_cpu_mp = bool(cpu_mp_workers > 1 and method in cpu_parallel_methods)
            use_gpu_mp = bool(
                gpu_mp_workers > 1
                and method in neural_methods
                and method in gpu_parallel_methods
                and _is_accelerator_device(str(method_cfg.get("device", "")))
            )

            if use_cpu_mp:
                if show_run_logs:
                    print(
                        f"[core-run {run_idx}/{len(job_specs)}] QUEUED "
                        f"method={method} dim={dim} seed={seed} device={device_label} mode=cpu-mp",
                        flush=True,
                    )
                pending_cpu_mp_jobs.append(
                    {
                        "method": method,
                        "dim": dim,
                        "seed": seed,
                        "run_idx": run_idx,
                        "run_dir": str(run_dir),
                        "method_cfg": copy.deepcopy(method_cfg),
                    }
                )
                continue

            if use_gpu_mp:
                if show_run_logs:
                    print(
                        f"[core-run {run_idx}/{len(job_specs)}] QUEUED "
                        f"method={method} dim={dim} seed={seed} device={device_label} mode=gpu-mp",
                        flush=True,
                    )
                pending_gpu_mp_jobs.append(
                    {
                        "method": method,
                        "dim": dim,
                        "seed": seed,
                        "run_idx": run_idx,
                        "run_dir": str(run_dir),
                        "method_cfg": copy.deepcopy(method_cfg),
                    }
                )
                continue

            serial_jobs.append(
                {
                    "method": method,
                    "dim": dim,
                    "seed": seed,
                    "run_idx": run_idx,
                    "run_dir": str(run_dir),
                    "method_cfg": copy.deepcopy(method_cfg),
                    "device_label": device_label,
                }
            )

        if pending_cpu_mp_jobs:
            if show_run_logs:
                print(
                    f"[core-stage] running {len(pending_cpu_mp_jobs)} CPU jobs with multiprocessing "
                    f"(workers={cpu_mp_workers})",
                    flush=True,
                )
            ctx = mp.get_context("spawn")
            shared_payload = {
                "x_train": prepared.train_x,
                "x_val": prepared.val_x,
                "x_test": prepared.test_x,
                "labels_val": prepared.dataset.val.labels,
                "labels_test": prepared.dataset.test.labels,
                "cell_ids_val": prepared.dataset.val.cell_ids,
                "cell_ids_test": prepared.dataset.test.cell_ids,
                "marker_names": prepared.dataset.markers,
            }
            with ctx.Pool(
                processes=cpu_mp_workers,
                initializer=_init_core_job_worker,
                initargs=(shared_payload,),
            ) as pool:
                for payload in pool.imap_unordered(_execute_core_job, pending_cpu_mp_jobs):
                    _consume_completed(payload)

        if pending_gpu_mp_jobs:
            if show_run_logs:
                print(
                    f"[core-stage] running {len(pending_gpu_mp_jobs)} accelerator jobs with multiprocessing "
                    f"(workers={gpu_mp_workers})",
                    flush=True,
                )
            ctx = mp.get_context("spawn")
            shared_payload = {
                "x_train": prepared.train_x,
                "x_val": prepared.val_x,
                "x_test": prepared.test_x,
                "labels_val": prepared.dataset.val.labels,
                "labels_test": prepared.dataset.test.labels,
                "cell_ids_val": prepared.dataset.val.cell_ids,
                "cell_ids_test": prepared.dataset.test.cell_ids,
                "marker_names": prepared.dataset.markers,
            }
            with ctx.Pool(
                processes=gpu_mp_workers,
                initializer=_init_core_job_worker,
                initargs=(shared_payload,),
            ) as pool:
                for payload in pool.imap_unordered(_execute_core_job, pending_gpu_mp_jobs):
                    _consume_completed(payload)

        if serial_jobs and show_run_logs:
            print(f"[core-stage] running {len(serial_jobs)} MPS/GPU+serial jobs", flush=True)

        for job in serial_jobs:
            method = str(job["method"])
            dim = int(job["dim"])
            seed = int(job["seed"])
            run_idx = int(job["run_idx"])
            method_cfg = dict(job["method_cfg"])
            if show_run_logs:
                print(
                    f"[core-run {run_idx}/{len(job_specs)}] START "
                    f"method={method} dim={dim} seed={seed} device={job['device_label']}",
                    flush=True,
                )
            run_t0 = time.time()
            method_impl = reg[method]
            set_seed(seed)
            result = method_impl.run(
                x_train=prepared.train_x,
                x_val=prepared.val_x,
                x_test=prepared.test_x,
                labels_val=prepared.dataset.val.labels,
                labels_test=prepared.dataset.test.labels,
                cell_ids_val=prepared.dataset.val.cell_ids,
                cell_ids_test=prepared.dataset.test.cell_ids,
                marker_names=prepared.dataset.markers,
                seed=seed,
                representation_dim=dim,
                config=method_cfg,
            )
            _consume_completed(
                {
                    "method": method,
                    "dim": dim,
                    "seed": seed,
                    "run_idx": run_idx,
                    "run_t0": run_t0,
                    "run_dir": str(job["run_dir"]),
                    "result": result,
                }
            )
    finally:
        if cpu_progress is not None:
            cpu_progress.close()
        if accel_progress is not None:
            accel_progress.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "reports" / "core_run_summary.csv", index=False)
    return runs, summary_df


def _save_per_class_tables(run_dir: Path, labels: np.ndarray | None, mse: np.ndarray, nll: np.ndarray) -> None:
    table = per_class_reconstruction_frame(labels=labels, nll_per_cell=nll, mse_per_cell=mse)
    table.to_csv(run_dir / "test_per_class_reconstruction.csv", index=False)


def _fallback_param_count(result: MethodRunResult) -> int:
    count = 0
    if result.components_mean is not None:
        count += int(result.components_mean.size)
    if result.components_var is not None:
        count += int(result.components_var.size)
    return count


def _execute_core_job(job: dict[str, Any]) -> dict[str, Any]:
    if _MP_SHARED is None:
        raise RuntimeError("Multiprocessing shared payload was not initialized")
    run_t0 = time.time()
    method = str(job["method"])
    seed = int(job["seed"])
    dim = int(job["dim"])
    reg = method_registry()
    method_impl = reg[method]
    set_seed(seed)
    result = method_impl.run(
        x_train=np.asarray(_MP_SHARED["x_train"]),
        x_val=np.asarray(_MP_SHARED["x_val"]),
        x_test=np.asarray(_MP_SHARED["x_test"]),
        labels_val=_MP_SHARED["labels_val"],
        labels_test=_MP_SHARED["labels_test"],
        cell_ids_val=_MP_SHARED["cell_ids_val"],
        cell_ids_test=_MP_SHARED["cell_ids_test"],
        marker_names=list(_MP_SHARED["marker_names"]),
        seed=seed,
        representation_dim=dim,
        config=dict(job["method_cfg"]),
    )
    return {
        "method": method,
        "dim": dim,
        "seed": seed,
        "run_idx": int(job["run_idx"]),
        "run_t0": None,
        "run_elapsed": float(time.time() - run_t0),
        "run_dir": str(job["run_dir"]),
        "result": result,
    }


def _display_device_label(method: str, method_cfg: dict[str, Any]) -> str:
    if method in {"nmf", "classical_archetypes"}:
        return "cpu(sklearn)"
    return str(method_cfg.get("device", "cpu"))


def _is_accelerator_device(device_label: str) -> bool:
    label = str(device_label).strip().lower()
    return label.startswith("cuda") or label.startswith("mps")


def _init_core_job_worker(shared_payload: dict[str, Any]) -> None:
    global _MP_SHARED
    _MP_SHARED = shared_payload


def _downsample_bundle_if_requested(bundle: DatasetBundle, dataset_cfg: dict[str, Any], seed: int) -> DatasetBundle:
    frac = _resolve_downsample_fraction(dataset_cfg)
    if frac >= 1.0:
        return bundle

    rng = np.random.default_rng(seed)
    train = _downsample_split(bundle.train, frac=frac, rng=rng, seed=seed + 11)
    val = _downsample_split(bundle.val, frac=frac, rng=rng, seed=seed + 13)
    test = _downsample_split(bundle.test, frac=frac, rng=rng, seed=seed + 17)
    print(
        "Downsampling applied: "
        f"fraction={frac:.4f} "
        f"train={len(bundle.train.x)}->{len(train.x)} "
        f"val={len(bundle.val.x)}->{len(val.x)} "
        f"test={len(bundle.test.x)}->{len(test.x)}",
        flush=True,
    )
    return DatasetBundle(
        name=bundle.name,
        markers=bundle.markers,
        label_column=bundle.label_column,
        cell_id_column=bundle.cell_id_column,
        train=train,
        val=val,
        test=test,
        raw_frame=bundle.raw_frame,
    )


def _resolve_downsample_fraction(dataset_cfg: dict[str, Any]) -> float:
    fraction = dataset_cfg.get("downsample_fraction")
    factor = dataset_cfg.get("downsample_factor")
    if fraction is not None:
        frac = float(fraction)
    elif factor is not None:
        f = float(factor)
        if f <= 0:
            raise ValueError("dataset.downsample_factor must be > 0")
        frac = 1.0 / f
    else:
        frac = 1.0

    if frac <= 0:
        raise ValueError("Downsample fraction must be > 0")
    return min(frac, 1.0)


def _downsample_split(split: Any, frac: float, rng: np.random.Generator, seed: int):
    n = len(split.x)
    if n == 0 or frac >= 1.0:
        return split

    n_keep = max(1, int(round(n * frac)))
    if n_keep >= n:
        return split

    indices = np.arange(n)
    if split.labels is not None:
        try:
            keep_idx, _ = train_test_split(
                indices,
                train_size=n_keep,
                random_state=seed,
                stratify=split.labels,
            )
        except Exception:
            keep_idx = rng.choice(indices, size=n_keep, replace=False)
    else:
        keep_idx = rng.choice(indices, size=n_keep, replace=False)
    keep_idx = np.sort(np.asarray(keep_idx, dtype=int))

    labels = None if split.labels is None else split.labels[keep_idx]
    return type(split)(
        x=split.x[keep_idx],
        cell_ids=split.cell_ids[keep_idx],
        labels=labels,
        frame=split.frame.iloc[keep_idx].reset_index(drop=True),
    )

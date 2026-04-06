from __future__ import annotations

from pathlib import Path

import pandas as pd

from cytof_archetypes.experiments.common import BenchmarkRun


def run_auxiliary_representation_models(runs: list[BenchmarkRun], config: dict, output_root: str | Path) -> pd.DataFrame:
    out = Path(output_root)
    tables_dir = out / "tables"
    reports_dir = out / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    aux_cfg = config.get("auxiliary_models", {})
    enabled = bool(aux_cfg.get("enabled", False))

    if not enabled:
        (reports_dir / "auxiliary_models_note.txt").write_text(
            "Auxiliary representation models are disabled. Core deconvolution experiments remain primary.\n",
            encoding="utf-8",
        )
        return pd.DataFrame()

    rows: list[dict[str, float | str | int]] = []
    for run in runs:
        if run.method in {"ae", "vae"}:
            rows.append(
                {
                    "method": run.method,
                    "seed": run.seed,
                    "latent_dim": run.representation_dim,
                    "val_mse": run.val_metrics["val_mse"],
                    "test_mse": run.test_metrics["test_mse"],
                    "test_nll": run.test_metrics["test_nll"],
                    "note": "DeepSets/Transformer/LSTM placeholders can be plugged via method registry.",
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "auxiliary_representation_models_summary.csv", index=False)
    return df

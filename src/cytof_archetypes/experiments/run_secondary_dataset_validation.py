from __future__ import annotations

from pathlib import Path

import pandas as pd

from cytof_archetypes.experiments.common import prepare_data, run_core_benchmark


def run_secondary_dataset_validation(config: dict, output_root: str | Path) -> pd.DataFrame:
    out = Path(output_root)
    secondary_cfg = config.get("secondary_dataset")
    if not secondary_cfg or not secondary_cfg.get("enabled", False):
        (out / "reports").mkdir(parents=True, exist_ok=True)
        (out / "reports" / "secondary_dataset_note.txt").write_text(
            "Secondary dataset run skipped (secondary_dataset.enabled is false).\n",
            encoding="utf-8",
        )
        return pd.DataFrame()

    prepared = prepare_data(
        dataset_cfg=secondary_cfg["dataset"],
        preprocessing_cfg=config.get("preprocessing", {}),
        seed=int(config.get("seed", 42)),
    )

    secondary_suite_cfg = {
        "seeds": secondary_cfg.get("seeds", config.get("seeds", [13])),
        "sweeps": {
            "k_values": secondary_cfg.get("k_values", config.get("sweeps", {}).get("k_values", [4, 6, 8, 10])),
            "latent_dims": secondary_cfg.get("latent_dims", config.get("sweeps", {}).get("latent_dims", [4, 6, 8, 10])),
        },
        "methods": config.get("methods", {}),
        "method_order": secondary_cfg.get(
            "method_order",
            ["nmf", "classical_archetypes", "deterministic_archetypal_ae", "probabilistic_archetypal_ae"],
        ),
    }

    _, summary_df = run_core_benchmark(
        prepared=prepared,
        suite_cfg=secondary_suite_cfg,
        output_root=out / "secondary_dataset",
    )
    summary_df.to_csv(out / "tables" / "secondary_dataset_summary.csv", index=False)
    return summary_df

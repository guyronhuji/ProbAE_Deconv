from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from cytof_archetypes.notebook_generation import templates


def generate_benchmark_notebooks(output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mapping = {
        "00_dataset_overview.ipynb": templates.notebook_00_dataset_overview(),
        "01_fit_vs_complexity.ipynb": templates.notebook_01_fit_vs_complexity(),
        "02_deconvolution_quality.ipynb": templates.notebook_02_deconvolution_quality(),
        "03_component_biology.ipynb": templates.notebook_03_component_biology(),
        "04_deterministic_vs_probabilistic.ipynb": templates.notebook_04_deterministic_vs_probabilistic(),
        "05_k_selection.ipynb": templates.notebook_05_k_selection(),
        "06_secondary_dataset_validation.ipynb": templates.notebook_06_secondary_dataset_validation(),
        "07_auxiliary_representation_models.ipynb": templates.notebook_07_auxiliary_representation_models(),
        "08_test_suite_runner.ipynb": templates.notebook_08_test_suite_runner(),
        "09_full_experiment_suite_runner.ipynb": templates.notebook_09_full_experiment_suite_runner(),
    }
    created: list[Path] = []
    for filename, notebook in mapping.items():
        target = out / filename
        nbf.write(notebook, target)
        created.append(target)
    return created


def generate_notebooks(output_dir: str | Path) -> list[Path]:
    # Backward-compatible alias.
    return generate_benchmark_notebooks(output_dir)

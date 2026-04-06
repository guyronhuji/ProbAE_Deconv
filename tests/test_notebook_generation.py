from pathlib import Path

from cytof_archetypes.notebook_generation import generate_notebooks


def test_generate_notebooks(tmp_path: Path):
    created = generate_notebooks(tmp_path)
    names = sorted(path.name for path in created)
    assert names == [
        "00_dataset_overview.ipynb",
        "01_fit_vs_complexity.ipynb",
        "02_deconvolution_quality.ipynb",
        "03_component_biology.ipynb",
        "04_deterministic_vs_probabilistic.ipynb",
        "05_k_selection.ipynb",
        "06_secondary_dataset_validation.ipynb",
        "07_auxiliary_representation_models.ipynb",
        "08_test_suite_runner.ipynb",
        "09_full_experiment_suite_runner.ipynb",
    ]
    for path in created:
        assert path.exists()

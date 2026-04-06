# Notebook Policy

This directory holds curated notebooks and dedicated experiment-suite notebooks.

The experiment suite writes its generated notebooks to:

- `notebooks/experiment_suite/`

You can change that location via `notebook_output_dir` in `configs/experiment_suite.yaml`.

For standalone notebook generation, use:

```bash
cytof-archetypes-generate-notebooks --output-dir notebooks/experiment_suite
```

Utility notebook included:

- `08_test_suite_runner.ipynb` to run all tests with progress bars and failure summaries.
- `09_full_experiment_suite_runner.ipynb` to run the full method suite and generate reports.
  : supports `DOWNSAMPLE_FACTOR` for faster trial runs.
  : supports `CPU_MULTIPROCESS_WORKERS` for parallel CPU baseline runs.

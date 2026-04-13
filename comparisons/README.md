# Levine32 Deconvolution Benchmark

This directory contains a reproducible benchmark comparing deconvolution methods on Levine32:

1. NMF
2. Classical archetypal analysis
3. Deterministic archetypal autoencoder
4. Probabilistic archetypal autoencoder

## Dataset
- Source file: `/Users/ronguy/Dropbox/Work/CyTOF/Experiments/ProbAE_Deconv/comparisons/outputs_tmp_progress_check/datasets/levine32_labeled_processed.h5ad`
- Raw cells in source: `265627`
- Labeled cells kept: `104184`
- Unlabeled cells removed: `161443`
- Markers: `32`

## Preprocessing
- Input already appears arcsinh(cofactor=5) + z-score normalized in source preprocessing logs.
- No extra clipping was applied in this benchmark.
- Shared representation for all methods: labeled marker matrix as provided in source.
- NMF nonnegative requirement: train-min shift saved in `outputs/datasets/representations/nmf_nonnegative_representation.npz`.

## Splits
- Train/val/test: `0.70/0.15/0.15`
- Stratified by label
- Fixed split seed: `42`
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

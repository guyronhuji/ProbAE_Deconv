# cytof_archetypes Package Reference

This file is the main technical documentation for the `cytof_archetypes` package.
It describes the full modeling pipeline, mathematical formulation, configuration schema, input/output contracts, and public API.

## 1. Package Purpose

`cytof_archetypes` is a Python package for archetypal deconvolution of single-cell CyTOF data, centered on a probabilistic archetypal autoencoder (PAE). It supports:

- Single-run training/evaluation of the PAE model (`train_from_config`, `evaluate_run_dir`)
- Baseline methods (NMF, classical archetypes, deterministic archetypal AE, AE, VAE)
- Reproducible benchmark orchestration with multiprocessing
- Evaluation metrics and plotting utilities
- Notebook generation for post-hoc analysis

Supported Python: `>=3.10`  
Current package version: `0.1.0`

## 2. High-Level Architecture

Main source tree:

- `cytof_archetypes.config`: default config, merging, device resolution
- `cytof_archetypes.datasets`: dataset loading/splitting (Levine32-style and compatible secondary datasets)
- `cytof_archetypes.preprocessing`: marker transforms/normalization
- `cytof_archetypes.models`: core probabilistic archetypal model and losses
- `cytof_archetypes.training`: single-run train/evaluate loop
- `cytof_archetypes.baselines`: baseline model interfaces and implementations
- `cytof_archetypes.evaluation`: metrics, interpretability scores, embeddings, plots, reporting
- `cytof_archetypes.experiments`: benchmark/suite orchestration
- `cytof_archetypes.notebook_generation`: generated analysis notebooks
- `cytof_archetypes.cli`: command-line entry points

## 3. Mathematical Formulation

### 3.1 Notation

- \(N\): number of cells
- \(M\): number of markers (features)
- \(K\): number of archetypes/components
- \(x_i \in \mathbb{R}^M\): observed marker vector for cell \(i\)
- \(w_i \in \Delta^{K-1}\): simplex mixture weights for cell \(i\)

Simplex constraint:

\[
w_{ik} \ge 0,\quad \sum_{k=1}^{K} w_{ik} = 1
\]

### 3.2 Encoder (all decoder families)

The encoder is an MLP producing logits \(z_i \in \mathbb{R}^K\), followed by softmax:

\[
w_i = \mathrm{softmax}(z_i)
\]

This enforces convex-mixture weights.

### 3.3 Gaussian Decoder Family

Parameters:

- Archetype means \(A_\mu \in \mathbb{R}^{K \times M}\)
- Archetype log-variances \(A_{\log \sigma^2} \in \mathbb{R}^{K \times M}\)

Per cell:

\[
\mu_i = w_i A_\mu,\quad \log \sigma_i^2 = w_i A_{\log \sigma^2}
\]

Per-cell Gaussian negative log-likelihood:

\[
\mathcal{L}^{(i)}_{\text{gauss}} =
\frac{1}{2}\sum_{m=1}^{M}\left[
\log \sigma_{im}^2 +
\frac{(x_{im}-\mu_{im})^2}{\sigma_{im}^2} +
\log(2\pi)
\right]
\]

### 3.4 Negative-Binomial (NB) Decoder Family

Parameters:

- Archetype marker logits \(A_{\text{logits}} \in \mathbb{R}^{K \times M}\)
- Gene/marker dispersion parameter vector \(\log\theta \in \mathbb{R}^{M}\)

Intermediate quantities:

\[
\rho_k = \mathrm{softmax}(A_{\text{logits},k,:}),\quad
\rho_i = w_i \rho,\quad
\theta = \mathrm{softplus}(\log\theta)
\]

With library size \(l_i\) (observed or configured):

\[
\mu_i = l_i \cdot \rho_i
\]

NB log-likelihood (per marker) follows the implementation in `models.losses.nb_nll`, using \((x,\mu,\theta)\) and summed across markers.

### 3.5 Total Training Objective

For both families, training uses weighted sum:

\[
\mathcal{L} =
\lambda_{\text{recon}} \mathcal{L}_{\text{recon}}
+ \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}
+ \lambda_{\text{div}} \mathcal{L}_{\text{div}}
+ \lambda_{\text{var}} \mathcal{L}_{\text{var}}
\]

Where:

- Reconstruction:
  - Gaussian: `gaussian_nll`
  - NB: `nb_nll`
- Entropy penalty (mean over cells):

\[
\mathcal{L}_{\text{ent}} = \frac{1}{N}\sum_i \left(-\sum_k w_{ik}\log(w_{ik}+\epsilon)\right)
\]

- Diversity penalty (encourages orthogonal archetype basis):
  - Normalize archetypes row-wise to unit norm
  - Compute Gram matrix \(G\)
  - Penalize \(\|G-I\|_F^2\) (mean)
- Variance regularization (Gaussian only):

\[
\mathcal{L}_{\text{var}} = \mathrm{mean}\left(\exp(A_{\log \sigma^2})\right)
\]

## 4. Data Contracts (Inputs)

### 4.1 Supported input file formats

- `.h5ad` (requires optional dependency `anndata`)
- `.csv`, `.tsv`, `.txt` tabular matrices

### 4.2 Required/optional columns

- Marker columns:
  - Explicitly given via `dataset.marker_columns`, or
  - Automatically inferred as numeric columns excluding label/cell-id columns
- Cell IDs:
  - `dataset.cell_id_column` (default `cell_id`)
  - Auto-generated if missing
- Labels:
  - `dataset.label_column` (default `label`)
  - Optional; used for stratified splitting and label-aware metrics

### 4.3 Splitting behavior

Given `val_fraction`, `test_fraction`:

1. Split full data into train+val vs test
2. Split train+val into train vs val using adjusted ratio
3. Stratified splitting is used only when labels exist and each class has at least 2 cells

### 4.4 Preprocessing

`MarkerPreprocessor` supports:

- `transform`: `none` or `arcsinh` (`arcsinh(x / cofactor)`)
- `normalization`: `zscore`, `robust_zscore`, or `none`
- Optional clipping: `clip_min`, `clip_max`

Gaussian pipeline:

- `x_encoder = x_target = preprocessed markers`

NB pipeline:

- `x_target = raw non-negative counts`
- `x_encoder` from `data.encoder_input`:
  - `raw_counts`
  - `log1p_normalized`: `log1p((x / library_size) * 1e4)`
- `library_size` from observed sum or `size_factor_key`

## 5. Tensor/Array Shapes

Core shapes in training and inference:

- Input matrix: `(n_cells, n_markers)`
- Encoder logits: `(n_cells, n_archetypes)`
- Weights: `(n_cells, n_archetypes)` (row-sum = 1)
- Gaussian decoder output:
  - `recon`: `(n_cells, n_markers)`
  - `logvar`: `(n_cells, n_markers)`
- NB decoder output:
  - `mu`: `(n_cells, n_markers)`
  - `theta`: `(n_cells, n_markers)` after broadcast
- Beta-binomial decoder output:
  - `probs`: `(n_cells, n_markers)`
  - `concentration`: `(n_cells, n_markers)` after broadcast

## 6. Configuration Reference

`load_config()` merges user YAML onto `DEFAULT_CONFIG`.

### 6.1 Single-run config keys (`train_from_config`)

- `seed` (`int`)
- `device` (`"cpu"`, `"cuda"`, `"mps"`, `"auto"`)
- `dataset` (`dict`)
  - `name`, `input_path`, `marker_columns`, `label_column`, `cell_id_column`, `val_fraction`, `test_fraction`
- `preprocessing` (`dict`)
  - `transform`, `arcsinh_cofactor`, `normalization`, `clip_min`, `clip_max`
- `model` (`dict`)
  - `decoder_family`: `gaussian`, `nb`, or `beta_binomial`
  - `n_archetypes`, `encoder_hidden_dims`, `activation`, `dropout`
  - NB-specific: `use_observed_library_size`, `size_factor_key`, `dispersion` (`gene`)
- `data` (`dict`)
  - `encoder_input` (`raw_counts` or `log1p_normalized`)
  - `decoder_target` (NB and beta-binomial currently require `raw_counts`)
- `loss` (`dict`)
  - `reconstruction_weight` (optional, default 1.0)
  - `entropy_reg_weight`, `diversity_reg_weight`, `variance_reg_weight`
- `training` (`dict`)
  - `batch_size`, `lr`, `weight_decay`, `max_epochs`, `patience`, `grad_clip`
  - `progress_bar`, `progress_bar_leave`, `progress_bar_desc`
  - `mixed_precision` exists in config but is currently not used by the trainer
- `output` (`dict`)
  - `base_dir`, `run_name`

Notes:

- `model.type` and `loss.type` are config metadata fields; control flow is determined primarily by `model.decoder_family`.
- In `training.trainer._prepare_data`, loading is currently wired to `load_levine32_bundle` directly.

### 6.2 Suite config keys (`run_experiment_suite`)

Major keys:

- `seed`, `output_dir`, `notebook_output_dir`
- progress controls (`show_progress`, `show_run_logs`, training progress options)
- multiprocessing controls:
  - `cpu_multiprocessing_workers`
  - `gpu_multiprocessing_workers`
  - `cpu_parallel_methods`
  - `gpu_parallel_methods`
- `dataset`, `preprocessing`, `seeds`, `sweeps`
- `methods` (method-specific hyperparameters)
- optional `secondary_dataset`, `auxiliary_models`, `marker_groups`

Multiprocessing is implemented with `multiprocessing.get_context("spawn").Pool(...)` in benchmark execution.

## 7. Runtime Outputs (Single Training Run)

`train_from_config()` writes a run directory, either named (`output.run_name`) or auto-incremented (`run_###`).

Typical structure:

- `config_resolved.yaml`: full resolved config
- `preprocessor.json`: fitted preprocessing state or count-decoder bypass marker
- `model_summary.txt`: architecture and parameter counts
- `training_log.csv`: per-epoch metrics
- `training_summary.json`: training summary
- `best_checkpoint.pt`, `final_checkpoint.pt`
- `README_run.md`: concise run summary
- `plots/`
  - loss curve and archetype visualizations
- `archetypes/`
  - gaussian: means/logvars/vars (`csv` + `npy`)
  - nb: logits, gene fractions, gene dispersion
  - beta-binomial: logits, gene fractions, gene concentration
- `weights/`
  - split-level cell weights, full `cell_weights.csv`, class means
- `metrics/`
  - split metrics JSON and diagnostics tables
- `embeddings/`
  - `archetype_weight_embedding_<split>.npz` with PCA/UMAP embeddings

`evaluate_run_dir()` reloads config/checkpoint and regenerates final artifacts for the requested checkpoint.

## 8. Public Python API

Below are the major callable APIs intended for package-level use.

### 8.1 `cytof_archetypes.config`

- `deep_update(base: dict, updates: dict) -> dict`
- `load_config(path: str | Path | None) -> dict`
- `save_config(config: dict, path: str | Path) -> None`
- `resolve_device(config: dict) -> str`

### 8.2 `cytof_archetypes.datasets`

- Dataclasses:
  - `SplitData`
  - `Levine32Bundle`
  - `DatasetBundle`
- Loaders:
  - `load_levine32_bundle(...) -> Levine32Bundle`
  - `load_dataset_bundle(dataset_cfg: dict, seed: int) -> DatasetBundle`
  - `split_manifest_frame(bundle: DatasetBundle) -> pd.DataFrame`

### 8.3 `cytof_archetypes.preprocessing`

- `MarkerPreprocessor`
  - `fit(x)`, `transform_array(x)`, `fit_transform(x)`
  - `state_dict()`, `from_state_dict(state)`

### 8.4 `cytof_archetypes.models`

- `ProbabilisticArchetypalAutoencoder`
  - `encode_logits(x)`, `encode(x)`, `decode_params(weights)`, `forward(x, library_size=None)`
- Losses:
  - `gaussian_nll(target, mean, logvar, reduction="mean")`
  - `nb_nll(x, mu, theta, reduction="mean")`
  - `entropy_penalty(weights)`
  - `diversity_penalty(archetype_means_or_logits)`
  - `variance_regularization(archetype_logvars)`

### 8.5 `cytof_archetypes.training`

- `train_from_config(config: dict[str, Any]) -> Path`
- `evaluate_run_dir(run_dir: str | Path, checkpoint: str = "best_checkpoint.pt") -> Path`
- Callback:
  - `EarlyStopping(patience=15, min_delta=0.0)`

### 8.6 `cytof_archetypes.baselines`

- Interface/data contracts:
  - `BaseMethod`
  - `SplitResult`
  - `MethodRunResult`
  - `gaussian_nll_per_cell(...)`
  - `write_method_artifacts(...)`
- Implementations:
  - `NMFMethod`
  - `ClassicalArchetypeMethod`
  - `DeterministicArchetypalMethod`
  - `ProbabilisticArchetypalMethod`
  - `AEMethod`
  - `VAEMethod`

`BaseMethod.run(...) -> MethodRunResult` receives train/val/test arrays and metadata, and returns:

- component-level parameters (`components_mean`, `components_var`)
- per-split reconstructions, logvars, latent, and optional weights
- optional training history

### 8.7 `cytof_archetypes.evaluation`

- Reconstruction/statistical metrics:
  - `compute_metrics`
  - `reconstruction_metrics_per_cell`
  - `per_class_reconstruction_frame`
  - `representation_structure_metrics`
  - `bootstrap_mean_ci`, `paired_wilcoxon`, `benjamini_hochberg`
- Deconvolution metrics:
  - `per_cell_weight_entropy`
  - `dominant_component_stats`
  - `class_component_means`
  - `class_purity_of_dominant`
  - `class_profile_separation`
- Interpretability:
  - `combined_interpretability_score`
  - `marker_coherence_score`
  - `class_specificity_score`
  - `sparsity_score`
  - `entropy_sparsity_score`
  - marker profile/top-marker/enrichment table builders
- Output utilities:
  - `save_archetype_outputs`
  - `save_cell_weights`
  - `save_class_weight_summary`
  - `save_diagnostics`
  - `save_embeddings_npz`, `umap_fit_transform_large`
  - plotting and reporting helpers

### 8.8 `cytof_archetypes.experiments`

- Suite API:
  - `SUITE_DEFAULTS`
  - `load_suite_config(path) -> dict`
  - `run_experiment_suite(config) -> Path`
- Core benchmark API:
  - `prepare_data(dataset_cfg, preprocessing_cfg, seed) -> PreparedData`
  - `method_registry() -> dict[str, BaseMethod]`
  - `run_core_benchmark(prepared, suite_cfg, output_root) -> tuple[list[BenchmarkRun], pd.DataFrame]`
- Additional benchmark analysis functions:
  - `run_fit_vs_complexity`
  - `run_deconvolution_quality`
  - `run_deterministic_vs_probabilistic`
  - `run_k_selection`
  - `run_component_biology`
  - `run_rare_population_preservation`
  - `run_interpretability_tradeoff`
  - `run_secondary_dataset_validation`
  - `run_auxiliary_representation_models`

### 8.9 `cytof_archetypes.notebook_generation`

- `generate_benchmark_notebooks(output_dir) -> list[Path]`
- `generate_notebooks(output_dir) -> list[Path]` (alias)
- Template builders: `notebook_00_...` through `notebook_09_...`

### 8.10 `cytof_archetypes.io` and `utils`

- IO:
  - `ensure_dir(path) -> Path`
  - `write_json(payload, path)`
  - `read_json(path) -> dict`
  - `next_run_dir(base_dir) -> Path`
- Utils:
  - `set_seed(seed)`
  - `get_logger(name="cytof_archetypes", level=logging.INFO)`

## 9. Command-Line API

Installed entry points (from `pyproject.toml`):

- `cytof-archetypes-train`
  - calls `cytof_archetypes.cli.train_cli`
  - args: `--config <yaml>`
- `cytof-archetypes-evaluate`
  - calls `evaluate_cli`
  - args: `--run-dir <path> [--checkpoint best_checkpoint.pt]`
- `cytof-archetypes-generate-notebooks`
  - calls `generate_notebooks_cli`
  - args: `--output-dir <path>`
- `cytof-archetypes-demo`
  - calls `demo_cli`
  - args: `--config <yaml> [--max-epochs <int>]`
- `cytof-archetypes-run-suite`
  - calls `run_suite_cli`
  - args: `[--config <yaml>]`

Script wrappers in `scripts/` provide equivalent entry points plus dataset-preparation helpers.

## 10. Error Handling and Validation

Common validation behavior:

- Missing dataset path raises `FileNotFoundError`
- Invalid marker selection or non-inferable markers raises `ValueError`
- Unsupported transform/normalization/decoder family raises `ValueError`
- NB mode with unsupported `data.decoder_target` raises `ValueError`
- Calling preprocessor transform before `fit` raises `RuntimeError`

## 11. Optional Dependencies

- `anndata`:
  - required for `.h5ad` loading and dataset preparation scripts
- `umap-learn`:
  - used for UMAP embedding generation; if unavailable/failing, code falls back to zeros or PCA depending on call site

## 12. Reproducibility

Seed control (`set_seed`) covers:

- Python `random`
- NumPy RNG
- `PYTHONHASHSEED`
- PyTorch CPU/GPU seeding (when torch is installed)

Environment/provenance logs are saved in both training and suite workflows (`environment_log.json` files and resolved configs).

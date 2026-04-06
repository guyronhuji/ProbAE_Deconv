# cytof_archetypes

Full experiment suite for probabilistic archetypal deconvolution on CyTOF datasets (Levine32-first), with matched deconvolution and latent-model baselines.

## Scientific focus

Primary question:

Can single-cell CyTOF profiles be modeled more accurately and interpretably as mixtures of probabilistic archetypal programs than by deterministic archetypes, classical deconvolution, and generic latent-variable models?

Core claims tested:

1. Cells can be represented as mixtures of a small number of latent biological programs.
2. Probabilistic archetypes capture within-state heterogeneity better than deterministic archetypes.
3. Probabilistic archetypal AE gives a stronger fit/interpretability/deconvolution balance.
4. Small K can preserve dominant and rare biologically meaningful populations.

## Included methods

Mandatory methods implemented:

- NMF
- Classical archetypal analysis (deterministic convex-combination approximation)
- Deterministic archetypal autoencoder
- Probabilistic archetypal autoencoder (main model)
- Standard AE
- VAE

Optional hooks:

- Secondary dataset validation via config (`secondary_dataset`)
- Auxiliary representation models scaffold (`auxiliary_models`)

## Install

```bash
pip install -e .
```

## Run full suite

```bash
python scripts/run_experiment_suite.py --config configs/experiment_suite.yaml
```

Or:

```bash
cytof-archetypes-run-suite --config configs/experiment_suite.yaml
```

## RunPod workflow

### 1) One-time local tools

```bash
brew install runpod/runpodctl/runpodctl
```

Add your SSH public key in RunPod console (`Settings -> SSH Public Keys`), then start a pod from the RunPod UI.

### 2) One-time setup inside pod

```bash
cd /workspace
git clone <YOUR_REPO_URL> ProbAE_Deconv
cd ProbAE_Deconv
bash runpod/setup_pod.sh
```

### 3) Run your suite config inside pod

```bash
cd /workspace/ProbAE_Deconv
bash runpod/run_suite.sh --config configs/experiment_suite.yaml --send
```

Useful options:

- `--tag myrun`
- `--downsample-factor 10`
- `--output-dir outputs/my_runpod_run`
- `--no-send` (skip transfer)

`run_suite.sh` writes a temporary RunPod config and forces NN methods to `cuda`.

### 4) Fetch results locally

```bash
bash runpod/fetch_results.sh
```

It will prompt for the `runpodctl send` transfer code and extract results under `outputs/runpod/`.

### Optional: launch from local via SSH

```bash
bash runpod/run_remote.sh \
  --ssh-target <pod-user>@ssh.runpod.io \
  --identity ~/.ssh/id_ed25519 \
  --config configs/experiment_suite.yaml \
  --send
```

## Main outputs

`outputs/experiment_suite/` contains:

- `tables/fit_vs_complexity_summary.csv`
- `tables/deconvolution_quality_summary.csv`
- `tables/component_marker_profiles.csv`
- `tables/component_marker_enrichment.csv`
- `tables/deterministic_vs_probabilistic_summary.csv`
- `tables/per_class_method_metrics.csv`
- `tables/fit_vs_interpretability.csv`
- `tables/k_selection_summary.csv`
- `tables/class_component_means.csv`
- `tables/per_cell_weight_entropy.csv`
- publication-oriented plots in `plots/` (`.png` + `.pdf`)
- suite notebooks in `notebooks/experiment_suite/`
- reproducibility logs/manifests in `reports/`
- analysis-LLM docs in `docs/`

## Required notebooks generated programmatically

- `00_dataset_overview.ipynb`
- `01_fit_vs_complexity.ipynb`
- `02_deconvolution_quality.ipynb`
- `03_component_biology.ipynb`
- `04_deterministic_vs_probabilistic.ipynb`
- `05_k_selection.ipynb`
- `06_secondary_dataset_validation.ipynb`
- `07_auxiliary_representation_models.ipynb`
- `08_test_suite_runner.ipynb` (runs full pytest suite with progress bars)
- `09_full_experiment_suite_runner.ipynb` (runs all methods/sweeps and generates reports)

## Reproducibility

The suite saves:

- fixed seeds and resolved config
- split manifest
- preprocessor snapshot
- environment/version log
- per-run artifacts per method/K/seed
- bootstrap CI summaries and paired Wilcoxon outputs (with FDR correction where applicable)

## Notes

- The classical archetypal baseline uses a deterministic simplex-projection alternating optimization approximation to keep the baseline lightweight and reproducible.
- Secondary dataset support is scaffolded through config and dataset registry; Levine32 is fully wired by default.

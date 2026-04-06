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

Add your SSH public key in RunPod console (`Settings -> SSH Public Keys`).

Then create a pod from this repo:

```bash
bash runpod/create_pod.sh
```

If you prefer, keep your API key in `.runpodkey` (ignored by git), either as:

```text
apiKey <YOUR_KEY>
```

or just:

```text
<YOUR_KEY>
```

### 2) One-time setup inside pod

```bash
cd /workspace
git clone https://github.com/guyronhuji/ProbAE_Deconv.git ProbAE_Deconv
cd ProbAE_Deconv
bash runpod/setup_pod.sh
```

Prepare the dataset directly in the pod (recommended):

```bash
cd /workspace/ProbAE_Deconv
bash runpod/prepare_dataset.sh --output data/levine32_processed.h5ad
```

This writes:

```bash
/workspace/ProbAE_Deconv/data/levine32_processed.h5ad
```

If you prefer, you can copy a local dataset file into the pod at:

```bash
/workspace/ProbAE_Deconv/data/levine32_processed.h5ad
```

Example transfer with `runpodctl`:

```bash
# local machine
cd /Users/ronguy/Dropbox/Work/CyTOF/Experiments/ProbAE_Deconv
runpodctl send data/levine32_processed.h5ad

# in pod
cd /workspace/ProbAE_Deconv/data
runpodctl receive <TRANSFER_CODE>
```

### 3) Run your suite config inside pod

```bash
cd /workspace/ProbAE_Deconv
bash runpod/run_suite.sh --config configs/experiment_suite.yaml --send
```

Useful options:

- `--tag myrun`
- `--downsample-factor 10`
- `--dataset-path /workspace/ProbAE_Deconv/data/levine32_processed.h5ad`
- `--auto-prepare-dataset` (default, auto-download Levine32 if missing)
- `--dataset-force-download` (redownload dataset during auto-prepare)
- `--output-dir outputs/my_runpod_run`
- `--gpu-parallel auto` (or `--gpu-parallel 2`)
- `--gpu-mem-per-job-gb 12` (used by `--gpu-parallel auto`)
- `--no-send` (skip transfer)

`run_suite.sh` writes a temporary RunPod config, forces NN methods to `cuda`, and can run NN jobs in parallel on GPU when VRAM allows.

### 3b) Recommended: run with tmux (survives disconnect/logout)

Inside the pod:

```bash
cd /workspace/ProbAE_Deconv
tmux new -s probae_suite
bash runpod/run_suite.sh --config configs/experiment_suite.yaml --send --gpu-parallel auto
```

Detach while leaving run active:

- `Ctrl-b` then `d`

Reattach later:

```bash
tmux ls
tmux attach -t probae_suite
```

Run in background without attaching:

```bash
tmux new -d -s probae_suite "cd /workspace/ProbAE_Deconv && bash runpod/run_suite.sh --config configs/experiment_suite.yaml --send --gpu-parallel auto"
```

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

## Azure workflow

Two supported Azure paths:

1. Fresh Azure GPU VM deployment (recommended for interactive GPU runs)
2. Azure ML command job (recommended for managed runs)
3. Azure GPU VM + tmux on an existing VM

### 1) Fresh Azure GPU VM deployment

This path creates a brand new VM instance every run, installs GPU drivers, bootstraps this repo, and prints exactly how to connect and run the suite.

Prerequisites:

```bash
az login
```

Deploy a new instance:

```bash
cd /Users/ronguy/Dropbox/Work/CyTOF/Experiments/ProbAE_Deconv
bash azure/deploy_vm.sh \
  --resource-group <RG_NAME> \
  --location <AZURE_REGION>
```

Useful options:

- `--size Standard_NC8as_T4_v3`
- `--name probae-<custom-name>`
- `--repo-ref main`
- `--subscription <SUBSCRIPTION_ID>`
- `--bootstrap-log-path /home/azureuser/probae_bootstrap.log`
- `--local-bootstrap-log-dir outputs/azure_bootstrap_logs`
- `--no-bootstrap` (create VM only; skip setup)
- `--no-gpu-driver` (skip NVIDIA extension install)

Then SSH in and run:

```bash
ssh azureuser@<PUBLIC_IP>
cd ~/ProbAE_Deconv
source .venv/bin/activate
bash runpod/run_suite.sh --config configs/experiment_suite.yaml --no-send --gpu-parallel auto
```

Check bootstrap/setup logs:

```bash
# on the VM
tail -f ~/probae_bootstrap.log

# on local machine (saved by deploy script)
ls -1 outputs/azure_bootstrap_logs/
```

### 2) Azure ML command job

Prerequisites:

```bash
az login
az extension add -n ml
```

Submit job:

```bash
cd /Users/ronguy/Dropbox/Work/CyTOF/Experiments/ProbAE_Deconv
bash azure/submit_aml_job.sh \
  --resource-group <RG_NAME> \
  --workspace <AML_WORKSPACE_NAME> \
  --compute <AML_COMPUTE_NAME> \
  --config configs/experiment_suite.yaml
```

Useful options:

- `--subscription <SUBSCRIPTION_ID>`
- `--job-name probae-azure-001`
- `--downsample-factor 10`
- `--dataset-path <mounted_or_registered_dataset_path>`
- `--gpu-parallel auto` or `--gpu-parallel 2`
- `--gpu-mem-per-job-gb 12`
- `--no-stream`

Fetch outputs:

```bash
bash azure/fetch_results.sh \
  --resource-group <RG_NAME> \
  --workspace <AML_WORKSPACE_NAME> \
  --job-name <AML_JOB_NAME>
```

### 3) Azure GPU VM + tmux (existing VM)

Inside VM:

```bash
cd /workspace
git clone https://github.com/guyronhuji/ProbAE_Deconv.git ProbAE_Deconv
cd ProbAE_Deconv
bash azure/setup_vm.sh
```

Then run in tmux:

```bash
cd /workspace/ProbAE_Deconv
tmux new -s probae_suite
bash runpod/run_suite.sh --config configs/experiment_suite.yaml --no-send --gpu-parallel auto
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

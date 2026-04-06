# Probabilistic Archetypal Autoencoder for Levine32 — Codex Design Document

## Purpose

Build a **standalone Python package** that implements a **probabilistic archetypal autoencoder** for CyTOF data and automatically generates **testing / analysis notebooks** using the **Levine32** benchmark dataset.

The package should support:

- loading and preprocessing Levine32
- training a probabilistic archetypal autoencoder
- saving interpretable archetype outputs
- evaluating latent structure and reconstruction quality
- generating notebooks that test and interpret the model on Levine32

This is a new standalone package, not a patch to an existing training repo.

---

## Scientific framing

We want an autoencoder in which each cell is represented by **mixture weights over archetypes**, but the archetypes themselves are **probability distributions**, not deterministic point profiles.

### Why probabilistic archetypes?

A deterministic archetype says:

\[
x \approx D w
\]

where each archetype is a single marker vector.

A probabilistic archetype says instead:

\[
x \sim p(x \mid w, \Theta)
\]

where each archetype has at least:

- a mean marker profile
- a marker-wise variance profile

This is better suited to CyTOF, because even a biologically coherent cell state has within-state variability.

---

## Target dataset: Levine32

The initial package target is **Levine32**, a widely used CyTOF benchmark dataset. The package should assume:

- 32 protein markers
- 14 manually gated populations plus unassigned cells
- 2 healthy donors
- both labeled and unlabeled / unassigned cells may be present depending on the version loaded

Do not hardcode exact label names unless they are detected from the loaded dataset.

The package should support loading Levine32 from:

1. an `.h5ad` or processed local file
2. a standard tabular export
3. optionally a helper that downloads or converts from a known source format later

The first implementation may assume a local prepared file path.

---

## Core model

### High-level idea

For each cell with marker vector \(x \in \mathbb{R}^{p}\), the encoder outputs archetype mixture weights:

\[
w = \mathrm{softmax}(f_{\text{enc}}(x))
\]

with:

- \(w_k \ge 0\)
- \(\sum_k w_k = 1\)

So \(w\) is directly interpretable as archetype usage.

The decoder does **not** output a single deterministic reconstruction vector directly. Instead, it constructs a **probabilistic reconstruction distribution** from learned archetype distributions.

---

## Probabilistic decoder

### Archetype parameters

For \(K\) archetypes and \(p\) markers, learn:

- archetype means:
  \[
  M \in \mathbb{R}^{K \times p}
  \]
- archetype log-variances:
  \[
  S \in \mathbb{R}^{K \times p}
  \]

where:

- \(M_k\) is the mean vector of archetype \(k\)
- \(S_k\) is the log-variance vector of archetype \(k\)

### Cell-level decoder outputs

Given weights \(w\), define:

\[
\mu(x) = w M
\]

\[
s(x) = w S
\]

\[
\sigma^2(x) = \exp(s(x))
\]

Then the reconstructed distribution is:

\[
x \sim \mathcal{N}(\mu(x), \operatorname{diag}(\sigma^2(x)))
\]

This yields a **diagonal Gaussian probabilistic decoder**.

This is the default and required first implementation.

---

## Why diagonal Gaussian first?

It is the right first compromise because it is:

- probabilistic
- interpretable
- numerically manageable
- easy to train
- easy to explain biologically

Do **not** start with full covariance per archetype.

Low-rank covariance can be considered later, but it is not required in v1.

---

## Loss function

The base reconstruction loss is the Gaussian negative log-likelihood:

\[
\mathcal{L}_{\text{nll}} =
\frac{1}{2}\sum_{m=1}^{p}
\left[
\log \sigma_m^2(x)
+
\frac{(x_m - \mu_m(x))^2}{\sigma_m^2(x)}
\right]
\]

Average over cells.

### Regularization terms

#### 1. Entropy penalty on weights

Encourage sparse archetype usage:

\[
\mathcal{L}_{\text{entropy}} =
-\frac{1}{N}\sum_i \sum_k w_{ik}\log(w_{ik} + \epsilon)
\]

Use a positive coefficient and **minimize** this term to reduce entropy.

#### 2. Diversity penalty on archetype means

Prevent archetype collapse.

Normalize mean archetypes by column / row appropriately and penalize cosine similarity:

\[
\mathcal{L}_{\text{diversity}} =
\left\| G - I \right\|_F^2
\]

where \(G\) is the cosine-similarity Gram matrix of normalized archetype mean vectors.

#### 3. Optional variance regularization

Prevent pathological variances:
- extremely tiny variances
- extremely huge variances

Add a mild penalty if needed, for example on mean log-variance magnitude.

### Total loss

\[
\mathcal{L} =
\mathcal{L}_{\text{nll}}
+ \lambda_{\text{entropy}}\mathcal{L}_{\text{entropy}}
+ \lambda_{\text{diversity}}\mathcal{L}_{\text{diversity}}
+ \lambda_{\text{var}}\mathcal{L}_{\text{var}}
\]

---

## Model variants

The package should support at least these model variants.

### Variant A — Base probabilistic archetypal autoencoder
- simplex weights
- archetype means
- archetype diagonal variances
- Gaussian NLL loss

This is the required default.

### Variant B — Deterministic fallback
Optional:
- same simplex weights
- archetype means only
- MSE reconstruction

Useful for debugging / ablations.

### Variant C — Tiny nonlinear post-decoder (optional later)
Optional later:
- compute mean mixture from archetypes
- pass through tiny MLP before outputting mean / variance

Do **not** implement this unless the linear-probabilistic version is working.

The design should leave room for it, but v1 should prioritize the linear probabilistic decoder.

---

## Encoder architecture

Default encoder:

\[
p \rightarrow 128 \rightarrow 64 \rightarrow K
\]

with:
- Linear layers
- ReLU or GELU
- optional dropout

Final logits are mapped to simplex weights using softmax.

Configurable:
- number of hidden layers
- hidden dims
- activation
- dropout

Keep the encoder moderate. Do not make it so powerful that the archetypes become less meaningful.

---

## Number of archetypes

This should be configurable.

Recommended starting sweep for Levine32:

- 4
- 6
- 8
- 10

Default:
- 8 archetypes

The package should support a simple experiment sweep over `n_archetypes`.

---

## Biological outputs to support

The model must save outputs that are directly interpretable.

### 1. Archetype mean matrix
Save:
- `archetype_means.csv`
- `archetype_means.npy`

Rows:
- archetypes

Columns:
- markers

### 2. Archetype variance matrix
Save:
- `archetype_logvars.csv`
- `archetype_vars.csv`
- `.npy` versions

### 3. Pure archetype profiles
For each archetype \(k\), decode the one-hot simplex vector \(e_k\) and save:
- pure mean profile
- pure variance profile

### 4. Cell-wise archetype weights
Save per-cell:
- cell ID
- label if available
- archetype weights

### 5. Class-level average archetype weights
For labeled cells:
- average archetype weights per class
- standard deviations per class

### 6. Reconstruction diagnostics
Save per cell:
- NLL
- squared error
- optional standardized residuals

---

## Interpretation goals on Levine32

The package and notebooks should support interpretation in terms relevant to Levine32-like hematopoietic / immune structure.

Possible biological groupings to comment on, only if supported by detected labels and markers:

- T-cell-like structure
- B-cell-like structure
- NK-like structure
- monocyte / myeloid structure
- progenitor / stem-like structure
- memory / naive separation if relevant markers are present
- activation / signaling variation if relevant markers are present

Do not invent biological claims unsupported by the data.

---

## Standalone package requirements

This must be a standalone installable Python package.

### Suggested package name

Use something like:

- `probabilistic_archetypes_cytof`
- or `cytof_archetypes`

Choose one clean, non-conflicting name.

### Packaging

Use:
- `pyproject.toml`
- `src/` layout
- installable package
- command-line entry points

---

## Suggested directory structure

```text
package_root/
  README.md
  pyproject.toml
  LICENSE
  .gitignore

  src/
    cytof_archetypes/
      __init__.py
      config.py
      io.py
      preprocessing.py
      datasets/
        __init__.py
        levine32.py
      models/
        __init__.py
        probabilistic_archetypal_ae.py
        losses.py
      training/
        __init__.py
        trainer.py
        callbacks.py
      evaluation/
        __init__.py
        metrics.py
        embeddings.py
        archetypes.py
        plots.py
      notebook_generation/
        __init__.py
        templates.py
        writer.py
      utils/
        __init__.py
        logging.py
        reproducibility.py

  configs/
    base.yaml
    levine32_default.yaml
    levine32_archetype_sweep.yaml

  scripts/
    train.py
    evaluate.py
    run_levine32_demo.py
    generate_notebooks.py

  notebooks/
    00_dataset_inspection.ipynb
    01_train_single_model.ipynb
    02_interpret_archetypes.ipynb
    03_sweep_n_archetypes.ipynb

  tests/
    test_model_shapes.py
    test_simplex_weights.py
    test_losses.py
    test_levine32_loader.py
    test_notebook_generation.py

  docs/
    DESIGN.md
    README_for_analysis_llms.md
    ANALYSIS_LLM_INSTRUCTIONS.md
```

---

## Package functionality

The standalone package must implement the following capabilities.

### 1. Data loading
Implement a Levine32 dataset loader that:
- loads a prepared file
- validates marker columns
- validates labels if present
- returns a standard object or pandas/anndata structure
- supports train/val/test splits

### 2. Preprocessing
Support:
- arcsinh transform if needed
- z-score or robust z-score
- optional marker clipping
- selection of marker columns
- split-aware normalization where appropriate

### 3. Training
Implement:
- mini-batch training
- early stopping
- checkpointing
- training summaries
- reproducible seeds

### 4. Evaluation
Implement:
- reconstruction metrics
- latent embedding extraction
- archetype output saving
- class-level summaries if labels available
- UMAP / PCA plotting
- correlation between archetype weights and labels / markers

### 5. Notebook generation
The package must **automatically generate testing notebooks** for Levine32.

This is a key requirement.

---

## Notebook generation requirements

Implement notebook generation utilities that create notebooks programmatically.

### Required notebooks

#### `00_dataset_inspection.ipynb`
- load Levine32
- summarize cells, labels, markers
- inspect class frequencies
- inspect marker distributions
- confirm preprocessing

#### `01_train_single_model.ipynb`
- load config
- train one probabilistic archetypal autoencoder
- show training curves
- inspect learned archetypes

#### `02_interpret_archetypes.ipynb`
- load trained model
- display archetype means and variances
- visualize per-class average weights
- show pure archetype profiles
- show UMAP of archetype weights
- discuss biology

#### `03_sweep_n_archetypes.ipynb`
- compare several `n_archetypes` settings
- summarize reconstruction metrics
- compare class-level archetype usage
- recommend a practical default

### Notebook generation implementation
Use either:
- `nbformat`
- or a light template-based notebook writer

The notebooks should be **generated files**, not hand-maintained copies.

They must be runnable after path/config editing.

---

## Command-line interface

Add CLI entry points.

### `cytof-archetypes-train`
Train one model:

```bash
cytof-archetypes-train --config configs/levine32_default.yaml
```

### `cytof-archetypes-evaluate`
Evaluate a trained model:

```bash
cytof-archetypes-evaluate --run-dir outputs/run_001
```

### `cytof-archetypes-generate-notebooks`
Generate the testing notebooks:

```bash
cytof-archetypes-generate-notebooks --output-dir notebooks_generated
```

### `cytof-archetypes-demo`
Run a compact end-to-end Levine32 demo:
- load data
- train small model
- save outputs
- generate notebooks

---

## Config design

Use YAML configs.

### Base config fields

```yaml
seed: 42
device: "cpu"

dataset:
  name: "levine32"
  input_path: "data/levine32_processed.h5ad"
  marker_columns: null
  label_column: "label"
  cell_id_column: "cell_id"

preprocessing:
  transform: "none"   # or "arcsinh"
  arcsinh_cofactor: 5.0
  normalization: "zscore"  # or "robust_zscore"

model:
  type: "probabilistic_archetypal_autoencoder"
  n_archetypes: 8
  encoder_hidden_dims: [128, 64]
  activation: "relu"
  dropout: 0.1

loss:
  type: "gaussian_nll"
  entropy_reg_weight: 1.0e-3
  diversity_reg_weight: 1.0e-3
  variance_reg_weight: 1.0e-5

training:
  batch_size: 256
  lr: 3.0e-4
  weight_decay: 1.0e-4
  max_epochs: 100
  patience: 15
  grad_clip: 1.0
  mixed_precision: false

output:
  base_dir: "outputs"
```

### Sweep config
Also provide config supporting sweeps over:
- `n_archetypes`
- regularization strengths

---

## Training loop requirements

The trainer must:

- support early stopping on validation NLL
- save best checkpoint
- save final checkpoint
- save training log CSV
- save summary JSON

Track:
- total loss
- reconstruction NLL
- entropy penalty
- diversity penalty
- optional variance penalty

---

## Evaluation outputs

Each run directory should contain:

```text
outputs/run_xxx/
  config_resolved.yaml
  training_log.csv
  training_summary.json
  best_checkpoint.pt

  metrics/
    val_metrics.json
    test_metrics.json

  archetypes/
    archetype_means.csv
    archetype_means.npy
    archetype_logvars.csv
    archetype_vars.csv
    pure_archetype_means.csv
    pure_archetype_vars.csv

  weights/
    cell_weights_val.csv
    cell_weights_test.csv
    class_mean_weights.csv

  embeddings/
    archetype_weight_embedding_val.npz
    archetype_weight_embedding_test.npz

  plots/
    loss_curve.png
    archetype_mean_heatmap.png
    archetype_variance_heatmap.png
    class_mean_weight_heatmap.png
    umap_weights_by_label.png

  README_run.md
```

---

## Evaluation analyses to implement

### 1. Reconstruction quality
- Gaussian NLL
- MSE
- per-class reconstruction loss
- per-marker reconstruction loss

### 2. Weight geometry
- UMAP on archetype weights
- PCA on archetype weights
- class centroids in weight space

### 3. Biological interpretation
- class mean weight heatmap
- marker profiles of pure archetypes
- variance profiles of pure archetypes
- marker correlations with archetype weights

### 4. Sweep interpretation
- compare number of archetypes
- identify over-merged vs over-fragmented solutions

---

## Testing requirements

Add tests for:

### Model behavior
- forward shape correctness
- simplex constraints:
  - weights nonnegative
  - row sums close to 1
- finite NLL
- finite penalties

### Data loading
- Levine32 loader sanity
- marker count validation
- label handling

### Notebook generation
- notebooks are created
- notebooks have expected sections / cells

### Serialization
- outputs save correctly
- CSV and NPY files written

---

## README requirements

Top-level README must explain:

- what probabilistic archetypes are
- why this differs from deterministic archetypes
- what Levine32 is in practical terms
- how to install the package
- how to train a model
- how to generate notebooks
- how to interpret outputs

---

## Analysis-LLM docs

Create:

### `docs/README_for_analysis_llms.md`
Explain:
- where archetype means/variances are saved
- where cell weights are saved
- how to inspect class mean weights
- how to inspect pure archetype profiles
- how to interpret variances

### `docs/ANALYSIS_LLM_INSTRUCTIONS.md`
Write direct instructions for a later analysis LLM:
- inspect class_mean_weights.csv first
- inspect pure archetype means and variances next
- compare archetypes to actual labels
- do not assume one archetype equals one biological class
- interpret archetypes as latent prototype distributions / programs
- use caution in inferring hierarchy or time

---

## Design principles

The implementation should be:

- modular
- readable
- reproducible
- interpretable
- easy to extend

Avoid:
- giant monolithic scripts
- hardcoded Levine32 label names
- overly complex covariance modeling in v1
- overly powerful nonlinear decoders in v1

---

## Phase plan

### Phase 1
Implement:
- Levine32 loader
- preprocessing
- probabilistic archetypal autoencoder
- training loop
- evaluation outputs
- notebook generation

### Phase 2
Add:
- archetype number sweep utilities
- more interpretation plots
- optional deterministic baseline

### Phase 3
Optional future extensions:
- low-rank covariance
- nonlinear post-decoder
- alternative simplex parameterizations
- variational residual noise
- transfer to other CyTOF datasets

---

## Minimal PyTorch design requirements

The core model should roughly expose:

```python
class ProbabilisticArchetypalAutoencoder(nn.Module):
    def __init__(
        self,
        n_markers: int,
        n_archetypes: int,
        encoder_hidden_dims=(128, 64),
        dropout: float = 0.1,
    ):
        ...

    def encode_logits(self, x):
        ...

    def encode(self, x):
        # returns simplex weights
        ...

    def decode_params(self, w):
        # returns mean and logvar
        ...

    def forward(self, x):
        # returns mean, logvar, weights
        ...
```

Also implement reusable functions:
- `gaussian_nll(...)`
- `entropy_penalty(...)`
- `diversity_penalty(...)`

---

## What Codex should deliver

Implement the full package and generate:

1. installable standalone package
2. working training and evaluation CLI
3. notebook generation system
4. tests
5. docs
6. example configs
7. Levine32 demo workflow

At the end, provide:
- files created
- example commands
- any assumptions made
- any limitations in the first version

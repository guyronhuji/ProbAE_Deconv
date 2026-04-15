# Multimodal Deconvolution (Additive Extension)

This extension adds multimodal probabilistic archetypal deconvolution without changing existing single-modality code paths.

## What was added

- New package: `cytof_archetypes.multimodal`
  - `config.py`: multimodal config defaults + load/save + path resolution
  - `data.py`: multimodal data loading, marker resolution, split construction, optional cell pair parsing
  - `model.py`: modality-specific encoders/decoders with a shared latent simplex (`K` archetypes)
  - `losses.py`: alignment losses (per-cell and per-sample) with distance options (`l2`, `cosine`, `jsd`)
  - `trainer.py`: multimodal trainer with optional alignment terms and early stopping
  - `evaluate.py`: artifact writing and reconstruction metrics
- New experiment suite runner:
  - `cytof_archetypes.experiments.run_multimodal_suite`
- New CLI commands:
  - `cytof-archetypes-train-multimodal`
  - `cytof-archetypes-run-multimodal-suite`
- New example configs:
  - `configs/multimodal_training.yaml`
  - `configs/multimodal_suite.yaml`

## Model formulation

For each modality `m` with input `x^m`:

- Encoder:
  - `z^m = softmax(Encoder_m(x^m))`, where `z^m` is in the shared `K`-simplex.
- Decoder:
  - modality specific likelihood (`gaussian`, `nb`, `beta_binomial`) with modality-specific decoder parameters.

Total loss:

`L = L_recon + lambda_cell * L_cell + lambda_sample * L_sample + L_reg`

Where:

- `L_recon = sum_m alpha_m * E[ -log p_m(x^m | z^m) ]`
- `L_cell = E_{(i,j) in P}[ d(z_i^{m_a}, z_j^{m_b}) ]` for paired cells
- `L_sample = E_s[ sum_{m<n} d( mean_{i in s}(z_i^m), mean_{j in s}(z_j^n) ) ]` for matched samples
- `L_reg` includes entropy/diversity/variance terms (same style as single-modality)

Alignment warmup is supported via `alignment.warmup_epochs`.

## Backward-compatibility guarantees

- No existing single-modality trainer/model entrypoints were replaced.
- Existing CLI commands remain unchanged.
- Existing suite runner remains unchanged.
- New behavior is only used through new multimodal commands/configs.

## Expected inputs

Each modality table should include:

- `cell_id_column`
- `sample_id_column`
- marker columns (explicitly provided or inferred numeric columns)

Feature overlap between modalities is not required:

- Each modality is encoded and decoded independently using its own feature space.
- The coupling happens in archetype-proportion space (`K`-dimensional weights), via optional alignment losses.
- This supports cross-modality matching even when modalities share zero genes/features.

Optional per-cell alignment file (`alignment.cell_pairs_path`) should include:

- `left_cell_id` and `right_cell_id` (configurable column names)

## Quick usage

Train one multimodal model:

```bash
cytof-archetypes-train-multimodal --config configs/multimodal_training.yaml
```

Run K sweep across seeds:

```bash
cytof-archetypes-run-multimodal-suite --config configs/multimodal_suite.yaml
```

## Notes

- Parallel sweep execution uses multiprocessing (spawn) when `multiprocessing_workers > 1`.
- Per-sample alignment uses shared sample IDs and compares sample-level mean archetype weights across modalities.

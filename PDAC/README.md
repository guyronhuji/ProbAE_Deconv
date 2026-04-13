# PDAC PAE K Sweep

This directory contains a focused `K`-sweep setup for the probabilistic archetypal autoencoder (PAE) on:

- `data/PDAC_normalized.h5ad`

with:

- `3` random seeds (`42, 123, 456`)
- configurable `K` values

## Run

From repo root:

```bash
python PDAC/run_pae_k_sweep.py --config PDAC/configs/pae_k_sweep.yaml
```

## Main outputs

Under `outputs/PDAC/pae_k_sweep`:

- `tables/pae_k_sweep_seed_level.csv` (one row per seed/K run)
- `tables/pae_k_sweep_k_aggregate.csv` (mean/std across seeds per K)
- `tables/pae_k_selection_summary.csv` (K-selection scoring summary for PAE)

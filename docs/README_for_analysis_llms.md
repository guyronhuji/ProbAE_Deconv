# README for Analysis LLMs

## Purpose of this suite

This project evaluates whether single-cell CyTOF profiles are better modeled as mixtures of **probabilistic archetypal programs** than by deterministic archetypes, classical deconvolution baselines, and generic nonlinear latent-variable models.

The main story is deconvolution quality + interpretability + fit, not sequence modeling.

## What each method represents

- `nmf`: classical linear nonnegative deconvolution baseline.
- `classical_archetypes`: deterministic simplex-combination archetype baseline.
- `deterministic_archetypal_ae`: simplex latent + deterministic archetype means.
- `probabilistic_archetypal_ae`: simplex latent + archetype means + archetype variances (main model).
- `ae`: generic nonlinear bottleneck baseline (non-deconvolution).
- `vae`: probabilistic latent baseline without archetypal structure.

## Primary comparisons to inspect

1. Deterministic vs probabilistic archetypal AE.
2. Deconvolution baselines vs archetypal AE variants.
3. Fit vs interpretability tradeoff against AE/VAE.
4. K-selection for minimal yet biologically sufficient component count.

## Key output locations

Core tables:

- `tables/k_selection_summary.csv`
- `tables/deterministic_vs_probabilistic_summary.csv`
- `tables/deconvolution_quality_summary.csv`
- `tables/fit_vs_complexity_summary.csv`
- `tables/fit_vs_interpretability.csv`
- `tables/per_class_method_metrics.csv`
- `tables/class_component_means.csv`
- `tables/per_cell_weight_entropy.csv`

Core figures:

- `plots/reconstruction_vs_k.png`
- `plots/nll_vs_k.png`
- `plots/class_component_mean_weight_heatmap.png`
- `plots/component_marker_heatmap.png`
- `plots/deterministic_vs_probabilistic_comparison.png`
- `plots/rare_population_preservation.png`
- `plots/pareto_fit_interpretability.png`

Reproducibility:

- `reports/config_resolved.json`
- `reports/split_manifest.csv`
- `reports/suite_environment_log.json`

## Most important figures for manuscript framing

1. K sweep / elbow and K recommendation outputs.
2. Deterministic vs probabilistic archetypal comparison.
3. Class × component weight heatmap.
4. Component/archetype marker heatmap.
5. Rare/transitional population preservation figure.
6. Fit vs interpretability Pareto figure.

## Meaning of “best K”

Best K is defined as the **smallest K within 0.02 of the method-wise maximum K-selection score**, where the score combines:

- fit quality
- rare-class reconstruction behavior
- interpretability score
- component non-redundancy

This is intentionally conservative to favor parsimony when performance is near-tied.

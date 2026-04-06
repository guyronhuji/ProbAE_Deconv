# Design Notes

This implementation follows the `probabilistic_archetypal_autoencoder_levine32_design.md`
specification in this repository.

Phase 1 capabilities currently implemented:

- Levine32 loader from `.h5ad`, `.csv`, `.tsv`, `.txt`
- Split-aware preprocessing (arcsinh, z-score/robust z-score, clipping)
- Probabilistic archetypal autoencoder with diagonal Gaussian decoder
- Loss terms: Gaussian NLL, entropy, archetype diversity, variance regularization
- Early stopping training loop with checkpointing and logs
- Evaluation artifacts (metrics, diagnostics, archetype outputs, weights, embeddings, plots)
- Programmatic notebook generation
- CLI entry points
- Unit tests for model shape/constraints, losses, loader, notebook generation

Phase 2 and beyond are intentionally left for future work.

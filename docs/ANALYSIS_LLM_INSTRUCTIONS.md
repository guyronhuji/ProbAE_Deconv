# ANALYSIS LLM INSTRUCTIONS

Follow this order:

1. Inspect `tables/k_selection_summary.csv` first.
2. Compare deterministic vs probabilistic archetypes next using `tables/deterministic_vs_probabilistic_summary.csv`.
3. Inspect class-component structure via `tables/class_component_means.csv` and `plots/class_component_mean_weight_heatmap.png`.
4. Inspect component biology via `tables/component_marker_profiles.csv` and `plots/component_marker_heatmap.png`.
5. Inspect rare/transitional preservation via `tables/per_class_method_metrics.csv` and `plots/rare_population_preservation.png`.

Interpretation constraints:

- Treat archetypes as latent biological programs, not guaranteed one-to-one cell types.
- Use actual dataset labels from outputs; do not assume label names.
- Do not overinterpret very small metric differences, especially near uncertainty bounds.
- Do not treat auxiliary Transformer/LSTM comparisons as the core claim.
- Do not infer causality or temporal dynamics from static marker patterns alone.

Reporting style guidance:

- Prioritize effect sizes and consistency across K/seeds over single best points.
- Explicitly note when deterministic and probabilistic models are close.
- Highlight rare-class behavior and interpretability tradeoffs alongside fit.

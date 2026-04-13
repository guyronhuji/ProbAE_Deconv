from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cytof_archetypes.models import ProbabilisticArchetypalAutoencoder


def save_archetype_outputs(
    model: ProbabilisticArchetypalAutoencoder,
    marker_names: list[str],
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if model.decoder_family == "gaussian":
        means = model.archetype_means.detach().cpu().numpy()
        logvars = model.archetype_logvars.detach().cpu().numpy()
        vars_ = np.exp(logvars)

        index = [f"arch_{i}" for i in range(means.shape[0])]
        mean_df = pd.DataFrame(means, index=index, columns=marker_names)
        logvar_df = pd.DataFrame(logvars, index=index, columns=marker_names)
        var_df = pd.DataFrame(vars_, index=index, columns=marker_names)

        mean_df.to_csv(out / "archetype_means.csv")
        logvar_df.to_csv(out / "archetype_logvars.csv")
        var_df.to_csv(out / "archetype_vars.csv")

        np.save(out / "archetype_means.npy", means)
        np.save(out / "archetype_logvars.npy", logvars)
        np.save(out / "archetype_vars.npy", vars_)

        # With the linear probabilistic decoder, pure archetypes are one-hot decodes.
        mean_df.to_csv(out / "pure_archetype_means.csv")
        var_df.to_csv(out / "pure_archetype_vars.csv")
        return

    logits = model.archetype_logits.detach().cpu().numpy()
    fractions = torch.softmax(model.archetype_logits.detach(), dim=1).cpu().numpy()
    index = [f"arch_{i}" for i in range(logits.shape[0])]

    logits_df = pd.DataFrame(logits, index=index, columns=marker_names)
    frac_df = pd.DataFrame(fractions, index=index, columns=marker_names)

    logits_df.to_csv(out / "archetype_logits.csv")
    frac_df.to_csv(out / "archetype_gene_fractions.csv")
    np.save(out / "archetype_logits.npy", logits)

    if model.decoder_family == "nb":
        theta = torch.nn.functional.softplus(model.log_theta.detach()).cpu().numpy()
        theta_df = pd.DataFrame({"gene": marker_names, "theta": theta})
        theta_df.to_csv(out / "gene_dispersion.csv", index=False)
    elif model.decoder_family == "beta_binomial":
        concentration = torch.nn.functional.softplus(model.log_concentration.detach()).cpu().numpy()
        concentration_df = pd.DataFrame({"gene": marker_names, "concentration": concentration})
        concentration_df.to_csv(out / "gene_concentration.csv", index=False)


def save_cell_weights(
    weights: np.ndarray,
    cell_ids: np.ndarray,
    labels: np.ndarray | None,
    out_path: str | Path,
) -> None:
    frame = pd.DataFrame(weights, columns=[f"w_{k}" for k in range(weights.shape[1])])
    frame.insert(0, "cell_id", cell_ids)
    if labels is not None and len(labels) == len(frame):
        frame.insert(1, "label", labels)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)


def save_class_weight_summary(
    weights: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
) -> pd.DataFrame:
    frame = pd.DataFrame(weights, columns=[f"w_{k}" for k in range(weights.shape[1])])
    frame["label"] = labels
    class_means = frame.groupby("label", as_index=True).mean(numeric_only=True)
    class_stds = frame.groupby("label", as_index=True).std(numeric_only=True).add_suffix("_std")
    merged = class_means.join(class_stds)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path)
    return merged


def save_diagnostics(
    x: np.ndarray,
    mean: np.ndarray,
    logvar: np.ndarray,
    cell_ids: np.ndarray,
    labels: np.ndarray | None,
    out_path: str | Path,
) -> None:
    inv_var = np.exp(-logvar)
    nll_per_marker = 0.5 * (logvar + ((x - mean) ** 2) * inv_var + np.log(2.0 * np.pi))
    nll_per_cell = np.sum(nll_per_marker, axis=1)
    sq_error = np.mean((x - mean) ** 2, axis=1)
    std_residual = (x - mean) / np.sqrt(np.exp(logvar))
    frame = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "nll": nll_per_cell,
            "squared_error": sq_error,
            "mean_abs_standardized_residual": np.mean(np.abs(std_residual), axis=1),
        }
    )
    if labels is not None and len(labels) == len(frame):
        frame["label"] = labels
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

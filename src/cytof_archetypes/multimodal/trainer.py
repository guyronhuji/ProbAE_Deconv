from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None

from cytof_archetypes.config import resolve_device
from cytof_archetypes.evaluation.reporting import save_environment_log, write_json
from cytof_archetypes.io import ensure_dir
from cytof_archetypes.models import (
    beta_binomial_nll,
    diversity_penalty,
    entropy_penalty,
    gaussian_nll,
    nb_nll,
    variance_regularization,
)
from cytof_archetypes.multimodal.config import save_multimodal_config
from cytof_archetypes.multimodal.data import CellPairSplit, PreparedMultimodalData, prepare_multimodal_data
from cytof_archetypes.multimodal.evaluate import save_multimodal_outputs
from cytof_archetypes.multimodal.losses import per_cell_alignment_loss, per_sample_alignment_loss
from cytof_archetypes.multimodal.model import MultimodalProbabilisticArchetypalAutoencoder
from cytof_archetypes.training.callbacks import EarlyStopping
from cytof_archetypes.utils import set_seed


@dataclass
class _SplitDataset(Dataset[dict[str, torch.Tensor]]):
    x_encoder: torch.Tensor
    x_target: torch.Tensor
    library_size: torch.Tensor

    def __len__(self) -> int:
        return int(self.x_encoder.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_encoder": self.x_encoder[idx],
            "x_target": self.x_target[idx],
            "library_size": self.library_size[idx],
        }


def _next_run_dir(base_dir: str | Path) -> Path:
    base = ensure_dir(base_dir)
    existing: list[int] = []
    for path in base.glob("multimodal_run_*"):
        suffix = path.name.replace("multimodal_run_", "")
        if suffix.isdigit():
            existing.append(int(suffix))
    next_idx = (max(existing) + 1) if existing else 1
    out = base / f"multimodal_run_{next_idx:03d}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _make_loader(split, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = _SplitDataset(
        x_encoder=torch.from_numpy(split.x_encoder.astype(np.float32)),
        x_target=torch.from_numpy(split.x_target.astype(np.float32)),
        library_size=torch.from_numpy(split.library_size.astype(np.float32)),
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _modality_loss_terms(
    model: MultimodalProbabilisticArchetypalAutoencoder,
    modality: str,
    batch: dict[str, torch.Tensor],
    loss_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    out = model.forward_modality(modality, batch["x_encoder"], batch["library_size"])
    decoder_family = model.decoder_family[modality]

    if decoder_family == "gaussian":
        recon = gaussian_nll(batch["x_target"], out["recon"], out["logvar"], reduction="mean")
        var_reg = variance_regularization(model.archetype_logvars[modality])
    elif decoder_family == "nb":
        recon = nb_nll(batch["x_target"], out["mu"], out["theta"], reduction="mean")
        var_reg = torch.zeros((), dtype=batch["x_encoder"].dtype, device=batch["x_encoder"].device)
    else:
        recon = beta_binomial_nll(
            m_counts=batch["x_target"],
            n_counts=batch["library_size"].unsqueeze(1),
            probs=out["probs"],
            concentration=out["concentration"],
            reduction="mean",
        )
        var_reg = torch.zeros((), dtype=batch["x_encoder"].dtype, device=batch["x_encoder"].device)

    ent = entropy_penalty(out["weights"])
    div = diversity_penalty(model.diversity_basis(modality))

    modality_recon_weights = loss_cfg.get("modality_reconstruction_weights", {})
    recon_weight = float(loss_cfg.get("reconstruction_weight", 1.0)) * float(modality_recon_weights.get(modality, 1.0))

    total = (
        recon_weight * recon
        + float(loss_cfg.get("entropy_reg_weight", 0.0)) * ent
        + float(loss_cfg.get("diversity_reg_weight", 0.0)) * div
        + float(loss_cfg.get("variance_reg_weight", 0.0)) * var_reg
    )
    return {
        "loss": total,
        "recon": recon,
        "entropy": ent,
        "diversity": div,
        "var_reg": var_reg,
    }


def _alignment_mode(alignment_cfg: dict[str, Any]) -> str:
    mode = str(alignment_cfg.get("mode", "none")).lower()
    if mode not in {"none", "cell", "sample", "both"}:
        raise ValueError("alignment.mode must be one of: none, cell, sample, both")
    return mode


def _alignment_scale(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return float(min(1.0, float(epoch + 1) / float(max(warmup_epochs, 1))))


def _cell_alignment_term(
    model: MultimodalProbabilisticArchetypalAutoencoder,
    prepared: PreparedMultimodalData,
    pair_split: CellPairSplit | None,
    alignment_cfg: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    if pair_split is None:
        return torch.zeros((), device=device)
    if len(pair_split.left_indices) == 0:
        return torch.zeros((), device=device)

    batch_size = int(alignment_cfg.get("cell_batch_size", 256))
    sample_size = min(batch_size, len(pair_split.left_indices))
    choose = np.random.choice(len(pair_split.left_indices), size=sample_size, replace=False)
    left_idx = pair_split.left_indices[choose]
    right_idx = pair_split.right_indices[choose]

    left_split = prepared.modalities[pair_split.left_modality].train
    right_split = prepared.modalities[pair_split.right_modality].train
    left_x = torch.from_numpy(left_split.x_encoder[left_idx].astype(np.float32)).to(device)
    right_x = torch.from_numpy(right_split.x_encoder[right_idx].astype(np.float32)).to(device)

    left_w = model.encode(pair_split.left_modality, left_x)
    right_w = model.encode(pair_split.right_modality, right_x)
    return per_cell_alignment_loss(left_w, right_w, metric=str(alignment_cfg.get("distance", "l2")))


def _sample_alignment_term(
    model: MultimodalProbabilisticArchetypalAutoencoder,
    prepared: PreparedMultimodalData,
    split_name: str,
    alignment_cfg: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    shared_samples = prepared.shared_samples.get(split_name)
    if shared_samples is None or len(shared_samples) == 0:
        return torch.zeros((), device=device)

    sample_batch_size = int(alignment_cfg.get("sample_batch_size", 64))
    sample_max_cells = int(alignment_cfg.get("sample_max_cells_per_modality", 512))
    n_pick = min(sample_batch_size, len(shared_samples))
    sampled = np.random.choice(shared_samples, size=n_pick, replace=False)

    means_by_modality: dict[str, torch.Tensor] = {}
    for modality_name in prepared.modality_order:
        split = getattr(prepared.modalities[modality_name], split_name)
        sample_ids = split.sample_ids.astype(str)
        means: list[torch.Tensor] = []
        for sample_id in sampled:
            idx = np.where(sample_ids == str(sample_id))[0]
            if len(idx) == 0:
                means.append(torch.zeros((model.n_archetypes,), dtype=torch.float32, device=device))
                continue
            if len(idx) > sample_max_cells:
                idx = np.random.choice(idx, size=sample_max_cells, replace=False)
            x = torch.from_numpy(split.x_encoder[idx].astype(np.float32)).to(device)
            w = model.encode(modality_name, x)
            means.append(w.mean(dim=0))
        means_by_modality[modality_name] = torch.stack(means, dim=0)

    return per_sample_alignment_loss(means_by_modality, metric=str(alignment_cfg.get("distance", "l2")))


def _evaluate_val(
    model: MultimodalProbabilisticArchetypalAutoencoder,
    prepared: PreparedMultimodalData,
    loaders: dict[str, DataLoader],
    loss_cfg: dict[str, Any],
    alignment_cfg: dict[str, Any],
    epoch: int,
    device: torch.device,
) -> dict[str, float]:
    mode = _alignment_mode(alignment_cfg)
    align_scale = _alignment_scale(epoch, warmup_epochs=int(alignment_cfg.get("warmup_epochs", 0)))

    model.eval()
    sums = {
        "loss": 0.0,
        "recon": 0.0,
        "entropy": 0.0,
        "diversity": 0.0,
        "var_reg": 0.0,
        "align_cell": 0.0,
        "align_sample": 0.0,
    }
    n_obs = 0

    with torch.no_grad():
        for modality, loader in loaders.items():
            for batch in loader:
                batch_t = {k: v.to(device) for k, v in batch.items()}
                terms = _modality_loss_terms(model, modality, batch_t, loss_cfg)
                bs = int(batch_t["x_encoder"].shape[0])
                sums["loss"] += float(terms["loss"].item()) * bs
                sums["recon"] += float(terms["recon"].item()) * bs
                sums["entropy"] += float(terms["entropy"].item()) * bs
                sums["diversity"] += float(terms["diversity"].item()) * bs
                sums["var_reg"] += float(terms["var_reg"].item()) * bs
                n_obs += bs

        cell_term = 0.0
        if mode in {"cell", "both"}:
            pair_split = prepared.pair_indices.get("val") if prepared.pair_indices is not None else None
            cell_term = float(
                _cell_alignment_term(model, prepared, pair_split=pair_split, alignment_cfg=alignment_cfg, device=device)
                .detach()
                .cpu()
                .item()
            )
        sample_term = 0.0
        if mode in {"sample", "both"}:
            sample_term = float(
                _sample_alignment_term(
                    model,
                    prepared,
                    split_name="val",
                    alignment_cfg=alignment_cfg,
                    device=device,
                )
                .detach()
                .cpu()
                .item()
            )

    denom = max(n_obs, 1)
    out = {key: sums[key] / denom for key in ["loss", "recon", "entropy", "diversity", "var_reg"]}
    out["align_cell"] = cell_term
    out["align_sample"] = sample_term
    out["loss_total"] = (
        out["loss"]
        + align_scale * float(alignment_cfg.get("per_cell_weight", 1.0)) * cell_term
        + align_scale * float(alignment_cfg.get("per_sample_weight", 1.0)) * sample_term
    )
    return out


def _build_model_specs(config: dict[str, Any], prepared: PreparedMultimodalData) -> dict[str, dict[str, Any]]:
    cfg_by_name = {str(m["name"]): m for m in config.get("modalities", [])}
    specs: dict[str, dict[str, Any]] = {}
    for modality_name in prepared.modality_order:
        modality_cfg = cfg_by_name[modality_name]
        model_cfg = modality_cfg.get("model", {})
        specs[modality_name] = {
            "n_markers": len(prepared.modalities[modality_name].marker_names),
            "decoder_family": str(model_cfg.get("decoder_family", "gaussian")),
            "encoder_hidden_dims": model_cfg.get("encoder_hidden_dims", [128, 64]),
            "activation": str(model_cfg.get("activation", "relu")),
            "dropout": float(model_cfg.get("dropout", 0.1)),
            "dispersion": str(model_cfg.get("dispersion", "gene")),
        }
    return specs


def train_multimodal_from_config(config: dict[str, Any]) -> Path:
    resolved = copy.deepcopy(config)
    set_seed(int(resolved.get("seed", 42)))

    prepared = prepare_multimodal_data(resolved)
    device = torch.device(resolve_device(resolved))

    output_cfg = resolved.get("output", {})
    run_name = output_cfg.get("run_name")
    if run_name:
        run_dir = Path(output_cfg.get("base_dir", "outputs/multimodal")) / str(run_name)
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = _next_run_dir(output_cfg.get("base_dir", "outputs/multimodal"))

    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "metrics")
    ensure_dir(run_dir / "reports")

    save_multimodal_config(resolved, run_dir / "config_resolved.yaml")
    write_json(run_dir / "reports" / "config_resolved.json", resolved)
    save_environment_log(run_dir / "reports" / "environment_log.json", extra={"suite": "multimodal_deconvolution"})

    model_specs = _build_model_specs(resolved, prepared)
    model = MultimodalProbabilisticArchetypalAutoencoder(
        modality_specs=model_specs,
        n_archetypes=int(resolved.get("shared_latent", {}).get("n_archetypes", 8)),
    ).to(device)

    train_cfg = resolved.get("training", {})
    loss_cfg = resolved.get("loss", {})
    alignment_cfg = resolved.get("alignment", {})
    mode = _alignment_mode(alignment_cfg)

    train_loaders = {
        name: _make_loader(modality.train, batch_size=int(train_cfg.get("batch_size", 256)), shuffle=True)
        for name, modality in prepared.modalities.items()
    }
    val_loaders = {
        name: _make_loader(modality.val, batch_size=int(train_cfg.get("batch_size", 256)), shuffle=False)
        for name, modality in prepared.modalities.items()
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )

    early_stopping = EarlyStopping(
        patience=int(train_cfg.get("patience", 15)),
        min_delta=float(train_cfg.get("early_stopping_min_delta", 0.0)),
    )

    max_epochs = int(train_cfg.get("max_epochs", 100))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    progress_on = bool(train_cfg.get("progress_bar", True)) and tqdm is not None
    history_rows: list[dict[str, float | int]] = []

    pbar = None
    if progress_on:
        pbar = tqdm(
            range(max_epochs),
            desc=str(train_cfg.get("progress_bar_desc", "multimodal-train")),
            leave=bool(train_cfg.get("progress_bar_leave", False)),
            unit="epoch",
        )
        iterator = pbar
    else:
        iterator = range(max_epochs)

    best_epoch = -1
    best_val = float("inf")
    checkpoint_path = run_dir / "checkpoints" / "best_checkpoint.pt"

    for epoch in iterator:
        model.train()
        sums = {
            "loss": 0.0,
            "recon": 0.0,
            "entropy": 0.0,
            "diversity": 0.0,
            "var_reg": 0.0,
        }
        n_obs = 0

        for modality, loader in train_loaders.items():
            for batch in loader:
                batch_t = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                terms = _modality_loss_terms(model, modality, batch_t, loss_cfg)
                terms["loss"].backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                bs = int(batch_t["x_encoder"].shape[0])
                sums["loss"] += float(terms["loss"].item()) * bs
                sums["recon"] += float(terms["recon"].item()) * bs
                sums["entropy"] += float(terms["entropy"].item()) * bs
                sums["diversity"] += float(terms["diversity"].item()) * bs
                sums["var_reg"] += float(terms["var_reg"].item()) * bs
                n_obs += bs

        align_scale = _alignment_scale(epoch, warmup_epochs=int(alignment_cfg.get("warmup_epochs", 0)))
        cell_align_value = 0.0
        sample_align_value = 0.0

        if mode in {"cell", "both"}:
            pair_split = prepared.pair_indices.get("train") if prepared.pair_indices is not None else None
            if pair_split is not None and len(pair_split.left_indices) > 0:
                optimizer.zero_grad(set_to_none=True)
                cell_term = _cell_alignment_term(
                    model,
                    prepared,
                    pair_split=pair_split,
                    alignment_cfg=alignment_cfg,
                    device=device,
                )
                cell_loss = align_scale * float(alignment_cfg.get("per_cell_weight", 1.0)) * cell_term
                cell_loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                cell_align_value = float(cell_term.detach().cpu().item())

        if mode in {"sample", "both"}:
            optimizer.zero_grad(set_to_none=True)
            sample_term = _sample_alignment_term(
                model,
                prepared,
                split_name="train",
                alignment_cfg=alignment_cfg,
                device=device,
            )
            sample_loss = align_scale * float(alignment_cfg.get("per_sample_weight", 1.0)) * sample_term
            sample_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            sample_align_value = float(sample_term.detach().cpu().item())

        train_loss = sums["loss"] / max(n_obs, 1)
        val_stats = _evaluate_val(
            model=model,
            prepared=prepared,
            loaders=val_loaders,
            loss_cfg=loss_cfg,
            alignment_cfg=alignment_cfg,
            epoch=epoch,
            device=device,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_recon": sums["recon"] / max(n_obs, 1),
                "train_entropy": sums["entropy"] / max(n_obs, 1),
                "train_diversity": sums["diversity"] / max(n_obs, 1),
                "train_var_reg": sums["var_reg"] / max(n_obs, 1),
                "train_align_cell": cell_align_value,
                "train_align_sample": sample_align_value,
                "val_loss": val_stats["loss"],
                "val_recon": val_stats["recon"],
                "val_entropy": val_stats["entropy"],
                "val_diversity": val_stats["diversity"],
                "val_var_reg": val_stats["var_reg"],
                "val_align_cell": val_stats["align_cell"],
                "val_align_sample": val_stats["align_sample"],
                "val_loss_total": val_stats["loss_total"],
            }
        )

        if val_stats["loss_total"] < best_val:
            best_val = float(val_stats["loss_total"])
            best_epoch = int(epoch)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss_total": best_val,
                },
                checkpoint_path,
            )

        should_stop = early_stopping.update(float(val_stats["loss_total"]))

        if pbar is not None:
            pbar.set_postfix(
                {
                    "best": f"{best_val:.4f}",
                    "train": f"{train_loss:.4f}",
                    "val": f"{val_stats['loss_total']:.4f}",
                    "patience": f"{early_stopping.num_bad_epochs}/{early_stopping.patience}",
                }
            )

        if should_stop:
            break

    if pbar is not None:
        pbar.close()

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(run_dir / "metrics" / "training_history.csv", index=False)

    split_rows: list[dict[str, str]] = []
    for modality_name, modality in prepared.modalities.items():
        for split_name in ("train", "val", "test"):
            split = getattr(modality, split_name)
            for cell_id, sample_id in zip(split.cell_ids, split.sample_ids):
                split_rows.append(
                    {
                        "modality": modality_name,
                        "split": split_name,
                        "cell_id": str(cell_id),
                        "sample_id": str(sample_id),
                    }
                )
    pd.DataFrame(split_rows).to_csv(run_dir / "reports" / "split_manifest.csv", index=False)

    outputs_summary = save_multimodal_outputs(
        model=model,
        prepared=prepared,
        run_dir=run_dir,
        device=device,
    )

    outputs_summary.update(
        {
            "best_epoch": best_epoch,
            "best_val_loss_total": best_val,
            "device": str(device),
            "history_rows": int(len(history_rows)),
        }
    )
    write_json(run_dir / "metrics" / "summary.json", outputs_summary)
    return run_dir

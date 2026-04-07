from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None

from cytof_archetypes.baselines.base import BaseMethod, MethodRunResult, SplitResult, make_unit_logvar
from cytof_archetypes.models import (
    ProbabilisticArchetypalAutoencoder,
    diversity_penalty,
    entropy_penalty,
    gaussian_nll,
    variance_regularization,
)
from cytof_archetypes.utils import set_seed


class DeterministicArchetypalAutoencoder(nn.Module):
    def __init__(
        self,
        n_markers: int,
        n_archetypes: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [n_markers, *hidden_dims, n_archetypes]
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.encoder = nn.Sequential(*layers)
        self.archetype_means = nn.Parameter(torch.randn(n_archetypes, n_markers) * 0.05)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.encoder(x), dim=-1)
        mean = weights @ self.archetype_means
        return mean, weights


class StandardAutoencoder(nn.Module):
    def __init__(
        self,
        n_markers: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        enc_layers: list[nn.Module] = []
        dims = [n_markers, *hidden_dims]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            enc_layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
        enc_layers.append(nn.Linear(dims[-1], latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        dims_dec = [latent_dim, *hidden_dims[::-1], n_markers]
        for in_dim, out_dim in zip(dims_dec[:-2], dims_dec[1:-1]):
            dec_layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        dec_layers.append(nn.Linear(dims_dec[-2], dims_dec[-1]))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        n_markers: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        enc_layers: list[nn.Module] = []
        dims = [n_markers, *hidden_dims]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            enc_layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], latent_dim)

        dec_layers: list[nn.Module] = []
        dims_dec = [latent_dim, *hidden_dims[::-1], n_markers]
        for in_dim, out_dim in zip(dims_dec[:-2], dims_dec[1:-1]):
            dec_layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        dec_layers.append(nn.Linear(dims_dec[-2], dims_dec[-1]))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        return recon, mu, logvar, z


@dataclass
class _NeuralConfig:
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 80
    patience: int = 12
    grad_clip: float = 1.0


class DeterministicArchetypalMethod(BaseMethod):
    method_name = "deterministic_archetypal_ae"

    def run(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        labels_val: np.ndarray | None,
        labels_test: np.ndarray | None,
        cell_ids_val: np.ndarray,
        cell_ids_test: np.ndarray,
        marker_names: list[str],
        seed: int,
        representation_dim: int,
        config: dict,
    ) -> MethodRunResult:
        set_seed(seed)
        device = _safe_torch_device(config.get("device", "cpu"))
        train_cfg = _NeuralConfig(**_subset_train_cfg(config))
        model = DeterministicArchetypalAutoencoder(
            n_markers=x_train.shape[1],
            n_archetypes=representation_dim,
            hidden_dims=tuple(config.get("hidden_dims", [128, 64])),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)
        param_count = int(sum(p.numel() for p in model.parameters()))
        progress_cfg = _training_progress_options(
            config=config,
            method_name=self.method_name,
            representation_dim=representation_dim,
            seed=seed,
        )
        history = _train_deterministic_archetypal(
            model,
            x_train,
            x_val,
            train_cfg,
            device,
            config,
            progress_cfg=progress_cfg,
        )
        split_results = _predict_deterministic_archetypal(
            model,
            x_train,
            x_val,
            x_test,
            labels_val,
            labels_test,
            cell_ids_val,
            cell_ids_test,
            device,
        )
        return MethodRunResult(
            method=self.method_name,
            seed=seed,
            representation_dim=representation_dim,
            params={"hidden_dims": config.get("hidden_dims", [128, 64]), "param_count": param_count},
            components_mean=model.archetype_means.detach().cpu().numpy().astype(np.float32),
            components_var=np.zeros((representation_dim, x_train.shape[1]), dtype=np.float32),
            marker_names=marker_names,
            split_results=split_results,
            training_history=history,
        )


class ProbabilisticArchetypalMethod(BaseMethod):
    method_name = "probabilistic_archetypal_ae"

    def run(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        labels_val: np.ndarray | None,
        labels_test: np.ndarray | None,
        cell_ids_val: np.ndarray,
        cell_ids_test: np.ndarray,
        marker_names: list[str],
        seed: int,
        representation_dim: int,
        config: dict,
    ) -> MethodRunResult:
        set_seed(seed)
        device = _safe_torch_device(config.get("device", "cpu"))
        train_cfg = _NeuralConfig(**_subset_train_cfg(config))
        model = ProbabilisticArchetypalAutoencoder(
            n_markers=x_train.shape[1],
            n_archetypes=representation_dim,
            encoder_hidden_dims=tuple(config.get("hidden_dims", [128, 64])),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)
        param_count = int(sum(p.numel() for p in model.parameters()))
        progress_cfg = _training_progress_options(
            config=config,
            method_name=self.method_name,
            representation_dim=representation_dim,
            seed=seed,
        )
        history = _train_probabilistic_archetypal(
            model,
            x_train,
            x_val,
            train_cfg,
            device,
            config,
            progress_cfg=progress_cfg,
        )
        split_results = _predict_probabilistic_archetypal(
            model,
            x_train,
            x_val,
            x_test,
            labels_val,
            labels_test,
            cell_ids_val,
            cell_ids_test,
            device,
        )
        return MethodRunResult(
            method=self.method_name,
            seed=seed,
            representation_dim=representation_dim,
            params={
                "hidden_dims": config.get("hidden_dims", [128, 64]),
                "entropy_reg_weight": float(config.get("entropy_reg_weight", 1e-3)),
                "diversity_reg_weight": float(config.get("diversity_reg_weight", 1e-3)),
                "param_count": param_count,
            },
            components_mean=model.archetype_means.detach().cpu().numpy().astype(np.float32),
            components_var=np.exp(model.archetype_logvars.detach().cpu().numpy()).astype(np.float32),
            marker_names=marker_names,
            split_results=split_results,
            training_history=history,
        )


class AEMethod(BaseMethod):
    method_name = "ae"

    def run(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        labels_val: np.ndarray | None,
        labels_test: np.ndarray | None,
        cell_ids_val: np.ndarray,
        cell_ids_test: np.ndarray,
        marker_names: list[str],
        seed: int,
        representation_dim: int,
        config: dict,
    ) -> MethodRunResult:
        set_seed(seed)
        device = _safe_torch_device(config.get("device", "cpu"))
        train_cfg = _NeuralConfig(**_subset_train_cfg(config))
        model = StandardAutoencoder(
            n_markers=x_train.shape[1],
            latent_dim=representation_dim,
            hidden_dims=tuple(config.get("hidden_dims", [128, 64])),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)
        param_count = int(sum(p.numel() for p in model.parameters()))
        progress_cfg = _training_progress_options(
            config=config,
            method_name=self.method_name,
            representation_dim=representation_dim,
            seed=seed,
        )
        history = _train_ae_like(
            model=model,
            x_train=x_train,
            x_val=x_val,
            train_cfg=train_cfg,
            device=device,
            loss_fn=lambda recon, _extra, batch: torch.mean((recon - batch) ** 2),
            progress_cfg=progress_cfg,
        )
        split_results = _predict_ae(model, x_train, x_val, x_test, labels_val, labels_test, cell_ids_val, cell_ids_test, device)
        return MethodRunResult(
            method=self.method_name,
            seed=seed,
            representation_dim=representation_dim,
            params={"hidden_dims": config.get("hidden_dims", [128, 64]), "param_count": param_count},
            components_mean=None,
            components_var=None,
            marker_names=marker_names,
            split_results=split_results,
            training_history=history,
        )


class VAEMethod(BaseMethod):
    method_name = "vae"

    def run(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        labels_val: np.ndarray | None,
        labels_test: np.ndarray | None,
        cell_ids_val: np.ndarray,
        cell_ids_test: np.ndarray,
        marker_names: list[str],
        seed: int,
        representation_dim: int,
        config: dict,
    ) -> MethodRunResult:
        set_seed(seed)
        device = _safe_torch_device(config.get("device", "cpu"))
        beta = float(config.get("beta", 1.0))
        train_cfg = _NeuralConfig(**_subset_train_cfg(config))

        model = VariationalAutoencoder(
            n_markers=x_train.shape[1],
            latent_dim=representation_dim,
            hidden_dims=tuple(config.get("hidden_dims", [128, 64])),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)
        param_count = int(sum(p.numel() for p in model.parameters()))
        progress_cfg = _training_progress_options(
            config=config,
            method_name=self.method_name,
            representation_dim=representation_dim,
            seed=seed,
        )

        def _vae_loss(recon: torch.Tensor, extra: dict[str, torch.Tensor], batch: torch.Tensor) -> torch.Tensor:
            recon_loss = torch.mean((recon - batch) ** 2)
            mu = extra["mu"]
            logvar = extra["logvar"]
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + beta * kl

        history = _train_ae_like(
            model=model,
            x_train=x_train,
            x_val=x_val,
            train_cfg=train_cfg,
            device=device,
            loss_fn=_vae_loss,
            is_vae=True,
            progress_cfg=progress_cfg,
        )

        split_results = _predict_vae(
            model,
            x_train,
            x_val,
            x_test,
            labels_val,
            labels_test,
            cell_ids_val,
            cell_ids_test,
            device,
        )
        return MethodRunResult(
            method=self.method_name,
            seed=seed,
            representation_dim=representation_dim,
            params={"hidden_dims": config.get("hidden_dims", [128, 64]), "beta": beta, "param_count": param_count},
            components_mean=None,
            components_var=None,
            marker_names=marker_names,
            split_results=split_results,
            training_history=history,
        )


def _subset_train_cfg(config: dict) -> dict:
    allowed = {"batch_size", "lr", "weight_decay", "max_epochs", "patience", "grad_clip"}
    return {key: config[key] for key in allowed if key in config}


def _safe_torch_device(requested: str | object) -> torch.device:
    req = str(requested).lower()
    if req == "mps":
        if bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            return torch.device("mps")
        req = "cuda" if torch.cuda.is_available() else "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        req = "cpu"
    if req == "auto":
        if bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            req = "mps"
        elif torch.cuda.is_available():
            req = "cuda"
        else:
            req = "cpu"
    return torch.device(req)


class _TensorBatchLoader:
    """Fast batch loader for tensors on any device.

    DataLoader+TensorDataset calls __getitem__ once per sample then torch.stack,
    producing hundreds of thousands of small GPU ops per epoch for large datasets.
    This loader does a single tensor index per batch instead — ~10–50x faster.
    """

    def __init__(self, tensor: torch.Tensor, batch_size: int, shuffle: bool) -> None:
        self.tensor = tensor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._n = len(tensor)

    def __len__(self) -> int:
        return math.ceil(self._n / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self._n, device=self.tensor.device)
        else:
            idx = torch.arange(self._n, device=self.tensor.device)
        for start in range(0, self._n, self.batch_size):
            yield (self.tensor[idx[start : start + self.batch_size]],)


def _to_loader(
    x: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device | None = None,
) -> _TensorBatchLoader:
    tensor = torch.from_numpy(x.astype(np.float32))
    if device is not None:
        tensor = tensor.to(device)
    return _TensorBatchLoader(tensor, batch_size, shuffle)


def _training_progress_options(
    config: dict[str, Any],
    method_name: str,
    representation_dim: int,
    seed: int,
) -> dict[str, Any]:
    enabled = bool(config.get("show_training_progress", False))
    level = str(config.get("training_progress_level", "epoch")).lower()
    if level not in {"epoch", "batch"}:
        level = "epoch"
    desc = str(
        config.get(
            "training_progress_desc",
            f"{method_name} dim={representation_dim} seed={seed}",
        )
    )
    leave = bool(config.get("training_progress_leave", False))
    live_plot = bool(config.get("live_loss_plot", False))
    return {"enabled": enabled, "level": level, "desc": desc, "leave": leave, "live_plot": live_plot}


def _make_live_plot(desc: str) -> tuple:
    """Set up a matplotlib figure for live loss updates in Jupyter.

    Returns (fig, ax, (display_fn, clear_fn)) on success, (None, None, None) otherwise.
    Does NOT call matplotlib.use() — the Jupyter kernel has already configured the backend.
    """
    try:
        import matplotlib.pyplot as plt
        from IPython.display import display as _ipy_display, clear_output as _clear_fn
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(desc)
        plt.tight_layout()
        plt.close(fig)  # prevent a blank figure showing up at creation time
        return fig, ax, (_ipy_display, _clear_fn)
    except Exception:
        return None, None, None


def _update_live_plot(
    fig: Any,
    ax: Any,
    display_fns: tuple,
    rows: list,
    best_val: float,
    patience_counter: int,
    patience: int,
    desc: str,
) -> None:
    _ipy_display, _clear_fn = display_fns
    epochs = [r["epoch"] for r in rows]
    ax.clear()
    ax.plot(epochs, [r["train_loss"] for r in rows], color="steelblue", linewidth=1.5, label="train")
    ax.plot(epochs, [r["val_loss"] for r in rows], color="tomato", linewidth=1.5, label="val")
    if np.isfinite(best_val):
        ax.axhline(best_val, color="mediumseagreen", linestyle="--", linewidth=1, alpha=0.8,
                   label=f"best={best_val:.3f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{desc}  |  epoch {epochs[-1]}  |  patience {patience_counter}/{patience}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _clear_fn(wait=True)
    _ipy_display(fig)


def _run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_eval: Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    train_cfg: _NeuralConfig,
    progress_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    progress_cfg = progress_cfg or {}
    live_plot = bool(progress_cfg.get("live_plot", False))
    # tqdm bar suppressed when live_plot is on — epoch/patience info lives in the plot title instead
    show_training_progress = bool(progress_cfg.get("enabled", False)) and tqdm is not None and not live_plot
    progress_level = str(progress_cfg.get("level", "epoch")).lower()
    if progress_level not in {"epoch", "batch"}:
        progress_level = "epoch"
    progress_desc = str(progress_cfg.get("desc", "training"))
    progress_leave = bool(progress_cfg.get("leave", False))
    plot_every = max(1, int(progress_cfg.get("plot_every", 1)))

    fig, ax, display_fns = _make_live_plot(progress_desc) if live_plot else (None, None, None)
    live_plot = fig is not None  # downgrade if IPython / matplotlib setup failed

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience_counter = 0
    rows: list[dict[str, float]] = []
    epoch_iter: Any = range(1, train_cfg.max_epochs + 1)
    if show_training_progress:
        epoch_iter = tqdm(
            epoch_iter,
            total=train_cfg.max_epochs,
            desc=f"{progress_desc} [epoch]",
            unit="epoch",
            leave=progress_leave,
        )

    try:
        for epoch in epoch_iter:
            model.train()
            train_loss_acc = None
            train_batch_count = 0
            batch_iter: Any = train_loader
            batch_bar: Any | None = None
            if show_training_progress and progress_level == "batch":
                batch_bar = tqdm(
                    train_loader,
                    total=len(train_loader),
                    desc=f"{progress_desc} [epoch {epoch}]",
                    unit="batch",
                    leave=False,
                )
                batch_iter = batch_bar

            try:
                for (batch_x,) in batch_iter:
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = loss_eval(batch_x)
                    loss.backward()
                    if train_cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                    optimizer.step()
                    loss_d = loss.detach()
                    train_loss_acc = loss_d if train_loss_acc is None else train_loss_acc + loss_d
                    train_batch_count += 1
                    if batch_bar is not None:
                        batch_bar.set_postfix({"loss": f"{float(loss_d.item()):.4f}"})
            finally:
                if batch_bar is not None:
                    batch_bar.close()

            model.eval()
            with torch.no_grad():
                val_loss_acc = None
                val_batch_count = 0
                for (batch_x,) in val_loader:
                    vl = loss_eval(batch_x)[0].detach()
                    val_loss_acc = vl if val_loss_acc is None else val_loss_acc + vl
                    val_batch_count += 1

            # One CUDA sync per epoch instead of ~1800 syncs
            train_mean = float((train_loss_acc / train_batch_count).item()) if train_batch_count else float("nan")
            val_mean = float((val_loss_acc / val_batch_count).item()) if val_batch_count else float("nan")
            rows.append({"epoch": epoch, "train_loss": train_mean, "val_loss": val_mean})

            if np.isfinite(val_mean) and val_mean < best_val:
                best_val = val_mean
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if show_training_progress and hasattr(epoch_iter, "set_postfix"):
                best_str = f"{best_val:.4f}" if np.isfinite(best_val) else "nan"
                epoch_iter.set_postfix(
                    {
                        "train": f"{train_mean:.4f}",
                        "val": f"{val_mean:.4f}",
                        "best": best_str,
                        "pat": f"{patience_counter}/{train_cfg.patience}",
                    },
                    refresh=False,  # avoid double-print in non-TTY (tee/pipe); update fires on next iteration
                )

            if live_plot and epoch % plot_every == 0:
                _update_live_plot(fig, ax, display_fns, rows, best_val,
                                  patience_counter, train_cfg.patience, progress_desc)

            if patience_counter >= train_cfg.patience:
                break
    finally:
        if show_training_progress and hasattr(epoch_iter, "close"):
            epoch_iter.close()
        if live_plot:
            # Final redraw to capture the stopping epoch
            _update_live_plot(fig, ax, display_fns, rows, best_val,
                              patience_counter, train_cfg.patience, progress_desc)

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def _train_deterministic_archetypal(
    model: DeterministicArchetypalAutoencoder,
    x_train: np.ndarray,
    x_val: np.ndarray,
    train_cfg: _NeuralConfig,
    device: torch.device,
    config: dict,
    progress_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    ent_w = float(config.get("entropy_reg_weight", 1e-3))
    div_w = float(config.get("diversity_reg_weight", 1e-3))
    loader_train = _to_loader(x_train, train_cfg.batch_size, True, device=device)
    loader_val = _to_loader(x_val, train_cfg.batch_size, False, device=device) if len(x_val) else _to_loader(x_train[:1], 1, False, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    def _loss_eval(batch_x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_x = batch_x.to(device)
        recon, weights = model(batch_x)
        mse = torch.mean((recon - batch_x) ** 2)
        loss = mse + ent_w * entropy_penalty(weights) + div_w * diversity_penalty(model.archetype_means)
        return loss, {"mse": mse}

    return _run_training_loop(model, loader_train, loader_val, opt, _loss_eval, train_cfg, progress_cfg=progress_cfg)


def _train_probabilistic_archetypal(
    model: ProbabilisticArchetypalAutoencoder,
    x_train: np.ndarray,
    x_val: np.ndarray,
    train_cfg: _NeuralConfig,
    device: torch.device,
    config: dict,
    progress_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    ent_w = float(config.get("entropy_reg_weight", 1e-3))
    div_w = float(config.get("diversity_reg_weight", 1e-3))
    var_w = float(config.get("variance_reg_weight", 1e-5))
    loader_train = _to_loader(x_train, train_cfg.batch_size, True, device=device)
    loader_val = _to_loader(x_val, train_cfg.batch_size, False, device=device) if len(x_val) else _to_loader(x_train[:1], 1, False, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    def _loss_eval(batch_x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_x = batch_x.to(device)
        mean, logvar, weights = model(batch_x)
        nll = gaussian_nll(batch_x, mean, logvar, reduction="mean")
        loss = (
            nll
            + ent_w * entropy_penalty(weights)
            + div_w * diversity_penalty(model.archetype_means)
            + var_w * variance_regularization(model.archetype_logvars)
        )
        return loss, {"nll": nll}

    return _run_training_loop(model, loader_train, loader_val, opt, _loss_eval, train_cfg, progress_cfg=progress_cfg)


def _train_ae_like(
    model: nn.Module,
    x_train: np.ndarray,
    x_val: np.ndarray,
    train_cfg: _NeuralConfig,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, dict[str, torch.Tensor], torch.Tensor], torch.Tensor],
    is_vae: bool = False,
    progress_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    loader_train = _to_loader(x_train, train_cfg.batch_size, True, device=device)
    loader_val = _to_loader(x_val, train_cfg.batch_size, False, device=device) if len(x_val) else _to_loader(x_train[:1], 1, False, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    def _loss_eval(batch_x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_x = batch_x.to(device)
        if is_vae:
            recon, mu, logvar, z = model(batch_x)
            loss = loss_fn(recon, {"mu": mu, "logvar": logvar, "z": z}, batch_x)
        else:
            recon, latent = model(batch_x)
            loss = loss_fn(recon, {"latent": latent}, batch_x)
        return loss, {}

    return _run_training_loop(model, loader_train, loader_val, opt, _loss_eval, train_cfg, progress_cfg=progress_cfg)


def _predict_deterministic_archetypal(
    model: DeterministicArchetypalAutoencoder,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    labels_val: np.ndarray | None,
    labels_test: np.ndarray | None,
    cell_ids_val: np.ndarray,
    cell_ids_test: np.ndarray,
    device: torch.device,
) -> dict[str, SplitResult]:
    model.eval()

    def _forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(x) == 0:
            return np.zeros((0, model.archetype_means.shape[1]), dtype=np.float32), np.zeros((0, model.archetype_means.shape[0]), dtype=np.float32)
        with torch.no_grad():
            xt = torch.from_numpy(x.astype(np.float32)).to(device)
            recon, weights = model(xt)
        return recon.cpu().numpy(), weights.cpu().numpy()

    tr_recon, tr_w = _forward(x_train)
    va_recon, va_w = _forward(x_val)
    te_recon, te_w = _forward(x_test)

    return {
        "train": SplitResult("train", x_train, tr_recon, make_unit_logvar(x_train), tr_w, tr_w, None, np.array([f"train_{i}" for i in range(len(x_train))])),
        "val": SplitResult("val", x_val, va_recon, make_unit_logvar(x_val), va_w, va_w, labels_val, cell_ids_val),
        "test": SplitResult("test", x_test, te_recon, make_unit_logvar(x_test), te_w, te_w, labels_test, cell_ids_test),
    }


def _predict_probabilistic_archetypal(
    model: ProbabilisticArchetypalAutoencoder,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    labels_val: np.ndarray | None,
    labels_test: np.ndarray | None,
    cell_ids_val: np.ndarray,
    cell_ids_test: np.ndarray,
    device: torch.device,
) -> dict[str, SplitResult]:
    model.eval()

    def _forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(x) == 0:
            return (
                np.zeros((0, model.n_markers), dtype=np.float32),
                np.zeros((0, model.n_markers), dtype=np.float32),
                np.zeros((0, model.n_archetypes), dtype=np.float32),
            )
        with torch.no_grad():
            xt = torch.from_numpy(x.astype(np.float32)).to(device)
            mean, logvar, weights = model(xt)
        return mean.cpu().numpy(), logvar.cpu().numpy(), weights.cpu().numpy()

    tr_mean, tr_logvar, tr_w = _forward(x_train)
    va_mean, va_logvar, va_w = _forward(x_val)
    te_mean, te_logvar, te_w = _forward(x_test)

    return {
        "train": SplitResult("train", x_train, tr_mean, tr_logvar, tr_w, tr_w, None, np.array([f"train_{i}" for i in range(len(x_train))])),
        "val": SplitResult("val", x_val, va_mean, va_logvar, va_w, va_w, labels_val, cell_ids_val),
        "test": SplitResult("test", x_test, te_mean, te_logvar, te_w, te_w, labels_test, cell_ids_test),
    }


def _predict_ae(
    model: StandardAutoencoder,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    labels_val: np.ndarray | None,
    labels_test: np.ndarray | None,
    cell_ids_val: np.ndarray,
    cell_ids_test: np.ndarray,
    device: torch.device,
) -> dict[str, SplitResult]:
    model.eval()

    def _forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(x) == 0:
            return np.zeros((0, model.decoder[-1].out_features), dtype=np.float32), np.zeros((0, model.encoder[-1].out_features), dtype=np.float32)
        with torch.no_grad():
            xt = torch.from_numpy(x.astype(np.float32)).to(device)
            recon, latent = model(xt)
        return recon.cpu().numpy(), latent.cpu().numpy()

    tr_recon, tr_latent = _forward(x_train)
    va_recon, va_latent = _forward(x_val)
    te_recon, te_latent = _forward(x_test)

    return {
        "train": SplitResult("train", x_train, tr_recon, make_unit_logvar(x_train), tr_latent, None, None, np.array([f"train_{i}" for i in range(len(x_train))])),
        "val": SplitResult("val", x_val, va_recon, make_unit_logvar(x_val), va_latent, None, labels_val, cell_ids_val),
        "test": SplitResult("test", x_test, te_recon, make_unit_logvar(x_test), te_latent, None, labels_test, cell_ids_test),
    }


def _predict_vae(
    model: VariationalAutoencoder,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    labels_val: np.ndarray | None,
    labels_test: np.ndarray | None,
    cell_ids_val: np.ndarray,
    cell_ids_test: np.ndarray,
    device: torch.device,
) -> dict[str, SplitResult]:
    model.eval()

    def _forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(x) == 0:
            return np.zeros((0, model.decoder[-1].out_features), dtype=np.float32), np.zeros((0, model.mu_head.out_features), dtype=np.float32)
        with torch.no_grad():
            xt = torch.from_numpy(x.astype(np.float32)).to(device)
            h = model.encoder(xt)
            mu = model.mu_head(h)
            recon = model.decoder(mu)
        return recon.cpu().numpy(), mu.cpu().numpy()

    tr_recon, tr_latent = _forward(x_train)
    va_recon, va_latent = _forward(x_val)
    te_recon, te_latent = _forward(x_test)

    return {
        "train": SplitResult("train", x_train, tr_recon, make_unit_logvar(x_train), tr_latent, None, None, np.array([f"train_{i}" for i in range(len(x_train))])),
        "val": SplitResult("val", x_val, va_recon, make_unit_logvar(x_val), va_latent, None, labels_val, cell_ids_val),
        "test": SplitResult("test", x_test, te_recon, make_unit_logvar(x_test), te_latent, None, labels_test, cell_ids_test),
    }

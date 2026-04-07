"""
Benchmark PAE training speed on the current device.
Run on the pod after git pull to diagnose epoch time.

Usage:
    python scripts/benchmark_training_speed.py
"""
import time
import torch
from cytof_archetypes.models.probabilistic_archetypal_ae import ProbabilisticArchetypalAutoencoder
from cytof_archetypes.models.losses import (
    gaussian_nll, entropy_penalty, diversity_penalty, variance_regularization,
)
from cytof_archetypes.baselines.neural import _TensorBatchLoader

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else
                      "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

N_MARKERS, K, BATCH = 31, 8, 512
N_TRAIN, N_VAL = 770_000, 165_000
N_TRAIN_BATCHES = (N_TRAIN + BATCH - 1) // BATCH
N_VAL_BATCHES   = (N_VAL   + BATCH - 1) // BATCH

model = ProbabilisticArchetypalAutoencoder(N_MARKERS, K, [128, 64]).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

x_batch = torch.randn(BATCH, N_MARKERS, device=device)

def one_train_step(x):
    opt.zero_grad(set_to_none=True)
    mean, logvar, w = model(x)
    loss = (gaussian_nll(x, mean, logvar)
            + 1e-3  * entropy_penalty(w)
            + 5e-3  * diversity_penalty(model.archetype_means)
            + 1e-4  * variance_regularization(model.archetype_logvars))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss

# ── 1. Single-batch compute ─────────────────────────────────────────────────
print("\n--- 1. Single-batch compute (200 steps, post-warmup) ---")
for _ in range(10):
    one_train_step(x_batch)
sync()

N = 200
t0 = time.time()
for _ in range(N):
    one_train_step(x_batch)
sync()
elapsed = time.time() - t0
ms = elapsed / N * 1000
print(f"  {ms:.2f} ms/batch")
print(f"  Projected epoch (train only, {N_TRAIN_BATCHES} batches): {ms*N_TRAIN_BATCHES/1000:.1f}s")

# ── 2. Full simulated epoch with _TensorBatchLoader ─────────────────────────
print("\n--- 2. Full simulated epoch (train + val, real loader) ---")
x_train = torch.randn(N_TRAIN, N_MARKERS, device=device)
x_val   = torch.randn(N_VAL,   N_MARKERS, device=device)
train_loader = _TensorBatchLoader(x_train, BATCH, shuffle=True)
val_loader   = _TensorBatchLoader(x_val,   BATCH, shuffle=False)

# warmup epoch
model.train()
acc = torch.zeros((), device=device)
for (bx,) in train_loader:
    one_train_step(bx)
sync()

# timed epoch
t0 = time.time()
model.train()
acc = torch.zeros((), device=device)
n = 0
for (bx,) in train_loader:
    acc.add_(one_train_step(bx).detach())
    n += 1
model.eval()
with torch.no_grad():
    vacc = torch.zeros((), device=device)
    vn = 0
    for (bx,) in val_loader:
        mean, logvar, w = model(bx)
        vacc.add_(gaussian_nll(bx, mean, logvar))
        vn += 1
_ = float(acc.item()); _ = float(vacc.item())  # flush
sync()
elapsed = time.time() - t0
print(f"  Train: {N_TRAIN_BATCHES} batches")
print(f"  Val:   {N_VAL_BATCHES} batches")
print(f"  Total epoch time: {elapsed:.2f}s")

# ── 3. Breakdown ─────────────────────────────────────────────────────────────
print("\n--- 3. Breakdown ---")
# data loading only
t0 = time.time()
for _ in range(3):
    for (bx,) in train_loader:
        pass
sync()
load_time = (time.time() - t0) / 3
print(f"  Data loading per epoch: {load_time:.3f}s")
print(f"  Compute per epoch (estimate): {elapsed - load_time:.2f}s")

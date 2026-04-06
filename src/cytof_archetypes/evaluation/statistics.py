from __future__ import annotations

import numpy as np

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
except Exception:  # pragma: no cover - optional dependency fallback
    scipy_wilcoxon = None


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.zeros(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = np.mean(sample)
    alpha = 1.0 - ci
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1.0 - alpha / 2))
    return {"mean": float(np.mean(values)), "ci_low": low, "ci_high": high}


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return {"stat": float("nan"), "p_value": float("nan")}
    if scipy_wilcoxon is None:
        return {"stat": float("nan"), "p_value": float("nan")}
    stat, p = scipy_wilcoxon(x, y, zero_method="wilcox", correction=False)
    return {"stat": float(stat), "p_value": float(p)}


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        candidate = ranked[i] * n / rank
        prev = min(prev, candidate)
        adjusted[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out.tolist()

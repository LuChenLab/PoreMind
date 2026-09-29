from __future__ import annotations

import numpy as np


def rolling_quantile(x: np.ndarray, window: int = 501, q: float = 0.5) -> np.ndarray:
    if window <= 1:
        return np.full_like(x, np.quantile(x, q))
    half = window // 2
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        out[i] = np.quantile(x[lo:hi], q)
    return out


def estimate_baseline(x: np.ndarray, method: str = "rolling_quantile", **kwargs) -> np.ndarray:
    if method == "rolling_quantile":
        return rolling_quantile(x, window=kwargs.get("window", 501), q=kwargs.get("q", 0.5))
    if method == "global_median":
        return np.full_like(x, np.median(x))
    raise ValueError("unsupported baseline method")

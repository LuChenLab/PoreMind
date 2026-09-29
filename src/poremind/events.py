from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


def _noise_scale(residual: np.ndarray, method: str = "mad", stats_mask: np.ndarray | None = None) -> float:
    if stats_mask is not None:
        residual = residual[stats_mask]
    if len(residual) <= 1:
        raise ValueError("effective points for noise estimation must be > 1 after exclude_current filtering")
    method = method.lower()
    if method == "mad":
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        return 1.4826 * mad + 1e-12
    if method == "std":
        return float(np.std(residual)) + 1e-12
    raise ValueError("noise_method must be 'mad' or 'std'")


@dataclass
class Event:
    start_idx: int
    end_idx: int
    baseline_local: float
    delta_i: float
    dwell_time_s: float
    snr: float


def _mask_to_events(mask: np.ndarray, baseline: np.ndarray, signal: np.ndarray, sampling_rate_hz: float, min_duration_s: float) -> List[Event]:
    min_samples = max(1, int(min_duration_s * sampling_rate_hz))
    sigma = float(np.std(signal - baseline)) + 1e-12
    if len(mask) == 0:
        return []

    m = mask.astype(np.int8, copy=False)
    d = np.diff(m)
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if bool(mask[0]):
        starts = np.r_[0, starts]
    if bool(mask[-1]):
        ends = np.r_[ends, len(mask)]

    if len(starts) == 0:
        return []

    dur = ends - starts
    valid = dur >= min_samples
    starts = starts[valid]
    ends = ends[valid]
    if len(starts) == 0:
        return []

    res = baseline - signal
    cumsum_res = np.concatenate(([0.0], np.cumsum(res, dtype=float)))
    cumsum_base = np.concatenate(([0.0], np.cumsum(baseline, dtype=float)))

    events: List[Event] = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        n = e - s
        delta_i = float((cumsum_res[e] - cumsum_res[s]) / n)
        local_base = float((cumsum_base[e] - cumsum_base[s]) / n)
        dwell = n / sampling_rate_hz
        snr = delta_i / sigma
        events.append(Event(s, e, local_base, delta_i, dwell, snr))
    return events


def detect_events_threshold(
    signal: np.ndarray,
    baseline: np.ndarray,
    sampling_rate_hz: float,
    sigma_k: float = 5.0,
    min_duration_s: float = 0.0002,
    noise_method: str = "mad",
    stats_mask: np.ndarray | None = None,
) -> List[Event]:
    residual = signal - baseline
    sigma = _noise_scale(residual, method=noise_method, stats_mask=stats_mask)
    threshold = -sigma_k * sigma
    mask = residual < threshold
    return _mask_to_events(mask, baseline, signal, sampling_rate_hz, min_duration_s)


def detect_events_cusum(
    signal: np.ndarray,
    baseline: np.ndarray,
    sampling_rate_hz: float,
    drift: float = 0.02,
    threshold: float = 8.0,
    min_duration_s: float = 0.0002,
    noise_method: str = "mad",
    stats_mask: np.ndarray | None = None,
) -> List[Event]:
    """One-sided CUSUM on standardized residual for blockade-like (negative) events."""
    residual = signal - baseline
    sigma = _noise_scale(residual, method=noise_method, stats_mask=stats_mask)
    z = residual / sigma

    s_neg = np.zeros_like(z)
    for i in range(1, len(z)):
        s_neg[i] = min(0.0, s_neg[i - 1] + z[i] + drift)
    mask = s_neg < -abs(threshold)
    return _mask_to_events(mask, baseline, signal, sampling_rate_hz, min_duration_s)


def detect_events_pelt(
    signal: np.ndarray,
    baseline: np.ndarray,
    sampling_rate_hz: float,
    model: str = "l2",
    penalty: float = 8.0,
    sigma_k: float = 3.0,
    min_duration_s: float = 0.0002,
    noise_method: str = "mad",
    stats_mask: np.ndarray | None = None,
) -> List[Event]:
    """PELT change-point segmentation + residual thresholding within segments."""
    try:
        import ruptures as rpt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("detect_events_pelt requires ruptures. Install with `pip install ruptures`.") from exc

    residual = signal - baseline
    algo = rpt.Pelt(model=model).fit(residual)
    bkps = algo.predict(pen=penalty)

    mask = np.zeros_like(signal, dtype=bool)
    start = 0
    sigma = _noise_scale(residual, method=noise_method, stats_mask=stats_mask)
    thr = -abs(sigma_k) * sigma
    for end in bkps:
        seg = residual[start:end]
        if len(seg):
            mask[start:end] = seg < thr
        start = end
    return _mask_to_events(mask, baseline, signal, sampling_rate_hz, min_duration_s)


def detect_events_hmm(
    signal: np.ndarray,
    baseline: np.ndarray,
    sampling_rate_hz: float,
    n_components: int = 2,
    covariance_type: str = "diag",
    n_iter: int = 200,
    min_duration_s: float = 0.0002,
) -> List[Event]:
    """Gaussian HMM on residual; state with lower mean is treated as event state."""
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("detect_events_hmm requires hmmlearn. Install with `pip install hmmlearn`.") from exc

    residual = signal - baseline
    X = residual.reshape(-1, 1)
    hmm = GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter, random_state=42)
    states = hmm.fit_predict(X)
    means = hmm.means_.reshape(-1)
    event_state = int(np.argmin(means))
    mask = states == event_state
    return _mask_to_events(mask, baseline, signal, sampling_rate_hz, min_duration_s)

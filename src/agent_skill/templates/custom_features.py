"""Custom feature examples for PoreMind Agent Skill."""

from __future__ import annotations

import numpy as np


def shape_features(seg, row=None):
    """Simple waveform shape features.

    Compatible with PoreMind custom_feature_fns. Returns a dict.
    """
    x = np.asarray(seg, dtype=float)
    if len(x) == 0:
        return {"p2p": np.nan, "energy": np.nan, "abs_area": np.nan}
    return {
        "p2p": float(np.max(x) - np.min(x)),
        "energy": float(np.sum(x ** 2)),
        "abs_area": float(np.sum(np.abs(x))),
    }


def kde_peak_feature(seg, row=None, bandwidth=0.1):
    """Estimate a KDE peak value for peptide-like events.

    Useful for FraC PP0/PP1/PP2 style analysis when peak-based blockade ratio
    is more stable than a simple segment mean.
    """
    x = np.asarray(seg, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return {"kde_peak_value": float(np.mean(x)) if len(x) else np.nan}
    try:
        from scipy.signal import find_peaks
        from scipy.stats import gaussian_kde
    except Exception:
        return {"kde_peak_value": float(np.median(x))}
    kde = gaussian_kde(x, bw_method=bandwidth)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 1000)
    ys = kde(xs)
    peaks, _ = find_peaks(ys)
    if len(peaks) == 0:
        idx = int(np.argmax(ys))
    else:
        idx = int(peaks[np.argmax(ys[peaks])])
    return {"kde_peak_value": float(xs[idx])}


def add_blockade_ratio_peak(feature_df, peak_col="custom_kde_peak_value"):
    """Add blockade_ratio_peak to a PoreMind feature dataframe.

    The exact sign convention depends on the experiment; verify with event plots.
    """
    df = feature_df.copy()
    if peak_col not in df.columns:
        # PoreMind may prefix custom keys with or without custom_. Try fallback.
        fallback = "kde_peak_value"
        if fallback in df.columns:
            peak_col = fallback
        else:
            raise ValueError(f"missing peak column: {peak_col}")
    if "global_baseline" not in df.columns:
        raise ValueError("global_baseline is required")
    baseline = df["global_baseline"].astype(float)
    peak = df[peak_col].astype(float)
    delta_i_feature = -baseline + peak
    df["blockade_ratio_peak"] = delta_i_feature / (-baseline + 1e-12)
    return df

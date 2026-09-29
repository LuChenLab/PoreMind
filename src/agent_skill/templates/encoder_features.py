"""PoreMindWaveformEncoder adapter template.

This file intentionally avoids assuming a final method name on the encoder. When
PoreMind exposes PoreMindWaveformEncoder, update `encode_events_with_poremind_encoder`
to call the real method, keeping the surrounding column and save conventions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _collect_event_waveforms(analysis, df: pd.DataFrame, expand: int = 0, current: str = "denoise") -> list[np.ndarray]:
    signals = analysis.denoised if current == "denoise" else {k: t.current for k, t in analysis.traces.items()}
    waveforms = []
    for _, r in df.iterrows():
        trace_id = str(r["trace_id"])
        sig = np.asarray(signals[trace_id], dtype=float)
        s = max(0, int(r["start_idx"]) - int(expand))
        e = min(len(sig), int(r["end_idx"]) + int(expand))
        waveforms.append(sig[s:e])
    return waveforms


def encode_events_with_poremind_encoder(
    analysis,
    data: str = "filtered",
    encoder_config: dict[str, Any] | None = None,
    prefix: str = "PME",
    expand: int = 0,
) -> pd.DataFrame:
    """Return a dataframe containing PoreMindWaveformEncoder embeddings.

    Update the method-dispatch block once the exact encoder API is finalized.
    """
    from poremind import PoreMindWaveformEncoder

    encoder_config = encoder_config or {}
    encoder = PoreMindWaveformEncoder(**encoder_config)

    df = analysis.filtered_df if data == "filtered" else analysis.feature_df
    if df is None:
        raise ValueError("analysis.filtered_df or analysis.feature_df is required before encoding")

    # Preferred future APIs. The first available method is used.
    if hasattr(encoder, "encode_analysis_events"):
        emb = encoder.encode_analysis_events(analysis, df=df, expand=expand)
    else:
        waveforms = _collect_event_waveforms(analysis, df=df, expand=expand)
        if hasattr(encoder, "transform"):
            emb = encoder.transform(waveforms)
        elif hasattr(encoder, "encode"):
            emb = encoder.encode(waveforms)
        else:
            raise AttributeError(
                "PoreMindWaveformEncoder must provide encode_analysis_events, transform, or encode"
            )

    emb = np.asarray(emb)
    if emb.ndim != 2 or emb.shape[0] != len(df):
        raise ValueError("encoder output must be a 2D array with one row per event")

    cols = [f"{prefix}_{i+1:03d}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, columns=cols, index=df.index)
    return emb_df


def attach_embeddings(analysis, emb_df: pd.DataFrame, data: str = "filtered"):
    if data == "filtered":
        if analysis.filtered_df is None:
            raise ValueError("filtered_df is missing")
        analysis.filtered_df = pd.concat([analysis.filtered_df.copy(), emb_df], axis=1)
    else:
        if analysis.feature_df is None:
            raise ValueError("feature_df is missing")
        analysis.feature_df = pd.concat([analysis.feature_df.copy(), emb_df], axis=1)
    return analysis


def save_embeddings(emb_df: pd.DataFrame, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_csv(path, index=False)
    return str(path)

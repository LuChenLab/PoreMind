from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .baseline import estimate_baseline
from .events import detect_events_threshold
from .features import events_to_dataframe
from .io import read_abf, read_abf_all, read_csv
from .preprocess import preprocess_signal


@dataclass
class AnalysisConfig:
    reader: Literal["abf", "csv"] = "abf"
    preprocess_method: str = "butterworth_filtfilt"
    preprocess_params: dict[str, Any] = field(default_factory=lambda: {"filtfilt_N": 2, "filtfilt_Wn": 0.1})
    baseline_method: str = "rolling_quantile"
    baseline_params: dict[str, Any] = field(default_factory=lambda: {"window": 501, "q": 0.5})
    detect_params: dict[str, Any] = field(default_factory=lambda: {"sigma_k": 5.0, "min_duration_s": 2e-4})


def _trace_to_event_df(trace, config: AnalysisConfig) -> pd.DataFrame:
    pre = preprocess_signal(trace.current, method=config.preprocess_method, **config.preprocess_params)
    baseline = estimate_baseline(pre, method=config.baseline_method, **config.baseline_params)
    events = detect_events_threshold(pre, baseline, trace.sampling_rate_hz, **config.detect_params)
    df = events_to_dataframe(events, trace.time, pre)
    df["source"] = trace.source
    df["sampling_rate_hz"] = trace.sampling_rate_hz
    df["channel"] = trace.channel
    df["sweep"] = trace.sweep
    return df


def analyze_abf_to_event_df(path: str | Path, config: AnalysisConfig | None = None, **reader_kwargs: Any) -> pd.DataFrame:
    config = config or AnalysisConfig()

    if config.reader == "abf":
        # 默认遍历所有 channel + sweep；如传入参数则读取指定单条
        if reader_kwargs:
            trace = read_abf(path, **reader_kwargs)
            return _trace_to_event_df(trace, config)
        traces = read_abf_all(path)
        return pd.concat([_trace_to_event_df(tr, config) for tr in traces], ignore_index=True)

    if config.reader == "csv":
        trace = read_csv(path, **reader_kwargs)
        return _trace_to_event_df(trace, config)

    raise ValueError("reader must be 'abf' or 'csv'")

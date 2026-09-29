from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Trace:
    current: np.ndarray
    sampling_rate_hz: float
    time: np.ndarray
    source: str
    channel: int | None = None
    sweep: int | None = None


def read_abf(path: str | Path, channel: int = 0, sweep: int = 0) -> Trace:
    """Read one (channel, sweep) pair from ABF into a Trace."""
    try:
        import pyabf  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency optional
        raise ImportError("read_abf requires pyabf. Install with `pip install pyabf`.") from exc

    p = Path(path)
    abf = pyabf.ABF(str(p))
    if sweep >= abf.sweepCount:
        raise ValueError(f"sweep {sweep} out of range, total={abf.sweepCount}")
    abf.setSweep(sweepNumber=sweep, channel=channel)
    current = np.asarray(abf.sweepY, dtype=float)
    time = np.asarray(abf.sweepX, dtype=float)
    sampling_rate_hz = float(abf.dataRate)
    return Trace(current=current, sampling_rate_hz=sampling_rate_hz, time=time, source=str(p), channel=channel, sweep=sweep)


def read_abf_all(path: str | Path) -> list[Trace]:
    """Read all channels and sweeps from an ABF file and return a trace list."""
    try:
        import pyabf  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency optional
        raise ImportError("read_abf_all requires pyabf. Install with `pip install pyabf`.") from exc

    p = Path(path)
    abf = pyabf.ABF(str(p))
    traces: list[Trace] = []

    channel_count = len(getattr(abf, "adcNames", [])) or 1
    for channel in range(channel_count):
        for sweep in range(abf.sweepCount):
            abf.setSweep(sweepNumber=sweep, channel=channel)
            traces.append(
                Trace(
                    current=np.asarray(abf.sweepY, dtype=float),
                    time=np.asarray(abf.sweepX, dtype=float),
                    sampling_rate_hz=float(abf.dataRate),
                    source=str(p),
                    channel=channel,
                    sweep=sweep,
                )
            )
    return traces


def read_csv(path: str | Path, current_col: str = "current", time_col: Optional[str] = "time", sampling_rate_hz: Optional[float] = None) -> Trace:
    """Fallback reader for CSV testing and quick prototyping."""
    p = Path(path)
    df = pd.read_csv(p)
    current = df[current_col].to_numpy(dtype=float)

    if time_col and time_col in df.columns:
        time = df[time_col].to_numpy(dtype=float)
        if sampling_rate_hz is None and len(time) > 1:
            dt = np.median(np.diff(time))
            sampling_rate_hz = 1.0 / dt
    else:
        if sampling_rate_hz is None:
            raise ValueError("sampling_rate_hz is required when CSV has no time column")
        time = np.arange(len(current), dtype=float) / sampling_rate_hz

    assert sampling_rate_hz is not None
    return Trace(current=current, sampling_rate_hz=float(sampling_rate_hz), time=time, source=str(p), channel=0, sweep=0)

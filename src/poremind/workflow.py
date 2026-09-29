from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import AdaBoostClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from .baseline import estimate_baseline
from .events import (
    Event,
    detect_events_cusum,
    detect_events_hmm,
    detect_events_pelt,
    detect_events_threshold,
)
from .features import select_feature_columns
from .io import Trace, read_abf, read_abf_all, read_csv
from .pl import PlotAccessor
from .preprocess import preprocess_signal

FeatureFn = Callable[[np.ndarray], dict[str, float]]


import inspect

def call_feature_fn(fn, seg, row=None):
    """
    Safely call custom feature function.

    Supports:
    - fn(seg)
    - fn(seg, row)
    - fn(seg, *args)
    """
    try:
        sig = inspect.signature(fn)
        params = sig.parameters.values()
        has_varargs = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL
            for p in params
        )

        if has_varargs:
            return fn(seg, row)
        try:
            sig.bind(seg, row)
            return fn(seg, row)
        except TypeError:
            return fn(seg)

    except (TypeError, ValueError):
        try:
            return fn(seg, row)
        except TypeError:
            return fn(seg)

@dataclass
class MultiSampleAnalysis:
    sample_paths: dict[str, str | Path]
    sample_to_group: dict[str, str] | None = None
    reader: str = "abf"
    reader_kwargs: dict[str, Any] = field(default_factory=dict)

    traces: dict[str, Trace] = field(default_factory=dict)
    denoised: dict[str, np.ndarray] = field(default_factory=dict)
    baselines: dict[str, np.ndarray] = field(default_factory=dict)
    events: dict[str, list[Event]] = field(default_factory=dict)
    simple_events: dict[str, list[Event]] = field(default_factory=dict)
    detect_events_simple_object: dict[str, list[Event]] = field(default_factory=dict)
    feature_df: pd.DataFrame | None = None
    filtered_df: pd.DataFrame | None = None
    best_model_package: dict[str, Any] | None = None
    DL_model_package: dict[str, Any] | None = None
    DL_model_packages: dict[str, dict[str, Any]] = field(default_factory=dict)
    ml_explain_results: dict[str, Any] = field(default_factory=dict)
    dl_attribute_results: dict[str, Any] = field(default_factory=dict)
    model_cv_results: dict[str, Any] = field(default_factory=dict)

    preprocess_state: dict[str, Any] = field(default_factory=dict)
    detect_state: dict[str, Any] = field(default_factory=dict)
    simple_detect_state: dict[str, Any] = field(default_factory=dict)
    feature_state: dict[str, Any] = field(default_factory=dict)
    trace_to_sample: dict[str, str] = field(default_factory=dict)
    pl: PlotAccessor = field(init=False)
    plot: PlotAccessor = field(init=False)

    def __post_init__(self) -> None:
        self.pl = PlotAccessor(self)
        self.plot = self.pl

    def load(self) -> "MultiSampleAnalysis":
        self.traces = {}
        self.trace_to_sample = {}
        for sid, path in self.sample_paths.items():
            if self.reader == "abf":
                # 默认遍历 ABF 全部 channel + sweep
                if self.reader_kwargs:
                    trace = read_abf(path, **self.reader_kwargs)
                    key = f"{sid}__ch{trace.channel}_sw{trace.sweep}"
                    self.traces[key] = trace
                    self.trace_to_sample[key] = sid
                else:
                    for trace in read_abf_all(path):
                        key = f"{sid}__ch{trace.channel}_sw{trace.sweep}"
                        self.traces[key] = trace
                        self.trace_to_sample[key] = sid
            else:
                trace = read_csv(path, **self.reader_kwargs)
                self.traces[sid] = trace
                self.trace_to_sample[sid] = sid
        return self

    # Step 1: denoise + preview/visualize
    def denoise(self, method: str = "butterworth_filtfilt", **kwargs: Any) -> "MultiSampleAnalysis":
        if not self.traces:
            self.load()
        self.preprocess_state = {"method": method, **kwargs}
        self.denoised = {sid: preprocess_signal(tr.current, method=method, **kwargs) for sid, tr in self.traces.items()}
        return self

    def preview_signal(self, sample_id: str, start_s: float = 0.0, end_s: float | None = None, max_points: int = 5000) -> pd.DataFrame:
        trace = self.traces[sample_id]
        sig = self.denoised.get(sample_id, trace.current)
        t = trace.time
        end_s = float(t[-1]) if end_s is None else end_s
        mask = (t >= start_s) & (t <= end_s)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return pd.DataFrame(columns=["time", "current"])
        if len(idx) > max_points:
            step = max(1, len(idx) // max_points)
            idx = idx[::step]
        return pd.DataFrame({"time": t[idx], "current": sig[idx]})

    def visualize_signal(self, sample_id: str, start_s: float = 0.0, end_s: float | None = None, max_points: int = 5000):
        df = self.preview_signal(sample_id, start_s=start_s, end_s=end_s, max_points=max_points)
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise ImportError("visualize_signal requires matplotlib") from exc
        ax = df.plot(x="time", y="current", figsize=(10, 3), title=f"Signal preview: {sample_id}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Current")
        plt.tight_layout()
        return ax

    # Step 2: event detection (multi-method)
    def detect_events(
        self,
        detect_method: str = "threshold",
        detect_params: dict[str, Any] | None = None,
        baseline_method: str = "rolling_quantile",
        baseline_params: dict[str, Any] | None = None,
        detect_direction: str = "down",
        merge_event: bool = False,
        merge_event_params: dict[str, Any] | None = None,
        exclude_current: bool = True,
        exclude_current_params: dict[str, Any] | None = None,
    ) -> "MultiSampleAnalysis":
        if not self.denoised:
            self.denoise()
        if detect_params is None:
            detect_params = self._default_detect_params(detect_method)
        baseline_params = baseline_params or {"window": 10000, "q": 0.5}
        self.detect_state = {
            "detect_method": detect_method,
            "detect_params": detect_params,
            "baseline_method": baseline_method,
            "baseline_params": baseline_params,
            "detect_direction": detect_direction,
            "merge_event": merge_event,
            "merge_event_params": merge_event_params or {"merge_gap_ms": 0.0},
            "exclude_current": exclude_current,
            "exclude_current_params": exclude_current_params,
        }

        self.baselines = {}
        self.events = {}
        trace_items = list(self.traces.items())
        iterator = trace_items
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(trace_items, desc="detect_events", unit="sample")
        except Exception:
            iterator = trace_items

        for sid, tr in iterator:
            if hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(str(sid))
            sig = self.denoised[sid]
            stats_mask = self._build_stats_mask(
                signal=sig,
                detect_direction=detect_direction,
                exclude_current=exclude_current,
                exclude_current_params=exclude_current_params,
            )
            baseline = self._estimate_baseline(sig, method=baseline_method, baseline_params=baseline_params, stats_mask=stats_mask)
            self.baselines[sid] = baseline
            evts = self._detect_events_by_method(
                sig,
                baseline,
                tr.sampling_rate_hz,
                detect_method=detect_method,
                detect_params=detect_params,
                detect_direction=detect_direction,
                stats_mask=stats_mask,
            )
            if merge_event:
                evts = self._merge_nearby_events(
                    evts,
                    signal=sig,
                    baseline=baseline,
                    sampling_rate_hz=tr.sampling_rate_hz,
                    merge_gap_ms=float((merge_event_params or {"merge_gap_ms": 0.0}).get("merge_gap_ms", 0.0)),
                )
            self.events[sid] = evts
        return self

    @staticmethod
    def _build_stats_mask(
        signal: np.ndarray,
        detect_direction: str,
        exclude_current: bool,
        exclude_current_params: dict[str, Any] | None,
    ) -> np.ndarray:
        if not exclude_current:
            return np.ones(len(signal), dtype=bool)
        params = (exclude_current_params or {}).copy()
        if "min" not in params and "max" not in params:
            if detect_direction == "up":
                params = {"min": None, "max": 0.0}
            else:
                params = {"min": 0.0, "max": None}

        lo = params.get("min")
        hi = params.get("max")
        mask = np.ones(len(signal), dtype=bool)
        if lo is not None:
            mask &= signal > float(lo)
        if hi is not None:
            mask &= signal < float(hi)
        if int(mask.sum()) <= 1:
            raise ValueError("effective points for baseline/noise statistics must be > 1 after exclude_current filtering")
        return mask

    @staticmethod
    def _estimate_baseline(signal: np.ndarray, method: str, baseline_params: dict[str, Any], stats_mask: np.ndarray) -> np.ndarray:
        if method == "global_quantile":
            q = float(baseline_params.get("q", 0.5))
            valid = signal[stats_mask]
            if len(valid) <= 1:
                raise ValueError("effective points for global_quantile baseline must be > 1 after exclude_current filtering")
            val = float(np.quantile(valid, q))
            return np.full_like(signal, val, dtype=float)
        if method == "rolling_quantile":
            if bool(np.all(stats_mask)):
                return estimate_baseline(signal, method="rolling_quantile", **baseline_params)
            window = int(baseline_params.get("window", 10000))
            q = float(baseline_params.get("q", 0.5))
            if window <= 1:
                valid = signal[stats_mask]
                if len(valid) <= 1:
                    raise ValueError("effective points for rolling_quantile baseline must be > 1 after exclude_current filtering")
                val = float(np.quantile(valid, q))
                return np.full_like(signal, val, dtype=float)
            half = window // 2
            out = np.empty_like(signal, dtype=float)
            for i in range(len(signal)):
                lo = max(0, i - half)
                hi = min(len(signal), i + half + 1)
                local_mask = stats_mask[lo:hi]
                if int(local_mask.sum()) <= 1:
                    raise ValueError("rolling_quantile window has <=1 effective points after exclude_current filtering")
                out[i] = np.quantile(signal[lo:hi][local_mask], q)
            return out
        if method == "global_median":
            valid = signal[stats_mask]
            if len(valid) <= 1:
                raise ValueError("effective points for global_median baseline must be > 1 after exclude_current filtering")
            return np.full_like(signal, float(np.median(valid)), dtype=float)
        return estimate_baseline(signal, method=method, **baseline_params)

    @staticmethod
    def _merge_nearby_events(
        events: list[Event],
        signal: np.ndarray,
        baseline: np.ndarray,
        sampling_rate_hz: float,
        merge_gap_ms: float,
    ) -> list[Event]:
        if len(events) <= 1:
            return events
        sigma = float(np.std(signal - baseline)) + 1e-12
        res = baseline - signal
        cumsum_res = np.concatenate(([0.0], np.cumsum(res, dtype=float)))
        cumsum_base = np.concatenate(([0.0], np.cumsum(baseline, dtype=float)))
        gap_samples = max(0, int((merge_gap_ms / 1000.0) * sampling_rate_hz))
        merged: list[Event] = []
        cur_start = events[0].start_idx
        cur_end = events[0].end_idx
        for e in events[1:]:
            if e.start_idx - cur_end <= gap_samples:
                cur_end = max(cur_end, e.end_idx)
            else:
                merged.append(
                    MultiSampleAnalysis._build_event(
                        cur_start,
                        cur_end,
                        signal,
                        baseline,
                        sampling_rate_hz,
                        sigma=sigma,
                        cumsum_res=cumsum_res,
                        cumsum_base=cumsum_base,
                    )
                )
                cur_start, cur_end = e.start_idx, e.end_idx
        merged.append(
            MultiSampleAnalysis._build_event(
                cur_start,
                cur_end,
                signal,
                baseline,
                sampling_rate_hz,
                sigma=sigma,
                cumsum_res=cumsum_res,
                cumsum_base=cumsum_base,
            )
        )
        return merged

    @staticmethod
    def _build_event(
        start_idx: int,
        end_idx: int,
        signal: np.ndarray,
        baseline: np.ndarray,
        sr: float,
        sigma: float | None = None,
        cumsum_res: np.ndarray | None = None,
        cumsum_base: np.ndarray | None = None,
    ) -> Event:
        n = max(1, end_idx - start_idx)
        if cumsum_res is not None and cumsum_base is not None:
            delta_i = float((cumsum_res[end_idx] - cumsum_res[start_idx]) / n)
            baseline_local = float((cumsum_base[end_idx] - cumsum_base[start_idx]) / n)
        else:
            seg_signal = signal[start_idx:end_idx]
            seg_base = baseline[start_idx:end_idx]
            delta_i = float(np.mean(seg_base - seg_signal))
            baseline_local = float(np.mean(seg_base))
        dwell = (end_idx - start_idx) / sr
        if sigma is None:
            sigma = float(np.std(signal - baseline)) + 1e-12
        snr = delta_i / sigma
        return Event(start_idx=start_idx, end_idx=end_idx, baseline_local=baseline_local, delta_i=delta_i, dwell_time_s=dwell, snr=snr)

    @staticmethod
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

    @staticmethod
    def _detect_events_by_method(
        signal: np.ndarray,
        baseline: np.ndarray,
        sampling_rate_hz: float,
        detect_method: str,
        detect_params: dict[str, Any],
        detect_direction: str = "down",
        stats_mask: np.ndarray | None = None,
    ) -> list[Event]:
        if detect_direction not in {"down", "up"}:
            raise ValueError("detect_direction must be 'down' or 'up'")
        work_signal = signal if detect_direction == "down" else -signal
        work_baseline = baseline if detect_direction == "down" else -baseline

        if detect_method == "threshold":
            if detect_direction == "down":
                return detect_events_threshold(signal, baseline, sampling_rate_hz, stats_mask=stats_mask, **detect_params)
            sigma_k = float(detect_params.get("sigma_k", 5.0))
            min_duration_s = float(detect_params.get("min_duration_s", 0.0))
            noise_method = str(detect_params.get("noise_method", "mad"))
            residual = signal - baseline
            sigma = MultiSampleAnalysis._noise_scale(residual, method=noise_method, stats_mask=stats_mask)
            mask = residual > sigma_k * sigma
            return MultiSampleAnalysis._mask_to_events(mask, baseline, signal, sampling_rate_hz, min_duration_s=min_duration_s)
        if detect_method == "zscore_threshold":
            residual = signal - baseline
            noise_method = str(detect_params.get("noise_method", "mad"))
            z = residual / MultiSampleAnalysis._noise_scale(residual, method=noise_method, stats_mask=stats_mask)
            z_thr = float(detect_params.get("z", 4.0))
            min_duration_s = float(detect_params.get("min_duration_s", 0.0))
            mask = z < -z_thr if detect_direction == "down" else z > z_thr
            return MultiSampleAnalysis._mask_to_events(mask, baseline, signal, sampling_rate_hz, min_duration_s=min_duration_s)
        if detect_method == "cusum":
            return detect_events_cusum(work_signal, work_baseline, sampling_rate_hz, stats_mask=stats_mask, **detect_params)
        if detect_method == "pelt":
            return detect_events_pelt(work_signal, work_baseline, sampling_rate_hz, stats_mask=stats_mask, **detect_params)
        if detect_method == "hmm":
            return detect_events_hmm(work_signal, work_baseline, sampling_rate_hz, **detect_params)
        raise ValueError("unsupported detect method")

    def detect_events_simple(
        self,
        detect_method: str = "threshold",
        detect_params: dict[str, Any] | None = None,
        baseline_method: str = "rolling_quantile",
        baseline_params: dict[str, Any] | None = None,
        sample_id: str | None = None,
        current: str = "denoise",
        start_ms: float = 0.0,
        end_ms: float = 1000.0,
        detect_direction: str = "down",
        merge_event: bool = False,
        merge_event_params: dict[str, Any] | None = None,
        exclude_current: bool = True,
        exclude_current_params: dict[str, Any] | None = None,
    ) -> dict[str, list[Event]]:
        if not self.traces:
            self.load()
        if current not in {"denoise", "raw"}:
            raise ValueError("current must be 'denoise' or 'raw'")
        if current == "denoise" and not self.denoised:
            self.denoise()
        if detect_params is None:
            detect_params = self._default_detect_params(detect_method)
        baseline_params = baseline_params or {"window": 10000, "q": 0.5}

        self.simple_detect_state = {
            "detect_method": detect_method,
            "detect_params": detect_params,
            "baseline_method": baseline_method,
            "baseline_params": baseline_params,
            "sample_id": sample_id,
            "current": current,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "detect_direction": detect_direction,
            "merge_event": merge_event,
            "merge_event_params": merge_event_params or {"merge_gap_ms": 0.0},
            "exclude_current": exclude_current,
            "exclude_current_params": exclude_current_params,
        }

        target_ids = [sample_id] if sample_id is not None else list(self.traces.keys())
        out: dict[str, list[Event]] = {}
        iterator = target_ids
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(target_ids, desc="detect_events_simple", unit="sample")
        except Exception:
            iterator = target_ids

        for sid in iterator:
            if hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(str(sid))
            tr = self.traces[sid]
            sig_full = self.denoised[sid] if current == "denoise" else tr.current
            t_ms = tr.time * 1000.0
            idx = np.flatnonzero((t_ms >= start_ms) & (t_ms <= end_ms))
            if len(idx) == 0:
                out[sid] = []
                continue
            lo, hi = int(idx[0]), int(idx[-1]) + 1
            sig = sig_full[lo:hi]
            stats_mask = self._build_stats_mask(
                signal=sig,
                detect_direction=detect_direction,
                exclude_current=exclude_current,
                exclude_current_params=exclude_current_params,
            )
            baseline = self._estimate_baseline(sig, method=baseline_method, baseline_params=baseline_params, stats_mask=stats_mask)
            evts_local = self._detect_events_by_method(
                sig,
                baseline,
                tr.sampling_rate_hz,
                detect_method=detect_method,
                detect_params=detect_params,
                detect_direction=detect_direction,
                stats_mask=stats_mask,
            )
            if merge_event:
                evts_local = self._merge_nearby_events(
                    evts_local,
                    signal=sig,
                    baseline=baseline,
                    sampling_rate_hz=tr.sampling_rate_hz,
                    merge_gap_ms=float((merge_event_params or {"merge_gap_ms": 0.0}).get("merge_gap_ms", 0.0)),
                )
            out[sid] = [
                Event(
                    start_idx=e.start_idx + lo,
                    end_idx=e.end_idx + lo,
                    baseline_local=e.baseline_local,
                    delta_i=e.delta_i,
                    dwell_time_s=e.dwell_time_s,
                    snr=e.snr,
                )
                for e in evts_local
            ]
        self.simple_events = out
        self.detect_events_simple_object = out
        return out

    @staticmethod
    def _default_detect_params(detect_method: str) -> dict[str, Any]:
        defaults = {
            "threshold": {"sigma_k": 5.0, "min_duration_s": 0.0, "noise_method": "mad"},
            "zscore_threshold": {"z": 4.0, "min_duration_s": 0.0, "noise_method": "mad"},
            "cusum": {"drift": 0.02, "threshold": 8.0, "min_duration_s": 0.0, "noise_method": "mad"},
            "pelt": {"model": "l2", "penalty": 8.0, "sigma_k": 3.0, "min_duration_s": 0.0, "noise_method": "mad"},
            "hmm": {"n_components": 2, "covariance_type": "diag", "n_iter": 200, "min_duration_s": 0.0},
        }
        if detect_method not in defaults:
            raise ValueError(f"unsupported detect method: {detect_method}")
        return defaults[detect_method].copy()

    @staticmethod
    def _mask_to_events(mask: np.ndarray, baseline: np.ndarray, signal: np.ndarray, sr: float, min_duration_s: float) -> list[Event]:
        min_samples = max(1, int(min_duration_s * sr))
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

        out: list[Event] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            n = e - s
            delta_i = float((cumsum_res[e] - cumsum_res[s]) / n)
            baseline_local = float((cumsum_base[e] - cumsum_base[s]) / n)
            out.append(Event(s, e, baseline_local, delta_i, n / sr, delta_i / sigma))
        return out

    # Step 3: feature extraction (built-in + custom)
    def extract_features(
        self,
        custom_feature_fns: dict[str, FeatureFn] | None = None,
        max_event_per_sample: int | None = None,
    ) -> pd.DataFrame:
        if not self.events:
            self.detect_events()
        custom_feature_fns = custom_feature_fns or {}
        if max_event_per_sample is not None and max_event_per_sample <= 0:
            raise ValueError("max_event_per_sample must be > 0 or None")
        rows: list[dict[str, Any]] = []

        sample_items = list(self.events.items())
        sample_iterator = sample_items
        detect_direction = str(self.detect_state.get("detect_direction", "down")).lower()
        if detect_direction not in {"down", "up"}:
            detect_direction = "down"
        try:
            from tqdm.auto import tqdm  # type: ignore

            sample_iterator = tqdm(sample_items, desc="extract_features(samples)", unit="sample")
        except Exception:
            sample_iterator = sample_items

        for sid, evts in sample_iterator:
            if hasattr(sample_iterator, "set_postfix_str"):
                sample_iterator.set_postfix_str(str(sid))
            tr = self.traces[sid]
            source_sample_id = self.trace_to_sample.get(sid, sid)
            sig = self.denoised[sid]
            base = self.baselines[sid]
            global_base = float(np.median(base))
            sig_len = len(sig)
            evts_use = evts if max_event_per_sample is None else evts[:max_event_per_sample]

            # prefix sums for fast window means (left/right baseline)
            cumsum_sig = np.concatenate(([0.0], np.cumsum(sig, dtype=float)))

            def _window_mean(lo: int, hi: int, fallback_idx: int) -> float:
                if hi <= lo:
                    return float(base[fallback_idx])
                return float((cumsum_sig[hi] - cumsum_sig[lo]) / (hi - lo))

            event_iterator = evts_use
            try:
                from tqdm.auto import tqdm  # type: ignore

                event_iterator = tqdm(
                    evts_use,
                    desc=f"extract_features(events:{sid})",
                    unit="event",
                    leave=False,
                )
            except Exception:
                event_iterator = evts_use

            for i, e in enumerate(event_iterator):
                if hasattr(event_iterator, "set_postfix_str"):
                    event_iterator.set_postfix_str(f"event={i + 1}")
                seg = sig[e.start_idx:e.end_idx]
                seg_mean = float(np.mean(seg))
                seg_std = float(np.std(seg))
                seg_min = float(np.min(seg))
                seg_max = float(np.max(seg))
                seg_abs_max = float(np.max(np.abs(seg)))

                left_lo = max(0, e.start_idx - 50)
                left_hi = e.start_idx
                right_lo = e.end_idx
                right_hi = min(sig_len, e.end_idx + 50)
                left_base = _window_mean(left_lo, left_hi, e.start_idx)
                right_base = _window_mean(right_lo, right_hi, e.end_idx - 1)

                if detect_direction == "up":
                    delta_i_feature = float((-global_base) - (-seg_mean))
                    blockade_ratio = float(delta_i_feature / ((-global_base) + 1e-12))
                else:
                    delta_i_feature = float(global_base - seg_mean)
                    blockade_ratio = float(delta_i_feature / (global_base + 1e-12))
                centered = seg - seg_mean
                denom = seg_std + 1e-12
                skew = float(np.mean((centered / denom) ** 3))
                kurt = float(np.mean((centered / denom) ** 4))
                rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-12
                peak_factor = float(seg_abs_max / rms)
                row = {
                    "trace_id": sid,
                    "sample_id": source_sample_id,
                    "channel": tr.channel,
                    "sweep": tr.sweep,
                    "event_id": i,
                    "start_idx": e.start_idx,
                    "end_idx": e.end_idx,
                    "start_time_s": float(tr.time[e.start_idx]),
                    "end_time_s": float(tr.time[e.end_idx - 1]),
                    "duration_s": e.dwell_time_s,
                    "delta_i": delta_i_feature,
                    "snr": e.snr,
                    "left_baseline": left_base,
                    "right_baseline": right_base,
                    "global_baseline": global_base,
                    "blockade_ratio": blockade_ratio,
                    "segment_mean": seg_mean,
                    "segment_std": seg_std,
                    "segment_min": seg_min,
                    "segment_max": seg_max,
                    "segment_skew": skew,
                    "segment_kurt": kurt,
                    "peak_factor": peak_factor,
                }
                for name, fn in custom_feature_fns.items():
                    feats = call_feature_fn(fn, seg, row)
                    row.update({f"{name}_{k}": v for k, v in feats.items()})
                if self.sample_to_group and source_sample_id in self.sample_to_group:
                    row["label"] = self.sample_to_group[source_sample_id]
                rows.append(row)

        self.feature_state = {"custom_features": list(custom_feature_fns)}
        self.feature_df = pd.DataFrame(rows)
        return self.feature_df

    def _resolve_dimred_df(self, data: str = "filtered") -> tuple[pd.DataFrame, str]:
        if data == "filtered":
            if self.filtered_df is None:
                self.filter_events()
            assert self.filtered_df is not None
            return self.filtered_df.copy(), "filtered"
        if data == "feature":
            if self.feature_df is None:
                self.extract_features()
            assert self.feature_df is not None
            return self.feature_df.copy(), "feature"
        raise ValueError("data must be 'filtered' or 'feature'")

    def _save_dimred_df(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        if target == "filtered":
            self.filtered_df = df
        elif target == "feature":
            self.feature_df = df
        return df

    def do_pca(
        self,
        feature_cols: list[str] | None = None,
        data: str = "filtered",
        random_state: int = 42,
    ) -> pd.DataFrame:
        df, target = self._resolve_dimred_df(data=data)
        use_cols = list(feature_cols) if feature_cols is not None else select_feature_columns(df)
        missing = [c for c in use_cols if c not in df.columns]
        if missing:
            raise ValueError(f"missing feature columns for PCA: {missing}")
        X = df[use_cols].fillna(0.0).to_numpy(dtype=float)
        if len(X) < 2:
            raise ValueError("PCA requires at least 2 rows")
        pca = PCA(n_components=2, random_state=random_state)
        emb = pca.fit_transform(X)
        df["PC1"] = emb[:, 0]
        df["PC2"] = emb[:, 1]
        return self._save_dimred_df(df, target)

    def do_tsne(
        self,
        feature_cols: list[str] | None = None,
        data: str = "filtered",
        random_state: int = 42,
        perplexity: float = 30.0,
        n_iter: int = 1000,
    ) -> pd.DataFrame:
        df, target = self._resolve_dimred_df(data=data)
        use_cols = list(feature_cols) if feature_cols is not None else select_feature_columns(df)
        missing = [c for c in use_cols if c not in df.columns]
        if missing:
            raise ValueError(f"missing feature columns for TSNE: {missing}")
        X = df[use_cols].fillna(0.0).to_numpy(dtype=float)
        n = len(X)
        if n < 2:
            raise ValueError("TSNE requires at least 2 rows")
        perp = max(1.0, min(float(perplexity), float(n - 1)))
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=perp, max_iter=n_iter, init="pca")
        emb = tsne.fit_transform(X)
        df["TSNE1"] = emb[:, 0]
        df["TSNE2"] = emb[:, 1]
        return self._save_dimred_df(df, target)

    def do_umap(
        self,
        feature_cols: list[str] | None = None,
        data: str = "filtered",
        random_state: int = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> pd.DataFrame:
        try:
            import umap.umap_ as umap  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("do_umap requires umap-learn. Install with `pip install umap-learn`.") from exc

        df, target = self._resolve_dimred_df(data=data)
        use_cols = list(feature_cols) if feature_cols is not None else select_feature_columns(df)
        missing = [c for c in use_cols if c not in df.columns]
        if missing:
            raise ValueError(f"missing feature columns for UMAP: {missing}")
        X = df[use_cols].fillna(0.0).to_numpy(dtype=float)
        if len(X) < 2:
            raise ValueError("UMAP requires at least 2 rows")
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist)
        emb = reducer.fit_transform(X)
        df["UMAP1"] = emb[:, 0]
        df["UMAP2"] = emb[:, 1]
        return self._save_dimred_df(df, target)

    @staticmethod
    def _blockade_gmm_mask(
        df: pd.DataFrame,
        rm_index: np.ndarray | None = None,
        blockade_col: str = "blockade_ratio",
        dwell_col: str = "duration_s",
        n_components: int = 2,
        visualize: bool = False,
        prior_mean: float | None = None,
    ) -> np.ndarray:
        if blockade_col not in df.columns:
            raise ValueError(f"missing blockade column: {blockade_col}")
        if dwell_col not in df.columns:
            raise ValueError(f"missing dwell column: {dwell_col}")

        X = df.copy()
        if rm_index is None:
            rm_index = np.ones(len(X), dtype=bool)

        data = X.loc[rm_index, blockade_col].to_numpy(dtype=float)
        if len(data) < 5:
            return rm_index

        # KDE curve is kept only for optional visualization.
        grid = np.linspace(np.nanmin(data), np.nanmax(data), 256)
        bw = max(1e-6, np.std(data) * 0.2)
        density = np.sum(np.exp(-0.5 * ((grid[:, None] - data[None, :]) / bw) ** 2), axis=1)

        if visualize:
            try:
                import matplotlib.pyplot as plt
            except Exception as exc:  # pragma: no cover
                raise ImportError("blockade_gmm visualization requires matplotlib") from exc
            fig, axs = plt.subplots(2, 1, sharex=False, gridspec_kw={"height_ratios": [1, 3]}, figsize=(4, 5))
            axs[0].plot(grid, density)
            axs[0].set_ylabel("Density")
            axs[1].scatter(X[blockade_col], np.log10(X[dwell_col].to_numpy(dtype=float) + 1e-12), c=rm_index, alpha=0.5, s=4)
            axs[1].set_xlabel(blockade_col)
            axs[1].set_ylabel(f"log10({dwell_col})")
            plt.tight_layout()

        data2 = X.loc[rm_index, blockade_col].to_numpy(dtype=float).reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(data2)
        labels = gmm.predict(data2)

        means = gmm.means_.flatten()
        covariances = np.sqrt(gmm.covariances_).flatten()
        valid_indices = np.where((means >= 0) & (means <= 1.2))[0]
        if len(valid_indices) == 0:
            valid_indices = np.arange(len(means))

        if prior_mean is None:
            if len(valid_indices) > 1:
                target = valid_indices[np.argmin(covariances[valid_indices])]
            else:
                target = valid_indices[0]
        else:
            if len(valid_indices) > 1:
                target = valid_indices[np.argmin(np.abs(means[valid_indices] - prior_mean))]
            else:
                target = valid_indices[0]

        local = rm_index.copy()
        local_idx = np.where(rm_index)[0]
        local[local_idx] = labels == target
        return local

    @staticmethod
    def _knn_background_mask(
        target_df: pd.DataFrame,
        background_df: pd.DataFrame,
        *,
        k: int = 10,
        feature_cols: list[str] | None = None,
        n_noise_match: int = 1,
        background_ratio: float = 1.0,
        metric: str = "euclidean",
        random_state: int = 42,
        background_blockage_lim: tuple[float, float] | None = None,
        blockade_col: str = "blockade_ratio",
    ) -> np.ndarray:
        """KNN-based background-noise filtering.

        For each target event, look at the k nearest neighbors in the union pool
        (target events + sampled background events). Mark as noise when the number
        of background-pool neighbors among the k nearest is >= ``n_noise_match``.

        Returns a boolean mask aligned with ``target_df`` rows (True = valid).
        """
        import warnings

        n_target = len(target_df)
        if n_target == 0:
            return np.ones(0, dtype=bool)

        if feature_cols is None:
            feature_cols = ["duration_s", "blockade_ratio"]
        for col in feature_cols:
            if col not in target_df.columns:
                raise ValueError(f"target_df missing feature column: {col}")

        bg = background_df
        if background_blockage_lim is not None and len(bg) > 0:
            if blockade_col not in bg.columns:
                raise ValueError(f"background_df missing blockade column: {blockade_col}")
            lo, hi = float(background_blockage_lim[0]), float(background_blockage_lim[1])
            vals = bg[blockade_col].to_numpy(dtype=float)
            bg = bg.loc[(vals >= lo) & (vals <= hi)]

        n_bg_raw = len(bg)
        if n_bg_raw == 0 or float(background_ratio) <= 0:
            warnings.warn(
                f"knn_background: background pool empty or background_ratio<=0 "
                f"(n_bg_raw={n_bg_raw}, background_ratio={background_ratio}); "
                f"all {n_target} target events kept as valid.",
                UserWarning,
            )
            return np.ones(n_target, dtype=bool)

        k = max(1, int(k))
        n_noise_match = max(1, min(int(n_noise_match), k))
        n_bg_target = max(1, int(round(float(background_ratio) * n_target)))

        if n_bg_raw >= n_bg_target:
            sampled_bg = bg.sample(n=n_bg_target, replace=False, random_state=int(random_state))
        else:
            sampled_bg = bg.sample(n=n_bg_target, replace=True, random_state=int(random_state))
            warnings.warn(
                f"knn_background: background pool smaller than requested size "
                f"({n_bg_raw} < {n_bg_target}); sampling with replacement.",
                UserWarning,
            )

        if float(background_ratio) < 1.0:
            warnings.warn(
                f"knn_background: background_ratio={background_ratio} < 1; "
                f"background pool will be smaller than target.",
                UserWarning,
            )

        tgt_X = target_df[feature_cols].to_numpy(dtype=float)
        bg_X = sampled_bg[feature_cols].to_numpy(dtype=float)
        pool_X = np.vstack([tgt_X, bg_X])
        is_background = np.concatenate([
            np.zeros(n_target, dtype=bool),
            np.ones(n_bg_target, dtype=bool),
        ])

        n_neighbors = min(k + 1, len(pool_X))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(pool_X)
        _, indices = nbrs.kneighbors(tgt_X)

        valid_mask = np.ones(n_target, dtype=bool)
        for i in range(n_target):
            nbr_idx = indices[i, 1:]
            n_bg_in_nbrs = int(is_background[nbr_idx].sum())
            if n_bg_in_nbrs >= n_noise_match:
                valid_mask[i] = False
        return valid_mask

    # Step 4: noise/outlier flagging
    def filter_events(
        self,
        method: str = "blockade_gmm",
        parameters: dict[str, Any] | None = None,
        blockage_lim: tuple[float, float] = (0.1, 1.0),
    ) -> None:
        if self.feature_df is None:
            self.extract_features()
        assert self.feature_df is not None
        df = self.feature_df.copy()

        method = method.lower()
        default_params: dict[str, dict[str, Any]] = {
            "blockade_gmm": {
                "blockade_col": "blockade_ratio",
                "dwell_col": "duration_s",
                "rm_index": None,
                "n_components": 2,
                "prior_mean": None,
                "visualize": False,
            },
            "peak_detection": {
                "blockade_col": "blockade_ratio",
                "dwell_col": "duration_s",
                "rm_index": None,
                "n_components": 2,
                "prior_mean": None,
                "visualize": False,
            },
            "isolation_forest": {
                "contamination": 0.05,
                "feature_cols": ["duration_s", "blockade_ratio", "segment_skew", "segment_kurt"],
            },
            "lof": {
                "contamination": 0.05,
                "feature_cols": ["duration_s", "blockade_ratio", "segment_skew", "segment_kurt"],
            },
            "knn_background": {
                "background_sample_ids": [],
                "k": 10,
                "feature_cols": ["duration_s", "blockade_ratio", "segment_skew", "segment_kurt"],
                "n_noise_match": 1,
                "background_ratio": 1.0,
                "metric": "euclidean",
                "random_state": 42,
                "background_blockage_lim": None,
            },
        }
        if method not in default_params:
            raise ValueError("unsupported outlier method")
        cfg = default_params[method].copy()
        if parameters:
            cfg.update(parameters)

        if len(df) == 0:
            df["is_noise"] = []
            df["quality_tag"] = []
            self.feature_df = df
            self.filtered_df = df.copy()
            return

        group_col = "sample_id" if "sample_id" in df.columns else ("trace_id" if "trace_id" in df.columns else None)
        if group_col is None:
            group_slices: list[tuple[str, pd.Index]] = [("__all__", df.index)]
        else:
            group_slices = [(str(sample_key), idx) for sample_key, idx in df.groupby(group_col).groups.items()]

        rm_series = None
        rm_index = cfg.get("rm_index")
        if rm_index is not None:
            rm_index = np.asarray(rm_index, dtype=bool)
            if len(rm_index) != len(df):
                raise ValueError("rm_index length must match feature dataframe rows")
            rm_series = pd.Series(rm_index, index=df.index)

        blockade_col = str(cfg.get("blockade_col", "blockade_ratio"))
        if blockade_col not in df.columns:
            raise ValueError(f"missing blockade column for blockage_lim filtering: {blockade_col}")
        lo, hi = float(blockage_lim[0]), float(blockage_lim[1])
        in_blockage_lim = (df[blockade_col].to_numpy(dtype=float) >= lo) & (df[blockade_col].to_numpy(dtype=float) <= hi)
        in_lim_series = pd.Series(in_blockage_lim, index=df.index)

        # Hard threshold first: out-of-range events are fixed as noise.
        df["is_noise"] = ~in_blockage_lim
        background_df_pre: pd.DataFrame | None = None
        bg_ids_set: set[str] = set()
        if method == "knn_background":
            bg_ids_cfg = cfg.get("background_sample_ids") or []
            if not bg_ids_cfg:
                raise ValueError("knn_background requires non-empty 'background_sample_ids'")
            if group_col is None:
                raise ValueError("knn_background requires per-sample events (sample_id/trace_id column missing)")
            bg_ids_set = {str(x) for x in bg_ids_cfg}
            bg_mask = df[group_col].isin(bg_ids_set)
            background_df_pre = df.loc[bg_mask].copy()

        # 背景 sample 不参与 KNN 过滤，但仍保留在 feature_df / filtered_df 中（视为 valid）
        group_slices_to_use = group_slices
        if method == "knn_background" and bg_ids_set:
            group_slices_to_use = [(k, idx) for k, idx in group_slices if k not in bg_ids_set]

        for sample_key, idx in group_slices_to_use:
            idx_in = idx[in_lim_series.loc[idx].to_numpy(dtype=bool)]
            if len(idx_in) == 0:
                continue
            sub_df = df.loc[idx_in].copy()
            sub_rm_index = None if rm_series is None else rm_series.loc[idx_in].to_numpy(dtype=bool)
            sample_prior_mean: float | None
            prior_mean_cfg = cfg.get("prior_mean")
            if isinstance(prior_mean_cfg, dict):
                sample_prior_mean = prior_mean_cfg.get(sample_key)
            else:
                sample_prior_mean = prior_mean_cfg

            if method in {"blockade_gmm", "peak_detection"}:
                valid_mask = self._blockade_gmm_mask(
                    sub_df,
                    rm_index=sub_rm_index,
                    blockade_col=str(cfg["blockade_col"]),
                    dwell_col=str(cfg["dwell_col"]),
                    n_components=int(cfg["n_components"]),
                    visualize=bool(cfg["visualize"]),
                    prior_mean=sample_prior_mean,
                )
                df.loc[idx_in, "is_noise"] = ~valid_mask
            elif method == "knn_background":
                valid_mask = self._knn_background_mask(
                    sub_df,
                    background_df_pre if background_df_pre is not None else df.iloc[0:0],
                    k=int(cfg["k"]),
                    feature_cols=list(cfg["feature_cols"]),
                    n_noise_match=int(cfg["n_noise_match"]),
                    background_ratio=float(cfg["background_ratio"]),
                    metric=str(cfg["metric"]),
                    random_state=int(cfg["random_state"]),
                    background_blockage_lim=cfg.get("background_blockage_lim"),
                    blockade_col=str(cfg.get("blockade_col", "blockade_ratio")),
                )
                df.loc[idx_in, "is_noise"] = ~valid_mask
            else:
                local_feature_cols = list(cfg.get("feature_cols") or select_feature_columns(sub_df))
                X = sub_df[local_feature_cols].fillna(0.0)
                if method == "isolation_forest":
                    detector = IsolationForest(contamination=float(cfg["contamination"]), random_state=42)
                    pred = detector.fit_predict(X)
                elif method == "lof":
                    detector = LocalOutlierFactor(contamination=float(cfg["contamination"]))
                    pred = detector.fit_predict(X)
                else:
                    raise ValueError("unsupported outlier method")
                df.loc[idx_in, "is_noise"] = pred == -1

        df["quality_tag"] = np.where(~df["is_noise"], "valid", "noise")
        self.feature_df = df
        self.filtered_df = df[df["quality_tag"] == "valid"].copy()
        return

    # Step 5: model selection (10-fold)
    @staticmethod
    def _metric_config(y: pd.Series) -> dict[str, Any]:
        classes = sorted(pd.unique(y))
        if len(classes) == 2:
            return {"average": "binary", "pos_label": classes[-1], "mode": "binary"}
        return {"average": "macro", "pos_label": None, "mode": "macro"}

    @staticmethod
    def _score_value(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray, cfg: dict[str, Any]) -> float:
        if metric_name == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if metric_name == "f1":
            if cfg["average"] == "binary":
                return float(f1_score(y_true, y_pred, average="binary", pos_label=cfg["pos_label"], zero_division=0))
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if metric_name == "recall":
            if cfg["average"] == "binary":
                return float(recall_score(y_true, y_pred, average="binary", pos_label=cfg["pos_label"], zero_division=0))
            return float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        raise ValueError("unsupported metric name")

    def _evaluate_model_cv(self, est: Any, X: pd.DataFrame, y: pd.Series, cv: int) -> dict[str, Any]:
        cfg = self._metric_config(y)
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        folds: list[dict[str, Any]] = []

        all_labels = sorted(pd.unique(y))
        zero_cm = np.zeros((len(all_labels), len(all_labels)), dtype=int)

        for fold_id, (tr_idx, te_idx) in enumerate(splitter.split(X, y), start=1):
            model = clone(est)
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            try:
                model.fit(X_tr, y_tr)
                pred_tr = model.predict(X_tr)
                pred_te = model.predict(X_te)

                fold = {
                    "fold": fold_id,
                    "train_n": int(len(y_tr)),
                    "test_n": int(len(y_te)),
                    "train_f1": self._score_value("f1", y_tr.to_numpy(), pred_tr, cfg),
                    "train_accuracy": self._score_value("accuracy", y_tr.to_numpy(), pred_tr, cfg),
                    "train_recall": self._score_value("recall", y_tr.to_numpy(), pred_tr, cfg),
                    "test_f1": self._score_value("f1", y_te.to_numpy(), pred_te, cfg),
                    "test_accuracy": self._score_value("accuracy", y_te.to_numpy(), pred_te, cfg),
                    "test_recall": self._score_value("recall", y_te.to_numpy(), pred_te, cfg),
                    "train_cm": confusion_matrix(y_tr, pred_tr, labels=all_labels),
                    "test_cm": confusion_matrix(y_te, pred_te, labels=all_labels),
                    "fit_error": None,
                }
            except Exception as exc:
                fold = {
                    "fold": fold_id,
                    "train_n": int(len(y_tr)),
                    "test_n": int(len(y_te)),
                    "train_f1": np.nan,
                    "train_accuracy": np.nan,
                    "train_recall": np.nan,
                    "test_f1": np.nan,
                    "test_accuracy": np.nan,
                    "test_recall": np.nan,
                    "train_cm": zero_cm.copy(),
                    "test_cm": zero_cm.copy(),
                    "fit_error": str(exc),
                }
            folds.append(fold)

        train_total = sum(f["train_n"] for f in folds)
        test_total = sum(f["test_n"] for f in folds)

        def wavg(key: str, split_total: int, n_key: str) -> float:
            items = [(f[key], f[n_key]) for f in folds if not np.isnan(f[key])]
            if not items:
                return float("nan")
            denom = max(1, sum(w for _, w in items))
            return float(sum(v * w for v, w in items) / denom)

        agg = {
            "train_n_total": train_total,
            "test_n_total": test_total,
            "train_f1_weighted": wavg("train_f1", train_total, "train_n"),
            "train_accuracy_weighted": wavg("train_accuracy", train_total, "train_n"),
            "train_recall_weighted": wavg("train_recall", train_total, "train_n"),
            "test_f1_weighted": wavg("test_f1", test_total, "test_n"),
            "test_accuracy_weighted": wavg("test_accuracy", test_total, "test_n"),
            "test_recall_weighted": wavg("test_recall", test_total, "test_n"),
            "average_mode": cfg["mode"],
        }
        return {"folds": folds, "aggregate": agg, "labels": all_labels}

    def build_best_model(
        self,
        models: dict[str, Any] | None = None,
        label_col: str = "label",
        feature_cols: list[str] | None = None,
        cv: int = 10,
        scoring: str = "accuracy",
        exclude_noise: bool = True,
    ) -> dict[str, Any]:
        if self.filtered_df is None:
            self.filter_events()
        assert self.filtered_df is not None
        df = self.filtered_df
        if exclude_noise and "is_noise" in df.columns:
            df = df[~df["is_noise"]].copy()
        if label_col not in df.columns:
            raise ValueError("label column missing, provide sample_to_group at init or add labels manually")

        default_feature_cols = ["duration_s", "blockade_ratio", "segment_std", "segment_skew", "segment_kurt"]
        feature_cols = list(feature_cols) if feature_cols is not None else default_feature_cols
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature columns missing in dataframe: {missing}")
        X = df[feature_cols].fillna(0.0)
        y = df[label_col]

        models = models or {
            "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
            "SVM": SVC(probability=True, random_state=42),
            "Neural Network": MLPClassifier(max_iter=500, random_state=42),
            "Elastic Net": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=500, random_state=42, n_jobs=-1),
            "Lasso": LogisticRegression(penalty="l1", solver="saga", max_iter=500, random_state=42, n_jobs=-1),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "LDA": LDA(),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
        }

        scoring_key_map = {
            "f1": "test_f1_weighted",
            "f1_macro": "test_f1_weighted",
            "accuracy": "test_accuracy_weighted",
            "accuracy_macro": "test_accuracy_weighted",
            "recall": "test_recall_weighted",
            "recall_macro": "test_recall_weighted",
        }
        score_key = scoring_key_map.get(scoring, "test_accuracy_weighted")

        if self.model_cv_results is None:
            self.model_cv_results = {}
        scores: dict[str, float] = {}
        best_name = None
        best_score = -np.inf
        best_estimator = None
        trained_models: dict[str, Any] = {}

        for name, est in models.items():
            cv_result = self._evaluate_model_cv(est, X, y, cv=cv)
            self.model_cv_results[name] = cv_result
            s = float(cv_result["aggregate"][score_key])
            scores[name] = s
            try:
                fitted = clone(est)
                fitted.fit(X, y)
                trained_models[name] = fitted
            except Exception:
                pass
            if not np.isnan(s) and s > best_score:
                best_score = s
                best_name = name
                best_estimator = clone(est)

        if best_estimator is None or best_name is None:
            raise RuntimeError("all candidate models failed during CV; check labels/class balance")
        best_estimator.fit(X, y)
        all_feature_df = self.feature_df.copy() if self.feature_df is not None else df.copy()
        all_X = all_feature_df[feature_cols].fillna(0.0)
        all_pred = best_estimator.predict(all_X)
        summary_cols = []
        for c in ["trace_id", "sample_id", label_col]:
            if c in all_feature_df.columns:
                summary_cols.append(c)
        all_samples_feature_pred = all_feature_df[summary_cols + feature_cols].copy()
        all_samples_feature_pred["best_model_pred"] = all_pred
        self.best_model_package = {
            "model": best_estimator,
            "feature_cols": feature_cols,
            "label_col": label_col,
            "scores": scores,
            "best_model": best_name,
            "models": trained_models,
            "cv_results": self.model_cv_results,
            "all_samples_feature_pred": all_samples_feature_pred,
            "preprocess_state": self.preprocess_state,
            "detect_state": self.detect_state,
            "feature_state": self.feature_state,
        }
        return self.best_model_package

    @staticmethod
    def _interp_signal(x: np.ndarray, target_length: int = 500) -> np.ndarray:
        try:
            from scipy import interpolate
        except Exception as exc:  # pragma: no cover
            raise ImportError("signal interpolation requires scipy") from exc
        l = len(x)
        if l <= 1:
            return np.full(target_length, float(x[0]) if l == 1 else 0.0, dtype=float)
        xp = np.linspace(0, l - 1, l)
        f = interpolate.interp1d(xp, x, kind="slinear")
        x_new = np.linspace(0, l - 1, target_length)
        y_new = f(x_new)
        return np.asarray(np.around(y_new, 4), dtype=float)

    @staticmethod
    def _pad_signal(x: np.ndarray, target_length: int = 500) -> np.ndarray:
        """Zero-pad short segments or center-truncate long segments."""
        if target_length <= 0:
            raise ValueError("target_length must be > 0")
        x = np.asarray(x, dtype=float)
        length = len(x)
        if length == target_length:
            return x.astype(np.float32, copy=False)
        if length > target_length:
            start = (length - target_length) // 2
            return x[start : start + target_length].astype(np.float32, copy=False)
        out = np.zeros(target_length, dtype=float)
        out[:length] = x
        return out.astype(np.float32, copy=False)

    @staticmethod
    def _scale_signal(
        x: np.ndarray,
        mode: str | None = "mad",
        baseline: float | None = None,
    ) -> np.ndarray:
        if mode is None or mode == "none":
            return x

        if mode == "mad":
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med))) + 1e-12
            return (x - med) / (1.4826 * mad)

        if mode == "minmax":
            lo, hi = float(np.min(x)), float(np.max(x))
            if hi - lo <= 1e-12:
                return np.zeros_like(x)
            return (x - lo) / (hi - lo)

        if mode == "blockade":
            if baseline is None:
                baseline = float(np.median(x))
            return (baseline - x) / (baseline + 1e-12)

        raise ValueError("scale mode must be one of: 'mad', 'minmax', 'blockade', 'none'")

    def _build_dl_inputs(
        self,
        df: pd.DataFrame,
        interp_length: int,
        expand: int,
        scale: str | None,
        feature_cols: list[str] | None,
        interp_method: str = "interp",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if interp_method not in {"interp", "padding"}:
            raise ValueError("interp_method must be 'interp' or 'padding'")
        if "trace_id" not in df.columns or "start_idx" not in df.columns or "end_idx" not in df.columns:
            raise ValueError("filtered dataframe must include trace_id/start_idx/end_idx for DL training")
        seqs: list[np.ndarray] = []
        for _, r in df.iterrows():
            trace_id = str(r["trace_id"])
            if trace_id not in self.denoised:
                raise ValueError(f"trace_id not found in denoised signals: {trace_id}")
            sig = self.denoised[trace_id]
            s = max(0, int(r["start_idx"]) - int(expand))
            e = min(len(sig), int(r["end_idx"]) + int(expand))
            seg = np.asarray(sig[s:e], dtype=float)

            if scale == "blockade":
                if "global_baseline" in r:
                    baseline = float(r["global_baseline"])
                elif "left_baseline" in r and "right_baseline" in r:
                    baseline = float((r["left_baseline"] + r["right_baseline"]) / 2.0)
                else:
                    baseline = float(np.median(seg))
            else:
                baseline = None
            seg = self._scale_signal(seg, mode=scale, baseline=baseline)

            if interp_method == "interp":
                seqs.append(self._interp_signal(seg, target_length=interp_length))
            else:
                seqs.append(self._pad_signal(seg, target_length=interp_length))
        X_seq = np.stack(seqs, axis=0).astype(np.float32)
        X_feat: np.ndarray | None = None
        if feature_cols is not None:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"feature columns missing for DL model: {missing}")
            X_feat = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        return X_seq, X_feat

    def build_DL_model(
        self,
        model: Any | None = None,
        model_name: str = "residual-CNN",
        feature_cols: list[str] | None = None,
        interp_length: int = 500,
        interp_method: str = "interp",
        expand: int = 50,
        scale: str | None = "blockade",
        device: str = "cuda",
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        epoch: int = 30,
        early_stop_patience: int = 5,
        cv: int = 10,
        label_col: str = "label",
    ) -> dict[str, Any]:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as exc:  # pragma: no cover
            raise ImportError("build_DL_model requires PyTorch") from exc

        if interp_method not in {"interp", "padding"}:
            raise ValueError("interp_method must be 'interp' or 'padding'")

        is_magjam = model_name in {"MAGJAM", "MAGJAM_d4"}
        # MAGJAM consumes waveform inputs only.  An empty list is intentional:
        # the existing len==0 fallback below converts it to None without
        # allowing the default handcrafted feature list to be reintroduced.
        if is_magjam:
            feature_cols = []

        if self.filtered_df is None:
            self.filter_events()
        assert self.filtered_df is not None
        df = self.filtered_df.copy()
        if label_col not in df.columns:
            raise ValueError("label column missing for DL model training")
        if feature_cols is None:
            feature_cols = ["duration_s", "blockade_ratio", "segment_std", "segment_skew", "segment_kurt"]
        if feature_cols is not None and len(feature_cols) == 0:
            feature_cols = None

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        dev = torch.device(device)

        X_seq, X_feat = self._build_dl_inputs(
            df,
            interp_length=interp_length,
            expand=expand,
            scale=scale,
            feature_cols=feature_cols,
            interp_method=interp_method,
        )
        classes = sorted(pd.unique(df[label_col]))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.asarray([class_to_idx[v] for v in df[label_col].tolist()], dtype=np.int64)

        n_classes = len(classes)
        if n_classes < 2:
            raise ValueError("DL model requires at least 2 classes")

        class DefaultConvExtractor(nn.Module):
            def __init__(
                self,
                out_dim: int = 32,
                dropout: float = 0.2,
            ):
                super().__init__()

                class ConvBlock(nn.Module):
                    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
                        super().__init__()
                        padding = kernel_size // 2
                        self.block = nn.Sequential(
                            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
                            nn.BatchNorm1d(out_ch),
                            nn.ReLU(inplace=True),
                            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
                            nn.BatchNorm1d(out_ch),
                        )

                        self.shortcut = (
                            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
                            if in_ch != out_ch
                            else nn.Identity()
                        )

                        self.act = nn.ReLU(inplace=True)

                    def forward(self, x):
                        return self.act(self.block(x) + self.shortcut(x))

                self.net = nn.Sequential(
                    ConvBlock(1, 32, kernel_size=7),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),

                    ConvBlock(32, 64, kernel_size=5),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),

                    ConvBlock(64, 128, kernel_size=3),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),

                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),

                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),

                    nn.Linear(64, out_dim),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.net(x)

        magjam_model_obj = None
        if is_magjam and model is None:
            from .magjam import MAGJAM_MODELS, MAGJAMExtractor

            magjam_model_obj = MAGJAM_MODELS[model_name](n_classes=n_classes, T=interp_length)
            extractor = MAGJAMExtractor(magjam_model_obj)
        else:
            extractor = model if model is not None else DefaultConvExtractor(out_dim=10)
        feat_dim = int(X_feat.shape[1]) if X_feat is not None else 0
        if not hasattr(extractor, "forward"):
            raise ValueError("model must be a torch module/feature extractor with forward")
        was_training = extractor.training
        extractor.eval()

        with torch.no_grad():
            dummy = torch.zeros(2, 1, interp_length, dtype=torch.float32)
            conv_dim = int(extractor(dummy).shape[1])

        if was_training:
            extractor.train()
        if magjam_model_obj is not None:
            if feat_dim != 0:
                raise RuntimeError("MAGJAM must not receive handcrafted feature columns")
            if conv_dim != int(magjam_model_obj.feat_dim):
                raise RuntimeError(
                    "MAGJAM extractor/head dimension mismatch: "
                    f"extractor={conv_dim}, head={magjam_model_obj.feat_dim}"
                )
            head = magjam_model_obj.head
        else:
            head = nn.Linear(conv_dim + feat_dim, n_classes)
        full_model = nn.ModuleDict({"extractor": extractor, "head": head}).to(dev)

        def _forward(x_seq_t, x_feat_t):
            z = full_model["extractor"](x_seq_t)
            if x_feat_t is not None:
                z = torch.cat([z, x_feat_t], dim=1)
            return full_model["head"](z)

        criterion = nn.CrossEntropyLoss()
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        folds: list[dict[str, Any]] = []
        fold_losses: list[dict[str, list[float]]] = []
        best_state = None
        best_fold_score = -np.inf

        for fold_id, (tr_idx, te_idx) in enumerate(splitter.split(X_seq, y_idx), start=1):
            tr_sub, val_sub = train_test_split(tr_idx, test_size=0.1, random_state=42, stratify=y_idx[tr_idx] if len(np.unique(y_idx[tr_idx])) > 1 else None)

            optimizer = torch.optim.Adam(full_model.parameters(), lr=learning_rate)
            full_model.train()
            tr_ds = TensorDataset(torch.from_numpy(X_seq[tr_sub]).unsqueeze(1), torch.from_numpy(y_idx[tr_sub]))
            val_ds = TensorDataset(torch.from_numpy(X_seq[val_sub]).unsqueeze(1), torch.from_numpy(y_idx[val_sub]))
            te_ds = TensorDataset(torch.from_numpy(X_seq[te_idx]).unsqueeze(1), torch.from_numpy(y_idx[te_idx]))
            if X_feat is not None:
                tr_ds = TensorDataset(torch.from_numpy(X_seq[tr_sub]).unsqueeze(1), torch.from_numpy(X_feat[tr_sub]), torch.from_numpy(y_idx[tr_sub]))
                val_ds = TensorDataset(torch.from_numpy(X_seq[val_sub]).unsqueeze(1), torch.from_numpy(X_feat[val_sub]), torch.from_numpy(y_idx[val_sub]))
                te_ds = TensorDataset(torch.from_numpy(X_seq[te_idx]).unsqueeze(1), torch.from_numpy(X_feat[te_idx]), torch.from_numpy(y_idx[te_idx]))

            tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

            best_val = float("inf")
            patience = 0
            best_epoch_state = {k: v.detach().cpu().clone() for k, v in full_model.state_dict().items()}
            train_loss_hist: list[float] = []
            val_loss_hist: list[float] = []
            epoch_iter = range(epoch)
            try:
                from tqdm.auto import tqdm  # type: ignore

                epoch_iter = tqdm(epoch_iter, desc=f"build_DL_model:{model_name}:fold{fold_id}", leave=False)
            except Exception:
                epoch_iter = range(epoch)

            for _ in epoch_iter:
                full_model.train()
                tr_losses = []
                for batch in tr_loader:
                    optimizer.zero_grad()
                    if X_feat is None:
                        x_seq_b, y_b = batch
                        logits = _forward(x_seq_b.to(dev), None)
                    else:
                        x_seq_b, x_feat_b, y_b = batch
                        logits = _forward(x_seq_b.to(dev), x_feat_b.to(dev))
                    loss = criterion(logits, y_b.to(dev))
                    loss.backward()
                    optimizer.step()
                    tr_losses.append(float(loss.detach().cpu()))
                tr_loss = float(np.mean(tr_losses)) if tr_losses else np.nan

                full_model.eval()
                with torch.no_grad():
                    val_losses = []
                    for batch in val_loader:
                        if X_feat is None:
                            x_seq_b, y_b = batch
                            logits = _forward(x_seq_b.to(dev), None)
                        else:
                            x_seq_b, x_feat_b, y_b = batch
                            logits = _forward(x_seq_b.to(dev), x_feat_b.to(dev))
                        val_losses.append(float(criterion(logits, y_b.to(dev)).detach().cpu()))
                val_loss = float(np.mean(val_losses)) if val_losses else np.nan
                train_loss_hist.append(tr_loss)
                val_loss_hist.append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    patience = 0
                    best_epoch_state = {k: v.detach().cpu().clone() for k, v in full_model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= early_stop_patience:
                        break

            full_model.load_state_dict(best_epoch_state)
            full_model.eval()
            y_te = y_idx[te_idx]
            preds = []
            probs = []
            with torch.no_grad():
                for batch in te_loader:
                    if X_feat is None:
                        x_seq_b, _ = batch
                        logits = _forward(x_seq_b.to(dev), None)
                    else:
                        x_seq_b, x_feat_b, _ = batch
                        logits = _forward(x_seq_b.to(dev), x_feat_b.to(dev))
                    p = torch.softmax(logits, dim=1)
                    probs.append(p.detach().cpu().numpy())
                    preds.append(torch.argmax(p, dim=1).detach().cpu().numpy())
            pred_idx = np.concatenate(preds)
            prob_arr = np.concatenate(probs, axis=0)
            test_acc = float(accuracy_score(y_te, pred_idx))
            fold = {
                "fold": fold_id,
                "train_n": int(len(tr_sub)),
                "test_n": int(len(te_idx)),
                "test_accuracy": test_acc,
                "test_f1": float(f1_score(y_te, pred_idx, average="macro", zero_division=0)),
                "test_recall": float(recall_score(y_te, pred_idx, average="macro", zero_division=0)),
                "test_cm": confusion_matrix(y_te, pred_idx, labels=np.arange(n_classes)),
                "fit_error": None,
            }
            folds.append(fold)
            fold_losses.append({"train_loss": train_loss_hist, "val_loss": val_loss_hist})
            if test_acc > best_fold_score:
                best_fold_score = test_acc
                best_state = {k: v.detach().cpu().clone() for k, v in full_model.state_dict().items()}

        if best_state is None:
            raise RuntimeError("DL training failed")
        full_model.load_state_dict(best_state)
        full_model.eval()

        # full-data prediction and write back to filtered_df
        with torch.no_grad():
            x_seq_all = torch.from_numpy(X_seq).unsqueeze(1).to(dev)
            if X_feat is None:
                logits_all = _forward(x_seq_all, None)
            else:
                x_feat_all = torch.from_numpy(X_feat).to(dev)
                logits_all = _forward(x_seq_all, x_feat_all)
            prob_all = torch.softmax(logits_all, dim=1).detach().cpu().numpy()
            pred_all = np.argmax(prob_all, axis=1)

        suffix = model_name
        out_df = df.copy()
        out_df[f"pred_label_{suffix}"] = [classes[i] for i in pred_all]
        for i, cls in enumerate(classes):
            out_df[f"pred_proba_{cls}_{suffix}"] = prob_all[:, i]
        self.filtered_df = out_df

        agg = {
            "test_accuracy_weighted": float(np.nanmean([f["test_accuracy"] for f in folds])),
            "test_f1_weighted": float(np.nanmean([f["test_f1"] for f in folds])),
            "test_recall_weighted": float(np.nanmean([f["test_recall"] for f in folds])),
        }
        self.model_cv_results[model_name] = {"folds": folds, "aggregate": agg, "labels": classes, "fold_losses": fold_losses, "type": "dl"}
        self.DL_model_package = {
            "model_name": model_name,
            "model": full_model,
            "model_state_dict": {k: v.detach().cpu().clone() for k, v in full_model.state_dict().items()},
            "classes": classes,
            "class_to_idx": class_to_idx,
            "feature_cols": feature_cols,
            "interp_length": interp_length,
            "interp_method": interp_method,
            "expand": expand,
            "scale": scale,
            "device": str(dev),
            "cv_results": self.model_cv_results[model_name],
        }
        self.DL_model_packages[model_name] = self.DL_model_package
        return self.DL_model_package

    def explain_ml(
        self,
        model_name: str | None = None,
        feature_cols: list[str] | None = None,
        data: str = "filtered",
        scoring: str = "accuracy",
        n_repeats: int = 10,
        random_state: int = 42,
        width: float = 5.0,
        height: float = 4.0,
    ) -> dict[str, Any]:
        """Explain a trained sklearn model with permutation feature importance."""
        if self.best_model_package is None:
            raise ValueError("run build_best_model before explain_ml")
        if data == "filtered":
            df = self.filtered_df if self.filtered_df is not None else self.feature_df
        elif data == "full":
            df = self.feature_df
        else:
            raise ValueError("data must be 'filtered' or 'full'")
        if df is None:
            raise ValueError("no feature dataframe found; run extract_features/filter_events first")

        if model_name is None:
            model_name = str(self.best_model_package["best_model"])
        models = self.best_model_package.get("models", {})
        if model_name == self.best_model_package.get("best_model"):
            model = self.best_model_package["model"]
        elif model_name in models:
            model = models[model_name]
        else:
            raise ValueError(f"model_name not found in trained ML models: {model_name}")

        label_col = str(self.best_model_package.get("label_col", "label"))
        if label_col not in df.columns:
            raise ValueError(f"label column not found for explain_ml: {label_col}")
        feature_cols = list(feature_cols) if feature_cols is not None else list(self.best_model_package["feature_cols"])
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature columns missing for explain_ml: {missing}")

        try:
            import matplotlib.pyplot as plt
            from sklearn.inspection import permutation_importance
        except Exception as exc:  # pragma: no cover
            raise ImportError("analysis.explain_ml requires matplotlib and scikit-learn") from exc

        X = df[feature_cols].fillna(0.0)
        y = df[label_col]
        imp = permutation_importance(
            model,
            X,
            y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        ranking = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance_mean": imp.importances_mean,
                "importance_std": imp.importances_std,
            }
        ).sort_values("importance_mean", ascending=False, ignore_index=True)
        ranking["rank"] = np.arange(1, len(ranking) + 1)

        fig, ax = plt.subplots(figsize=(width, height))
        plot_df = ranking.iloc[::-1]
        ax.barh(plot_df["feature"], plot_df["importance_mean"], xerr=plot_df["importance_std"], color="#4C78A8")
        ax.set_xlabel(f"Permutation importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Feature importance ({model_name})")
        plt.tight_layout()

        result = {
            "ranking": ranking,
            "ax": ax,
            "model_name": model_name,
            "method": "permutation_importance",
            "scoring": scoring,
        }
        self.ml_explain_results[model_name] = result
        return result

    def explain_dl(
        self,
        model_name: str = "residual-CNN",
        target: str | int = "pred",
        method: str = "integrated_gradients",
        baseline: str = "zero",
        batch_size: int = 32,
        n_steps: int = 50,
        data: str = "filtered",
        device: str | None = None,
    ) -> None:
        """Store Captum per-position attributions for a trained DL model."""
        if method != "integrated_gradients":
            raise ValueError("method must be 'integrated_gradients'")
        if baseline not in {"zero", "mean"}:
            raise ValueError("baseline must be 'zero' or 'mean'")
        if model_name not in self.DL_model_packages:
            raise ValueError(f"model_name not found in trained DL models: {model_name}")
        if data == "filtered":
            df = self.filtered_df
        elif data == "full":
            df = self.feature_df
        else:
            raise ValueError("data must be 'filtered' or 'full'")
        if df is None:
            raise ValueError("no dataframe found for explain_dl")

        try:
            import torch
            import torch.nn as nn
            from captum.attr import IntegratedGradients
            from tqdm import tqdm
        except Exception as exc:  # pragma: no cover
            raise ImportError("analysis.explain_dl requires PyTorch and Captum") from exc

        dl_pkg = self.DL_model_packages[model_name]
        dev = torch.device(device or dl_pkg.get("device", "cpu"))
        model_obj = dl_pkg["model"]
        model_obj.to(dev)
        model_obj.eval()

        feature_cols = dl_pkg["feature_cols"]
        X_seq, X_feat = self._build_dl_inputs(
            df,
            interp_length=int(dl_pkg["interp_length"]),
            expand=int(dl_pkg["expand"]),
            scale=dl_pkg["scale"],
            feature_cols=feature_cols,
            interp_method=str(dl_pkg.get("interp_method", "interp")),
        )
        classes = list(dl_pkg["classes"])
        class_to_idx = dict(dl_pkg["class_to_idx"])

        class DLForward(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, x_seq_t, x_feat_t=None):
                z = self.wrapped["extractor"](x_seq_t)
                if x_feat_t is not None:
                    z = torch.cat([z, x_feat_t], dim=1)
                return self.wrapped["head"](z)

        forward_model = DLForward(model_obj).to(dev)
        ig = IntegratedGradients(forward_model)

        all_attr: list[np.ndarray] = []
        all_pred_idx: list[np.ndarray] = []
        all_target_idx: list[np.ndarray] = []
        for start in tqdm(range(0, len(X_seq), batch_size), desc=f"method use {method}:{model_name}", leave=False):
            end = min(len(X_seq), start + batch_size)
            x_seq_t = torch.from_numpy(X_seq[start:end]).unsqueeze(1).to(dev)
            x_feat_t = torch.from_numpy(X_feat[start:end]).to(dev) if X_feat is not None else None
            if baseline == "zero":
                base = torch.zeros_like(x_seq_t)
            else:
                base = x_seq_t.mean(dim=0, keepdim=True).expand_as(x_seq_t).clone()

            with torch.no_grad():
                logits = forward_model(x_seq_t, x_feat_t)
                pred_idx = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            if target == "pred":
                target_idx = pred_idx
            elif isinstance(target, int):
                target_idx = torch.full_like(pred_idx, int(target))
            else:
                if target not in class_to_idx:
                    raise ValueError(f"target class not found: {target}")
                target_idx = torch.full_like(pred_idx, int(class_to_idx[target]))

            attr = ig.attribute(
                x_seq_t,
                baselines=base,
                target=target_idx,
                additional_forward_args=x_feat_t,
                n_steps=n_steps,
            )
            all_attr.append(attr.detach().cpu().numpy()[:, 0, :])
            all_pred_idx.append(pred_idx.detach().cpu().numpy())
            all_target_idx.append(target_idx.detach().cpu().numpy())

        attr_arr = np.concatenate(all_attr, axis=0)
        pred_idx_arr = np.concatenate(all_pred_idx, axis=0)
        target_idx_arr = np.concatenate(all_target_idx, axis=0)

        metadata = pd.DataFrame(index=np.arange(len(df)))
        metadata["row_index"] = list(df.index)
        for c in ["sample_id", "trace_id", "event_id"]:
            metadata[c] = df[c].to_numpy() if c in df.columns else np.nan
        metadata["pred_class"] = [classes[int(i)] for i in pred_idx_arr]
        metadata["target_class"] = [classes[int(i)] for i in target_idx_arr]

        self.dl_attribute_results[model_name] = {
            "model_name": model_name,
            "method": method,
            "baseline": baseline,
            "target": target,
            "interp_length": int(dl_pkg["interp_length"]),
            "interp_method": str(dl_pkg.get("interp_method", "interp")),
            "expand": int(dl_pkg["expand"]),
            "scale": dl_pkg["scale"],
            "feature_cols": feature_cols,
            "inputs": X_seq,
            "attributions": attr_arr,
            "metadata": metadata,
        }
        return None

    # Step 6: reuse current pipeline + trained model on new samples
    def classify_new_samples(
        self,
        new_sample_paths: dict[str, str | Path],
        reader: str | None = None,
        reader_kwargs: dict[str, Any] | None = None,
        custom_feature_fns: dict[str, FeatureFn] | None = None,
        model: str | Any | None = None,
    ) -> tuple["MultiSampleAnalysis", pd.DataFrame]:
        if self.best_model_package is None and self.DL_model_package is None:
            self.build_best_model()

        other = MultiSampleAnalysis(
            sample_paths=new_sample_paths,
            sample_to_group=None,
            reader=reader or self.reader,
            reader_kwargs=reader_kwargs or self.reader_kwargs,
        )
        preprocess_kwargs = self.preprocess_state.copy()
        if "kwargs" in preprocess_kwargs and isinstance(preprocess_kwargs["kwargs"], dict):
            nested = preprocess_kwargs.pop("kwargs")
            preprocess_kwargs.update(nested)

        other.load().denoise(**preprocess_kwargs).detect_events(**self.detect_state)
        features = other.extract_features(custom_feature_fns=custom_feature_fns)

        # resolve selected model
        selected_model_name = None
        selected_model_obj = None
        selected_feature_cols = None
        dl_pkg = None
        if isinstance(model, str):
            selected_model_name = model
        elif model is not None:
            selected_model_obj = model

        if selected_model_name is None and selected_model_obj is None:
            if self.best_model_package is not None:
                selected_model_name = str(self.best_model_package["best_model"])
            elif self.DL_model_package is not None:
                selected_model_name = str(self.DL_model_package["model_name"])

        if selected_model_name is not None and self.best_model_package is not None:
            models = self.best_model_package.get("models", {})
            if selected_model_name in models:
                selected_model_obj = models[selected_model_name]
                selected_feature_cols = list(self.best_model_package["feature_cols"])

        if selected_model_name is not None and selected_model_obj is None:
            if selected_model_name in self.DL_model_packages:
                dl_pkg = self.DL_model_packages[selected_model_name]
            elif self.DL_model_package is not None and selected_model_name == self.DL_model_package.get("model_name"):
                dl_pkg = self.DL_model_package

        out = features.copy()
        if dl_pkg is not None:
            try:
                import torch
            except Exception as exc:  # pragma: no cover
                raise ImportError("DL prediction requires PyTorch") from exc
            X_seq, X_feat = other._build_dl_inputs(
                features,
                interp_length=int(dl_pkg["interp_length"]),
                expand=int(dl_pkg["expand"]),
                scale=dl_pkg["scale"],
                feature_cols=dl_pkg["feature_cols"],
                interp_method=str(dl_pkg.get("interp_method", "interp")),
            )
            model_obj = dl_pkg["model"]
            model_obj.eval()
            dev = torch.device(dl_pkg.get("device", "cpu"))
            model_obj.to(dev)
            with torch.no_grad():
                x_seq_t = torch.from_numpy(X_seq).unsqueeze(1).to(dev)
                if X_feat is None:
                    z = model_obj["extractor"](x_seq_t)
                    logits = model_obj["head"](z)
                else:
                    x_feat_t = torch.from_numpy(X_feat).to(dev)
                    z = model_obj["extractor"](x_seq_t)
                    logits = model_obj["head"](torch.cat([z, x_feat_t], dim=1))
                p = torch.softmax(logits, dim=1).detach().cpu().numpy()
                pred_idx = np.argmax(p, axis=1)
            classes = dl_pkg["classes"]
            suffix = dl_pkg["model_name"]
            out[f"pred_label_{suffix}"] = [classes[i] for i in pred_idx]
            for i, cls in enumerate(classes):
                out[f"pred_proba_{cls}_{suffix}"] = p[:, i]
        else:
            if selected_model_obj is None:
                raise ValueError("no valid model found. Train with build_best_model/build_DL_model or pass `model` explicitly.")
            if selected_feature_cols is None:
                if self.best_model_package is not None:
                    selected_feature_cols = list(self.best_model_package["feature_cols"])
                else:
                    raise ValueError("feature columns for selected model are unknown")
            X = features[selected_feature_cols].fillna(0.0)
            pred = selected_model_obj.predict(X)
            out["pred_label"] = pred
            if hasattr(selected_model_obj, "predict_proba"):
                proba = selected_model_obj.predict_proba(X)
                classes = getattr(selected_model_obj, "classes_", np.arange(proba.shape[1]))
                for i, cls in enumerate(classes):
                    out[f"pred_proba_{cls}"] = proba[:, i]
        other.feature_df = out.copy()
        return other, out


def create_analysis_object(
    sample_paths: dict[str, str | Path],
    sample_to_group: dict[str, str] | None = None,
    reader: str = "abf",
    reader_kwargs: dict[str, Any] | None = None,
) -> MultiSampleAnalysis:
    """Factory for building a multi-sample, step-by-step analysis object."""
    return MultiSampleAnalysis(sample_paths=sample_paths, sample_to_group=sample_to_group, reader=reader, reader_kwargs=reader_kwargs or {})

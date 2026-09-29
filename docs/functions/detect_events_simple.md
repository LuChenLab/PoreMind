# `MultiSampleAnalysis.detect_events_simple`
**Module:** `workflow.py`

Run event detection on a selected time window for quick method selection and parameter tuning.

## Parameters

- `detect_method` (`str`): `threshold`, `zscore_threshold`, `cusum`, `pelt`, or `hmm`; default `"threshold"`.
- `detect_params` (`dict[str, Any] | None`): Method-specific parameters; defaults are selected automatically.
- `baseline_method` (`str`): `"rolling_quantile"` by default; also supports `"global_quantile"` and `"global_median"`.
- `baseline_params` (`dict[str, Any] | None`): Workflow fallback `{"window": 10000, "q": 0.5}`; `global_quantile` accepts `{"q": ...}`.
- `sample_id` (`str | None`): Trace to process; `None` processes all loaded traces.
- `current` (`str`): `"denoise"` (default) or `"raw"`.
- `start_ms` / `end_ms` (`float`): Window bounds in milliseconds; defaults `0.0` and `1000.0`.
- `detect_direction` (`str`): `"down"` (default) or `"up"`.
- `merge_event` (`bool`): Whether to merge adjacent events; default `False`.
- `merge_event_params` (`dict[str, Any] | None`): Use `{"merge_gap_ms": ...}` for the merge gap.
- `exclude_current` (`bool`): Whether to mask current values during baseline/noise estimation; default `True`.
- `exclude_current_params` (`dict[str, Any] | None`): Statistical bounds, with direction-dependent defaults when omitted.

## Returns

`dict[str, list[Event]]`, keyed by trace id. Event indices are mapped back to full-trace indices.

## Side effects

- Stores the result in `analysis.detect_events_simple_object` and the compatibility mirror `analysis.simple_events`.
- Stores settings in `analysis.simple_detect_state`.
- Uses a progress bar when `tqdm` is available and raises `ValueError` when too few effective statistics points remain.


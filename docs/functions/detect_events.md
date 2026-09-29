# `MultiSampleAnalysis.detect_events`
**Module:** `workflow.py`

Estimate a baseline and detect events on every loaded/denoised trace. For local parameter tuning, use `detect_events_simple` first.

## Parameters

- `detect_method` (`str`): `threshold`, `zscore_threshold`, `cusum`, `pelt`, or `hmm`; default `"threshold"`.
- `detect_params` (`dict[str, Any] | None`): Method-specific parameters. If omitted, method defaults are used.
- `baseline_method` (`str`): `"rolling_quantile"` by default; also supports `"global_quantile"` and `"global_median"`.
- `baseline_params` (`dict[str, Any] | None`): Workflow fallback is `{"window": 10000, "q": 0.5}`. For `global_quantile`, use `{"q": ...}`; for `global_median`, no parameter is required.
- `detect_direction` (`str`): `"down"` by default or `"up"`.
- `merge_event` (`bool`): Merge nearby events when `True`; default `False`.
- `merge_event_params` (`dict[str, Any] | None`): Use `{"merge_gap_ms": ...}` when merging.
- `exclude_current` (`bool`): Use a current mask for baseline/noise statistics; default `True`.
- `exclude_current_params` (`dict[str, Any] | None`): Bounds in the form `{"min": ..., "max": ...}`. When omitted, defaults are direction-dependent (`up`: below 0; `down`: above 0).

## Returns

`MultiSampleAnalysis`. Events are stored in `analysis.events`, baselines in `analysis.baselines`, and settings in `analysis.detect_state`.

## Notes

- A `tqdm` progress bar is used when available.
- At most one effective point after current masking raises `ValueError`.
- `global_quantile` is the Agent Skill policy default, but the Python method default remains `rolling_quantile`.


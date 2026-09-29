# `estimate_baseline`
**Module:** `baseline.py`

Estimate a baseline array with the low-level baseline dispatcher. The higher-level workflow additionally supports `global_quantile` and applies its statistical mask before calling this helper.

## Parameters

- `x` (`np.ndarray`): Input signal.
- `method` (`str`): `"rolling_quantile"` (default) or `"global_median"`.
- `**kwargs` (`Any`): For `rolling_quantile`, `window` defaults to `501` and `q` to `0.5`.

## Returns

`np.ndarray` with the same length as `x`.


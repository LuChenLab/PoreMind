# `remove_slow_drift`
**Module:** `preprocess.py`

Removes slow baseline drift using moving-average baseline subtraction.

## Parameters
- `x` (`np.ndarray`): Input signal.
- `window` (`int`): Baseline window length.

## Returns
- `np.ndarray`

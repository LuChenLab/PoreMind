# `butterworth_filtfilt`
**Module:** `preprocess.py`

Apply zero-phase Butterworth filtering with `scipy.signal.filtfilt`.

## Parameters

- `x` (`np.ndarray`): Input signal.
- `filtfilt_N` (`int`): Filter order, default `2`.
- `filtfilt_Wn` (`float`): Normalized cutoff frequency, default `0.1`.

## Returns

`np.ndarray`. Requires `scipy`.


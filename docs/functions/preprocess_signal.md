# `preprocess_signal`
**Module:** `preprocess.py`

Dispatch a one-dimensional signal to a supported preprocessing function.

## Parameters

- `x` (`np.ndarray`): Input signal.
- `method` (`str`): `butterworth_filtfilt` by default. Other options are `none`, `moving_average`, `median`, and `drift_corrected_moving_average`.
- `**kwargs` (`Any`): Method-specific arguments.

## Returns

`np.ndarray`.


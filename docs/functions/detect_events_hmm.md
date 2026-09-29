# `detect_events_hmm`
**Module:** `events.py`

Low-level Gaussian HMM detector that treats the state with the lower residual mean as the event state.

## Parameters

- `signal`, `baseline` (`np.ndarray`): Signal and baseline arrays.
- `sampling_rate_hz` (`float`): Sampling rate.
- `n_components` (`int`): Number of states, default `2`.
- `covariance_type` (`str`): HMM covariance type, default `"diag"`.
- `n_iter` (`int`): Maximum EM iterations, default `200`.
- `min_duration_s` (`float`): Minimum duration, default `0.0002` seconds.

## Returns

`list[Event]`. Requires `hmmlearn`.


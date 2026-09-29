# `detect_events_cusum`
**Module:** `events.py`

Low-level one-sided CUSUM detector for negative residual/blockade events.

## Parameters

- `signal`, `baseline` (`np.ndarray`): Signal and baseline arrays.
- `sampling_rate_hz` (`float`): Sampling rate.
- `drift` (`float`): CUSUM drift, default `0.02`.
- `threshold` (`float`): Trigger threshold, default `8.0`.
- `min_duration_s` (`float`): Minimum duration, default `0.0002` seconds.
- `noise_method` (`str`): `mad` (default) or `std`.
- `stats_mask` (`np.ndarray | None`): Optional mask for noise estimation.

## Returns

`list[Event]`.


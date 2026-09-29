# `detect_events_threshold`
**Module:** `events.py`

Low-level threshold detector for negative residual/blockade events.

## Parameters

- `signal`, `baseline` (`np.ndarray`): Signal and baseline arrays.
- `sampling_rate_hz` (`float`): Sampling rate.
- `sigma_k` (`float`): Noise multiplier, default `5.0`.
- `min_duration_s` (`float`): Minimum event duration, default `0.0002` seconds.
- `noise_method` (`str`): `mad` (default) or `std`.
- `stats_mask` (`np.ndarray | None`): Optional mask for noise estimation.

## Returns

`list[Event]`.


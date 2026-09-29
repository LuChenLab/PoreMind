# `detect_events_pelt`
**Module:** `events.py`

Low-level PELT change-point segmentation followed by residual thresholding.

## Parameters

- `signal`, `baseline` (`np.ndarray`): Signal and baseline arrays.
- `sampling_rate_hz` (`float`): Sampling rate.
- `model` (`str`): Ruptures cost model, default `"l2"`.
- `penalty` (`float`): PELT penalty, default `8.0`.
- `sigma_k` (`float`): Residual multiplier, default `3.0`.
- `min_duration_s` (`float`): Minimum duration, default `0.0002` seconds.
- `noise_method` (`str`): `mad` (default) or `std`.
- `stats_mask` (`np.ndarray | None`): Optional mask for noise estimation.

## Returns

`list[Event]`. Requires `ruptures`.


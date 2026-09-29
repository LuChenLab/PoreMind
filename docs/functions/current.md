# `PlotAccessor.current`
**Module:** `pl.py`

Plot raw or denoised current over a time interval.

## Parameters

- `sample_id` (`str | None`): Trace id; `None` selects the first trace.
- `current` (`str`): `"denoise"` (default) or `"raw"`.
- `start_ms` / `end_ms` (`float`): Plot range in milliseconds; defaults `0.0` and `1.0`.
- `width` / `height` (`float`): Figure dimensions; defaults `10.0` and `3.0`.

## Returns

`matplotlib Axes`.


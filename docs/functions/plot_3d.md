# `PlotAccessor.plot_3d`
**Module:** `pl.py`

Create a 3D feature scatter with optional categorical or numeric coloring.

## Parameters

- `x` / `y` / `z` (`str`): Axis columns; defaults `blockade_ratio`, `duration_s`, and `segment_std`.
- `xlim` / `ylim` / `zlim` (`tuple[float, float] | None`): Optional axis limits.
- `x_log2` (`bool`): Log transform x; default `False`.
- `y_log2` (`bool`): Log transform y; default `True`.
- `z_log2` (`bool`): Log transform z; default `False`.
- `data` (`str`): `"filtered"` (default) or `"feature"`.
- `value` (`str | None`): Optional column used for coloring.
- `select_sample_id` (`str | None`): Optional sample filter.
- `width` / `height` (`float`): Figure dimensions; defaults `7.0` and `5.0`.

## Returns

`matplotlib Axes3D`.


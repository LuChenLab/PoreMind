# `PlotAccessor.plot_2d`
**Module:** `pl.py`

Create a 2D feature scatter with optional categorical or numeric coloring.

## Parameters

- `x` / `y` (`str`): Axis columns; defaults `blockade_ratio` and `duration_s`.
- `xlim` / `ylim` (`tuple[float, float] | None`): Optional axis limits.
- `x_log2` (`bool`): Log transform x; default `False`.
- `y_log2` (`bool`): Log transform y; default `True`.
- `data` (`str`): `"filtered"` (default) or `"feature"`.
- `value` (`str | None`): Optional column used for coloring.
- `select_sample_id` (`str | None`): Optional sample filter.
- `width` / `height` (`float`): Figure dimensions; defaults `6.0` and `4.5`.
- `size` (`float`): Scatter point size; default `6`.
- `title` (`str | None`): Optional title.

## Returns

`matplotlib Axes`.


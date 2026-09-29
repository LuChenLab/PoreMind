# `PlotAccessor.stacked_bar`
**Module:** `pl.py`

Plot category proportions grouped by sample or another column.

## Parameters

- `group_col` (`str`): Grouping column, default `"sample_id"`.
- `value_col` (`str`): Category column, default `"pred_label"`.
- `data` (`str`): `"filtered"` (default) or `"feature"`.
- `label_color` (`dict[str, str] | None`): Optional category colors.
- `cmap` (`str`): Color map, default `"tab20"`.
- `width` / `height` (`float`): Figure dimensions; defaults `8.0` and `4.5`.

## Returns

`matplotlib Axes`.


# `PlotAccessor.box_significance`
**Module:** `pl.py`

Plot grouped boxplots and test each group against a reference group.

## Parameters

- `group_col` / `value_col` (`str`): Group and numeric columns; defaults `label` and `blockade_ratio`.
- `data` (`str`): `"filtered"` (default) or `"feature"`.
- `method` (`str`): `"ttest"` (default) or `"ranksum"`.
- `label_color` (`dict[str, str] | None`): Optional group colors.
- `cmap` (`str`): Color map, default `"tab20"`.
- `reference_group` (`str | None`): Reference; default is the group with the highest median.
- `line_offset` / `line_height` (`float`): Significance-line spacing factors; defaults `0.02` and `1.7`.
- `ylim` (`tuple[float, float] | None`): Optional y range.
- `log2` (`bool`): Transform values with `log2(abs(x)+1e-12)` before testing; default `False`.
- `width` / `height` (`float`): Figure dimensions; defaults `6.0` and `5.0`.

## Returns

`matplotlib Axes`.


# `PlotAccessor.model_cm`
**Module:** `pl.py`

Plots aggregated row-wise confusion matrix percentage across folds for a selected model.

## Parameters
- `model_name` (`str`): Model key in CV results.
- `split` (`str`): train or test.
- `width` (`float`): Figure width.
- `height` (`float`): Figure height.
- `cmap` (`str`): Matplotlib colormap name. Default `Reds`.
- `decimals` (`int`): Decimal places for cell annotations. Default `1`.

## Returns
- `matplotlib Axes`

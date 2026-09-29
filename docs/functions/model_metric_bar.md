# `PlotAccessor.model_metric_bar`
**Module:** `pl.py`

Plots weighted metric bars across models in descending order, with gradient colors and value labels.

## Parameters
- `metric` (`str`): f1, accuracy, or recall.
- `split` (`str`): train or test.
- `width` (`float`): Figure width.
- `height` (`float`): Figure height.
- `cmap` (`str`): Matplotlib colormap name for score gradient. Default `RdBu_r` (high score red, low score blue).
- `decimals` (`int`): Decimal places shown above each bar. Default `3`.
- `y_lim` (`tuple[float, float]`): y-axis limits for score display. Default `(0.0, 1.1)`.

## Returns
- `matplotlib Axes`

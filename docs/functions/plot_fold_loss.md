# `PlotAccessor.plot_fold_loss`
**Module:** `pl.py`

Plot train/validation loss curves for each fold from a DL model recorded in `model_cv_results`.

## Parameters
- `model_name` (`str`): Model key in `model_cv_results`.
- `type` (`str`): `train` or `val` fold loss to display (default `train`).
- `width` (`float`): Figure width.
- `height` (`float`): Figure height.

## Returns
- `matplotlib Axes`

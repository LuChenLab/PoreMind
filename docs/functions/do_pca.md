# `MultiSampleAnalysis.do_pca`
**Module:** `workflow.py`

Compute 2D PCA embedding on selected feature columns and append `PC1`, `PC2` to the target table.

## Parameters
- `feature_cols` (`list[str] | None`): Feature columns used for PCA.
- `data` (`str`): Target table: `filtered` (default) or `feature`.
- `random_state` (`int`): Random state for PCA.

## Returns
- `pandas.DataFrame` updated table containing `PC1`, `PC2`.

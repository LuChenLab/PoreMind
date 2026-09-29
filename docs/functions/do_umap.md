# `MultiSampleAnalysis.do_umap`
**Module:** `workflow.py`

Compute 2D UMAP embedding on selected feature columns and append `UMAP1`, `UMAP2` to the target table.

## Parameters
- `feature_cols` (`list[str] | None`): Feature columns used for UMAP.
- `data` (`str`): Target table: `filtered` (default) or `feature`.
- `random_state` (`int`): Random state.
- `n_neighbors` (`int`): UMAP neighbors.
- `min_dist` (`float`): UMAP min distance.

## Returns
- `pandas.DataFrame` updated table containing `UMAP1`, `UMAP2`.

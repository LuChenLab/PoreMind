# `MultiSampleAnalysis.do_tsne`
**Module:** `workflow.py`

Compute 2D t-SNE embedding on selected feature columns and append `TSNE1`, `TSNE2` to the target table.

## Parameters
- `feature_cols` (`list[str] | None`): Feature columns used for t-SNE.
- `data` (`str`): Target table: `filtered` (default) or `feature`.
- `random_state` (`int`): Random state.
- `perplexity` (`float`): t-SNE perplexity (auto-clipped by sample size).
- `n_iter` (`int`): t-SNE iterations.

## Returns
- `pandas.DataFrame` updated table containing `TSNE1`, `TSNE2`.

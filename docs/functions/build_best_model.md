# `MultiSampleAnalysis.build_best_model`
**Module:** `workflow.py`

Evaluates multiple candidate models with CV, computes weighted train/test metrics, and fits best model on full data.

## Parameters
- `models` (`dict[str, Any] | None`): Optional model dictionary; defaults include 10 sklearn models.
- `label_col` (`str`): Label column in feature table.
- `feature_cols` (`list[str] | None`): Feature columns used for training. Default: `["duration_s", "blockade_ratio", "segment_std", "segment_skew", "segment_kurt"]`.
- `cv` (`int`): Number of CV folds (default 10).
- `scoring` (`str`): Selection metric: accuracy/f1/recall variants.
- `exclude_noise` (`bool`): Whether to exclude rows marked as noise.

## Returns
- `dict` model package including best model, selected `feature_cols`, scores, `cv_results`, and `all_samples_feature_pred` (all-sample feature/group table with `best_model_pred` as last column).

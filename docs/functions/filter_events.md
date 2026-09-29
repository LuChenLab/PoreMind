# `MultiSampleAnalysis.filter_events`
**Module:** `workflow.py`

Flag noisy events and retain valid events. A hard `blockage_lim` filter is applied to the raw unitless `blockade_ratio` before the selected method.

## Parameters

- `method` (`str`): `"blockade_gmm"` by default; also supports `"peak_detection"`, `"isolation_forest"`, `"lof"`, and `"knn_background"`.
- `parameters` (`dict[str, Any] | None`): Method-specific overrides.
  - `blockade_gmm` / `peak_detection`: `blockade_col`, `dwell_col`, `rm_index`, `n_components=2`, `prior_mean=None`, `visualize=False`.
  - `isolation_forest` / `lof`: `contamination=0.05`, `feature_cols=["duration_s", "blockade_ratio", "segment_skew", "segment_kurt"]`.
  - `knn_background`: `background_sample_ids`, `k=10`, `feature_cols`, `n_noise_match=1`, `background_ratio=1.0`, `metric="euclidean"`, `random_state=42`, and optional `background_blockage_lim`.
- `blockage_lim` (`tuple[float, float]`): Hard `blockade_ratio` range, default `(0.1, 1.0)`.

## Returns

`None`. The method updates `analysis.feature_df` with `is_noise` and `quality_tag`, and stores valid rows in `analysis.filtered_df`.

## Notes

- Filtering is grouped by `sample_id`, falling back to `trace_id`.
- `knn_background` requires background sample identifiers and compares target events with sampled background events in the chosen feature space.
- Keep `blockade_ratio` on its raw ratio scale; do not pass a percentage such as `(10, 80)`.


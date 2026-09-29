# `MultiSampleAnalysis.build_DL_model`
**Module:** `workflow.py`

Train a PyTorch event classifier from `filtered_df`. The method stores the trained package in `DL_model_package` and `DL_model_packages[model_name]`, writes full-data predictions back to `filtered_df`, and records fold metrics in `model_cv_results`.

## Parameters

- `model` (`Any | None`): Optional custom waveform feature extractor. It must be an instantiated PyTorch module with a callable `forward`; a string such as `"one_DCNN"` is invalid.
- `model_name` (`str`): Result key and prediction-column suffix. The default key is `residual-CNN`; the UI commonly uses `1D-CNN`. `MAGJAM` is the depth-8 variant and `MAGJAM_d4` is depth 4. Both MAGJAM variants use waveform inputs only. Other names are allowed as result keys.
- `feature_cols` (`list[str] | None`): Handcrafted features concatenated to a custom/default extractor output. `None` uses `duration_s`, `blockade_ratio`, `segment_std`, `segment_skew`, and `segment_kurt`; an empty list disables them. MAGJAM always disables them.
- `interp_length` (`int`): Target waveform length. Default `500`.
- `interp_method` (`str`): `"interp"` (default, the existing linear resampling path) or `"padding"` (zero-pad short segments and center-truncate long segments).
- `expand` (`int`): Samples added on both sides of each event before scaling and length conversion. Default `50`.
- `scale` (`str | None`): `"blockade"` (default), `"mad"`, `"minmax"`, `"none"`, or `None`.
- `device` (`str`): Preferred device, default `"cuda"`; falls back to CPU when CUDA is unavailable.
- `batch_size` (`int`): Training batch size, default `64`.
- `learning_rate` (`float`): Adam learning rate, default `1e-3`.
- `epoch` (`int`): Maximum epochs per fold, default `30`.
- `early_stop_patience` (`int`): Validation-loss patience, default `5`.
- `cv` (`int`): Stratified cross-validation folds, default `10`.
- `label_col` (`str`): Label column, default `"label"`.

Leave `model=None` when selecting the built-in MAGJAM backbone. Supplying a custom `model` takes the custom-extractor path, even if the result key is named `MAGJAM`.

## Examples

```python
magjam = analysis.build_DL_model(
    model_name="MAGJAM",
    interp_length=500,
    interp_method="interp",
    scale="blockade",
)
magjam_d4 = analysis.build_DL_model(
    model_name="MAGJAM_d4",
    interp_length=500,
    interp_method="padding",
)
```

## Returns

`dict[str, Any]` containing the trained `ModuleDict`, state dict, class mapping, CV results, `feature_cols`, `interp_length`, `interp_method`, `expand`, `scale`, and device information.


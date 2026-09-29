# `MultiSampleAnalysis.classify_new_samples`
**Module:** `workflow.py`

Reuse the fitted preprocessing, detection state, feature settings, and model to classify events from new samples. This is the single public prediction method for both classical ML and DL models.

## Parameters

- `new_sample_paths` (`dict[str, str | Path]`): Mapping from new sample identifiers to input paths.
- `reader` (`str | None`): Optional reader override (`"abf"` or `"csv"`).
- `reader_kwargs` (`dict[str, Any] | None`): Optional reader keyword arguments.
- `custom_feature_fns` (`dict[str, Callable] | None`): Optional custom feature functions forwarded to `extract_features`.
- `model` (`str | Any | None`): Optional trained model name or model object. Classical ML names come from `build_best_model` (for example `"Random Forest"`); DL names are the exact `model_name` used during training, such as `"residual-CNN"`, `"1D-CNN"`, `"MAGJAM"`, or `"MAGJAM_d4"`.

If `model` is omitted, the stored best classical model is preferred; otherwise the stored DL package is used.

## Returns

`tuple[MultiSampleAnalysis, pandas.DataFrame]`:

- The new analysis object, including traces, events, features, and `pl`/`plot` accessors.
- The per-event prediction table. ML columns are `pred_label` and `pred_proba_<class>`; DL columns are `pred_label_<model_name>` and `pred_proba_<class>_<model_name>`.

DL prediction reuses the stored `interp_length`, `interp_method`, `expand`, `scale`, and handcrafted-feature configuration from the trained package.


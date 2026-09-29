# Recipe: Unknown Sample Prediction

Unknown-sample prediction must reuse the training run settings.

## Hard Rules

1. Do not retune unknown samples independently and then compare them directly.
2. Use the training run `02_params.lock.json`.
3. Use the training feature columns and model.
4. If pore state differs strongly from the training data, warn that prediction reliability is low.

For a DL package, reuse the stored `model_name`, `interp_length`, `interp_method`, `expand`, `scale`, and `feature_cols`. The current public method is `analysis.classify_new_samples(...)` for both ML and DL; use `model="MAGJAM"` or `model="MAGJAM_d4"` for the corresponding trained package.

## Recommended Flow

```python
new_analysis, pred = analysis.classify_new_samples(
    new_sample_paths=new_sample_paths,
    reader=reader,
    reader_kwargs=reader_kwargs,
    custom_feature_fns=custom_feature_fns,
    model=selected_model,
)
```

## Outputs

- `s9_model/predictions.csv`
- `s9_model/prediction_probability_distribution.png`
- `s9_model/stacked_prediction_by_sample.png`

## Interpretation Limits

State that predictions are event-level. Sample-level interpretation should combine event proportions, confidence, replicates, and experiment context.

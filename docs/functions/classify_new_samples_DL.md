# `classify_new_samples_DL` compatibility note

The current workflow does **not** implement a separate `MultiSampleAnalysis.classify_new_samples_DL` method. Use the unified method below for both classical and deep-learning models:

```python
new_analysis, pred = analysis.classify_new_samples(
    new_sample_paths={"unknown_01": "unknown_01.abf"},
    reader="abf",
    model="MAGJAM",  # or "MAGJAM_d4", "1D-CNN", or another trained DL key
)
```

The selected DL package supplies its original `interp_length`, `interp_method`, `expand`, `scale`, and feature configuration. See [`classify_new_samples.md`](./classify_new_samples.md).


# Quick Start (Python API)

[简体中文](zh/quickstart.md)

This example outlines a labeled, multi-sample analysis. Replace the file paths and sample/group IDs with the input data. CSV files use `reader="csv"` and require a `current` column; a `time` column is optional when the sampling rate is supplied.

```python
from poremind import create_analysis_object

analysis = create_analysis_object(
    {
        "sample_a": "path/to/sample_a.abf",
        "sample_b": "path/to/sample_b.abf",
    },
    sample_to_group={"sample_a": "group_a", "sample_b": "group_b"},
    reader="abf",
).load()

# Preprocess and inspect the traces.
analysis.denoise()

# Tune the detector on a short time window, then detect events on full traces.
preview_events = analysis.detect_events_simple(
    detect_method="threshold",
    start_ms=0.0,
    end_ms=1000.0,
)
analysis.detect_events(detect_method="threshold")

# Build event-level features and retain events that pass quality filtering.
features = analysis.extract_features()
analysis.filter_events(
    method="blockade_gmm",
    parameters={"n_components": 2, "prior_mean": None},
    blockage_lim=(0.1, 1.0),
)

# Explore features and train a classical classifier.
analysis.do_pca(
    feature_cols=["duration_s", "blockade_ratio"],
    data="filtered",
)
best_pkg = analysis.build_best_model(cv=5, scoring="accuracy")

# Predict events in samples that were not used for training.
new_analysis, predictions = analysis.classify_new_samples(
    {"unknown_01": "path/to/unknown_01.abf"},
    reader="abf",
    model=best_pkg["best_model"],
)
```

`sample_to_group` assigns each sample's group to the `label` column created during feature extraction. For supervised learning, provide the group labels before training. The [notebook walkthrough](_generated/quickstart.ipynb) shows the current analysis example with code and saved outputs.

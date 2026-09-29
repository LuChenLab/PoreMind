# Analysis workflow

The API and local UI use the same event-centered workflow. The detector-tuning step is a preview on a selected local interval; full event detection runs separately over the loaded traces.

| Stage | Purpose | Main outputs |
| --- | --- | --- |
| Import | Read ABF or CSV current traces and assign sample groups | Traces and sample metadata |
| Preprocess | Denoise signals and inspect raw or processed traces | Processed traces |
| Pre-Events | Tune event-detection settings on a short time window | Preview events and detector settings |
| Events | Detect events across full traces | Events and baseline information |
| Features & Filter | Calculate event features and remove low-quality or outlier events | Feature table and filtered events |
| Reduction | Explore numeric event features with PCA, t-SNE, or UMAP | Coordinates appended to the event table |
| Train Model | Fit classical ML or waveform DL models | Model package and evaluation results |
| Predict | Classify events in new samples with a trained model | Per-event labels and scores |
| Export | Save selected tables, parameters, and analysis outputs | Files in the chosen export directory |

## Event data and labels

Each trace is associated with a sample ID. `sample_to_group` maps sample IDs to experimental groups; feature extraction uses these group values as event labels. Without that mapping, add a `label` column before supervised training.

The default classical-model features are `duration_s`, `blockade_ratio`, `segment_std`, `segment_skew`, and `segment_kurt`. `blockade_ratio` is a unitless ratio. The common display and filtering range is 0–1, but out-of-range values should be checked rather than silently rescaled.

## Detection and filtering

The Python API supports `threshold`, `zscore_threshold`, `cusum`, `pelt`, and `hmm` event detection, along with rolling or global baseline options. Set detector parameters explicitly when reproducing a run; UI and API defaults can differ. See [event detection reference](functions/detect_events.md) and [preview detection reference](functions/detect_events_simple.md).

Filtering first applies `blockage_lim`, then the selected quality or anomaly method, such as `blockade_gmm`, `peak_detection`, `isolation_forest`, `lof`, or `knn_background`.

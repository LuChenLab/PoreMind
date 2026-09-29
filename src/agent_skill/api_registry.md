# PoreMind API Registry

This registry records the APIs and parameters implemented by the current source tree. Prefer these APIs over custom reimplementation. Checkpoint and confirmation rules remain defined in SKILL.md and workflow_state_machine.md.

## Imports

~~~python
from poremind import (
    AnalysisConfig, LabeledDataset, MultiSampleAnalysis,
    analyze_abf_to_event_df, create_analysis_object,
    predict_events, train_event_classifier,
)
~~~

MAGJAM symbols are lazy exports because importing them requires PyTorch:

~~~python
from poremind import MAGJAM_MODELS, MSSJambaHybrid, MAGJAMExtractor, MAGJAMFullModel
~~~

PoreMindWaveformEncoder is not exported by the current package. If it is unavailable, do not fabricate embeddings; continue with handcrafted features or PoreMind DL.

## Create Analysis Object

~~~python
create_analysis_object(
    sample_paths: dict[str, str | Path],
    sample_to_group: dict[str, str] | None = None,
    reader: str = "abf",
    reader_kwargs: dict[str, Any] | None = None,
) -> MultiSampleAnalysis
~~~

Readers are "abf" and "csv". ABF load() expands all channel/sweep pairs. CSV uses current_col="current", optional time_col="time", and requires sampling_rate_hz when no time column is present.

## Load, Preprocess, and Preview

~~~python
analysis.load() -> MultiSampleAnalysis
analysis.denoise(method="butterworth_filtfilt", **kwargs) -> MultiSampleAnalysis
analysis.preview_signal(sample_id, start_s=0.0, end_s=None, max_points=5000) -> pd.DataFrame
analysis.visualize_signal(sample_id, start_s=0.0, end_s=None, max_points=5000)
~~~

Supported preprocessing methods are none, moving_average, median, drift_corrected_moving_average, and butterworth_filtfilt. The denoise default is butterworth_filtfilt(filtfilt_N=2, filtfilt_Wn=0.1).

Low-level helpers are moving_average(x, window=5), median_filter(x, window=5), remove_slow_drift(x, window=1001), butterworth_filtfilt(x, filtfilt_N=2, filtfilt_Wn=0.1), and preprocess_signal(x, method="butterworth_filtfilt", **kwargs).

analysis.pl.current uses milliseconds for start_ms and end_ms and defaults to sample_id=None, current="denoise", start_ms=0.0, end_ms=1.0, width=10.0, height=3.0.

## Baseline Estimation

The workflow supports:

~~~text
rolling_quantile: window=10000, q=0.5 when called through detect_events
global_quantile: q=0.5 unless explicitly supplied
global_median: no extra parameter
~~~

The Python workflow default is baseline_method="rolling_quantile". The Agent Skill policy intentionally recommends global_quantile; when following the policy, pass it explicitly and select q from direction and baseline position.

The low-level estimate_baseline(x, method="rolling_quantile", **kwargs) supports rolling_quantile and global_median; its standalone rolling-window default is window=501, q=0.5.

## Event Detection

Local preview:

~~~python
analysis.detect_events_simple(
    detect_method="threshold", detect_params=None,
    baseline_method="rolling_quantile", baseline_params=None,
    sample_id=None, current="denoise",
    start_ms=0.0, end_ms=1000.0,
    detect_direction="down",
    merge_event=False, merge_event_params=None,
    exclude_current=True, exclude_current_params=None,
) -> dict[str, list[Event]]
~~~

Full detection:

~~~python
analysis.detect_events(
    detect_method="threshold", detect_params=None,
    baseline_method="rolling_quantile", baseline_params=None,
    detect_direction="down",
    merge_event=False, merge_event_params=None,
    exclude_current=True, exclude_current_params=None,
) -> MultiSampleAnalysis
~~~

Detection methods are threshold, zscore_threshold, cusum, pelt, and hmm. Default detector parameters are:

~~~python
threshold = {"sigma_k": 5.0, "min_duration_s": 0.0, "noise_method": "mad"}
zscore_threshold = {"z": 4.0, "min_duration_s": 0.0, "noise_method": "mad"}
cusum = {"drift": 0.02, "threshold": 8.0, "min_duration_s": 0.0, "noise_method": "mad"}
pelt = {"model": "l2", "penalty": 8.0, "sigma_k": 3.0, "min_duration_s": 0.0, "noise_method": "mad"}
hmm = {"n_components": 2, "covariance_type": "diag", "n_iter": 200, "min_duration_s": 0.0}
~~~

detect_direction defaults to down, merge_event to False, and exclude_current to True. With no bounds, up uses values below zero and down uses values above zero. exclude_current_params controls baseline/noise statistics, not the later blockage_lim event filter.

## Event Plots

~~~python
analysis.pl.event_current_simple(...)
analysis.pl.event_current(...)
analysis.pl.event_current_label(...)
~~~

The current source keeps the historical spellings lable_col and lable_color in event_current_label.

## Feature Extraction

~~~python
analysis.extract_features(custom_feature_fns=None, max_event_per_sample=None) -> pd.DataFrame
~~~

Built-in columns include trace_id, sample_id, channel, sweep, event_id, start_idx, end_idx, start_time_s, end_time_s, duration_s, delta_i, snr, left_baseline, right_baseline, global_baseline, blockade_ratio, segment_mean, segment_std, segment_min, segment_max, segment_skew, segment_kurt, and peak_factor. Custom functions receive an event segment and may return a dictionary of values.

## Filtering

~~~python
analysis.filter_events(
    method="blockade_gmm",
    parameters=None,
    blockage_lim=(0.1, 1.0),
) -> None
~~~

Supported methods are blockade_gmm, peak_detection, isolation_forest, lof, and knn_background. The hard blockage_lim is applied first on raw blockade_ratio. The method updates feature_df with is_noise and quality_tag, and writes valid rows to filtered_df.

For knn_background, relevant parameters are background_sample_ids, k=10, feature_cols, n_noise_match=1, background_ratio=1.0, metric="euclidean", random_state=42, and optional background_blockage_lim.

## Dimensionality Reduction

~~~python
analysis.do_pca(feature_cols=None, data="filtered", random_state=42)
analysis.do_tsne(feature_cols=None, data="filtered", random_state=42, perplexity=30.0, n_iter=1000)
analysis.do_umap(feature_cols=None, data="filtered", random_state=42, n_neighbors=15, min_dist=0.1)
~~~

do_umap requires umap-learn. The skill policy uses PCA as the default representation and treats t-SNE/UMAP as optional follow-ups.

## Visualization

~~~python
analysis.pl.plot_2d(...)
analysis.pl.plot_3d(...)
analysis.pl.stacked_bar(...)
analysis.pl.box_significance(...)
analysis.pl.model_cm(...)
analysis.pl.model_metric_bar(...)
~~~

analysis.plot is an alias of analysis.pl.

## Machine Learning

~~~python
analysis.build_best_model(
    models=None, label_col="label", feature_cols=None,
    cv=10, scoring="accuracy", exclude_noise=True,
) -> dict
~~~

Default features are duration_s, blockade_ratio, segment_std, segment_skew, and segment_kurt. Built-in candidates are Random Forest, Logistic Regression, SVM, MLP, Elastic Net, Lasso, Decision Tree, LDA, AdaBoost, and Gaussian Naive Bayes.

## Deep Learning

~~~python
analysis.build_DL_model(
    model=None, model_name="residual-CNN", feature_cols=None,
    interp_length=500, interp_method="interp", expand=50,
    scale="blockade", device="cuda", batch_size=64,
    learning_rate=1e-3, epoch=30, early_stop_patience=5,
    cv=10, label_col="label",
) -> dict
~~~

interp_method="interp" is the default existing resampling path. interp_method="padding" zero-pads short event segments and center-truncates long segments. Built-in model names include MAGJAM (d8) and MAGJAM_d4 (d4); both consume waveform inputs only and ignore handcrafted feature_cols.

For a custom model, model must be an instantiated PyTorch module with forward, for example model=one_DCNN(), not model="one_DCNN".
Leave model=None when selecting the built-in MAGJAM backbone; supplying a custom model takes the custom-extractor path.

~~~python
analysis.explain_dl(model_name="MAGJAM", baseline="zero", n_steps=50)
analysis.pl.attribute(sample_id="A1", event_index=1, model_name="MAGJAM")
analysis.pl.plot_fold_loss(model_name="MAGJAM", type="train")
~~~

DL explainability requires PyTorch and Captum.

## New Sample Prediction

~~~python
analysis.classify_new_samples(
    new_sample_paths, reader=None, reader_kwargs=None,
    custom_feature_fns=None, model="MAGJAM",
)
~~~

This is the unified ML/DL prediction method. It reuses training preprocessing, detection, and DL settings, including interp_length, interp_method, expand, scale, and feature_cols.

## Low-Level Model Package APIs

~~~python
train_event_classifier(dataset, model_name="random_forest", model_params=None)
predict_events(package, df)
save_model_package(package, path)
load_model_package(path)
~~~

Low-level train_event_classifier supports random_forest and optional xgboost; it selects numeric feature columns automatically with select_feature_columns.

# PoreMind Single-Molecule Nanopore Analysis Framework

This document describes the implementation that is currently present in this repository. It is an offline, multi-sample workflow for ABF and CSV current traces; it does not claim a real-time acquisition service, HDF5 reader, REST API, or a separate report-generation package.

## 1. Implemented workflow

```text
ABF/CSV files
    -> create_analysis_object(...).load()
    -> denoise and preview
    -> detect_events_simple (local tuning)
    -> detect_events (full traces)
    -> extract_features
    -> filter_events
    -> PCA / t-SNE / UMAP / plots
    -> build_best_model or build_DL_model
    -> classify_new_samples
```

The step-by-step notebook follows this order. The local preview is intentionally separate from full event detection: `detect_events_simple` stores its result in `detect_events_simple_object` and `simple_events`, while `detect_events` populates `events` and `baselines`.

## 2. Repository structure

```text
PoreMind/
  pyproject.toml
  README.md
  README.zh.md
  src/
    poremind/
      io.py          # ABF/CSV readers and Trace
      preprocess.py  # signal preprocessing helpers
      baseline.py    # rolling quantile and global median helpers
      events.py      # low-level threshold/CUSUM/PELT/HMM detectors
      features.py    # Event table and numeric feature selection
      workflow.py    # MultiSampleAnalysis orchestration and ML/DL
      magjam.py      # standalone MAGJAM components and registry
      ml.py          # low-level ML package APIs
      pl.py          # plotting and DL attribution accessor
      pipeline.py    # compact ABF-to-event-dataframe pipeline
    ui/
      app.py         # Gradio Blocks application
      controller.py  # UI-to-workflow adapter
      session.py     # UI state and parameter storage
    agent_skill/     # checkpointed agent instructions and API registry
  docs/functions/    # per-function reference
  tests/             # workflow and integration tests
```

## 3. Data model

### Trace

`poremind.io.Trace` stores:

- `current`: one-dimensional current array;
- `time`: time array in seconds;
- `sampling_rate_hz`: sampling rate;
- `source`: source path;
- optional `channel` and `sweep` identifiers.

`read_abf` reads one channel/sweep pair. `read_abf_all` reads every channel and sweep. `read_csv` reads a current column and either derives the sampling rate from a time column or requires `sampling_rate_hz`.

### Event

`poremind.events.Event` stores `start_idx`, `end_idx`, `baseline_local`, `delta_i`, `dwell_time_s`, and `snr`. The workflow maps local preview indices back to full-trace indices.

### Feature table

`extract_features` produces event-level rows containing identifiers, time/index bounds, duration, direction-aware `delta_i`, `snr`, baseline values, `blockade_ratio`, segment statistics, and optional custom columns. The standard model feature set is:

```text
duration_s, blockade_ratio, segment_std, segment_skew, segment_kurt
```

For downward detection, `delta_i = global_baseline - segment_mean`. For upward detection, the workflow uses the corresponding sign-normalized expression so that `blockade_ratio` remains direction-consistent.

## 4. Signal preprocessing

`MultiSampleAnalysis.denoise` applies one method to every loaded trace and caches the result in `analysis.denoised`. Supported methods are `none`, `moving_average`, `median`, `drift_corrected_moving_average`, and `butterworth_filtfilt`. The workflow default is zero-phase `butterworth_filtfilt` with `filtfilt_N=2` and `filtfilt_Wn=0.1`.

## 5. Baseline and event detection

The workflow accepts `rolling_quantile`, `global_quantile`, and `global_median`. The Python method defaults are `baseline_method="rolling_quantile"` and `baseline_params={"window": 10000, "q": 0.5}`. `global_quantile` uses the quantile of points retained by `exclude_current_params`; `global_median` uses their median.

Detection methods are `threshold`, `zscore_threshold`, `cusum`, `pelt`, and `hmm`. The method default is `detect_direction="down"`, `exclude_current=True`, and `merge_event=False`. When no bounds are provided, the statistical current range is direction-dependent: `up` uses values below zero and `down` uses values above zero. This statistical mask is not the same as the later event-quality filter.

`detect_events_simple` uses a default 0–1000 ms window and is intended for local parameter tuning. `detect_events` processes the complete denoised traces. Both methods record their parameters in workflow state and raise an error when too few effective points remain for baseline/noise statistics.

## 6. Event filtering and visualization

`filter_events` first applies `blockage_lim=(0.1, 1.0)` to the raw unitless `blockade_ratio`, then applies one of:

- `blockade_gmm`;
- `peak_detection`;
- `isolation_forest`;
- `lof`;
- `knn_background`.

The method updates `feature_df` with `is_noise` and `quality_tag`, and stores rows tagged `valid` in `filtered_df`. Plotting is available through `analysis.pl` and the compatibility alias `analysis.plot`, including current/event views, 2D/3D feature plots, stacked proportions, boxplots, model confusion matrices, metric bars, fold loss, and DL attribution.

## 7. Modeling

### Classical ML

`build_best_model` evaluates a default set of scikit-learn models using event-level cross-validation and fits the selected estimator on all selected rows. The default features are the five columns listed above. The package stores the best estimator, candidate models, scores, CV results, feature columns, and all-sample predictions.

### Deep learning

`build_DL_model` accepts a waveform extractor and creates the common PoreMind `ModuleDict({"extractor": ..., "head": ...})` layout. Its input defaults are `interp_length=500` and `interp_method="interp"`.

`interp` uses the existing linear resampling path. `padding` zero-pads short segments and center-truncates long segments. `scale` accepts `mad`, `minmax`, `blockade`, or `none`.

The built-in MAGJAM choices are:

| `model_name` | Backbone | Input | Handcrafted features |
| --- | --- | --- | --- |
| `MAGJAM` | MSS + Jamba-style hybrid, d8 | `[B, 1, T]` | disabled |
| `MAGJAM_d4` | same hybrid, d4 | `[B, 1, T]` | disabled |

MAGJAM is defined in the standalone `src/poremind/magjam.py` module and is registered through `MAGJAM_MODELS`. `explain_dl` and `pl.attribute` reuse the same stored input preprocessing and are therefore compatible with both variants when PyTorch and Captum are installed.

For a custom DL model, pass an instantiated `torch.nn.Module` with a `forward` method. Passing a string such as `"one_DCNN"` is not a valid custom model object. Leave `model=None` when selecting the built-in MAGJAM backbone; a supplied model takes the custom-extractor path.

## 8. New-sample prediction

`classify_new_samples` creates a new analysis object, reuses the training preprocessing and detection state, extracts features, and dispatches to either a stored ML model or a stored DL package. ML output columns are `pred_label` and `pred_proba_<class>`. DL output columns are suffixed with the trained model name, for example `pred_label_MAGJAM` and `pred_proba_A_MAGJAM`.

## 9. Current scope and limitations

- Input readers currently implemented: ABF and CSV.
- The package is an offline analysis workflow; near-real-time acquisition and REST serving are outside the current code.
- PoreMindWaveformEncoder is not exported by the current package. The agent-skill policy keeps it as an optional/future-compatible extension and must not fabricate embeddings when the interface is unavailable.
- Event-level CV can overestimate generalization when events from the same sample are correlated; sample-level splitting should be used when the study design permits it.


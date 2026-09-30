from __future__ import annotations

from contextlib import contextmanager
import ctypes
from html import escape
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import threading
from urllib.parse import quote
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from .controller import AnalysisController
from poremind.features import select_feature_columns


# Mirrors MultiSampleAnalysis.build_best_model's PoreMind defaults.
_MODEL_DEFAULT_FEATURE_COLS = [
    "duration_s",
    "blockade_ratio",
    "segment_std",
    "segment_skew",
    "segment_kurt",
]


_PARAM_HELP = {
    "file_input": {
        "en": "Select one or more raw recordings to load. ABF files are Axon Binary Files; each CSV must provide time and current columns.",
    },
    "sample_annotations": {
        "en": "Enter a group or class name for each source file before loading. This becomes the `label` used by supervised model training and label-based plots; blank annotations default to the sample ID. Reload samples after editing.",
    },
    "reader": {
        "en": "Select how raw traces are read: use `abf` for Axon Binary Files or `csv` for tables with time and current columns. The choice must match the uploaded file format.",
    },
    "denoise_method": {
        "en": "Choose preprocessing applied before event detection: `butterworth_filtfilt` applies a zero-phase Butterworth low-pass filter; `moving_average` and `median` smooth over sample windows; `drift_corrected_moving_average` first removes a slow baseline trend and then smooths; `none` leaves the signal unchanged. Check that short events remain visible.",
    },
    "detect_method": {
        "en": "Choose how event samples are identified relative to the estimated baseline: `threshold` applies a noise-scaled residual cutoff; `zscore_threshold` applies a cutoff to standardized residuals; CUSUM accumulates sustained deviations; PELT finds change points and then checks segment residuals; HMM assigns residuals to Gaussian states and treats the lowest-mean state as the event state. The current HMM implementation does not apply `detect_direction`; see the direction help. Start with `threshold` for a simple baseline and compare methods on the same trace.",
    },
    "detect_direction": {
        "en": "Select event polarity relative to the estimated baseline. For `threshold`, `zscore_threshold`, CUSUM, and PELT, `down` detects downward deflections (blockades) and `up` detects upward deflections. This choice also suggests a baseline quantile (5% for up, 95% for down), which you may edit. The current HMM implementation always treats the lowest-mean residual state as the event state and does not switch states for `up`; use another detector for upward HMM-style events.",
    },
    "baseline_method": {
        "en": "Choose the reference-current estimator. `rolling_quantile` estimates a local baseline that can follow slow drift; `global_quantile` uses one quantile value for the whole trace; `global_median` uses one robust median value. Rolling estimation can take longer on long traces.",
    },
    "exclude_current": {
        "en": "When enabled, `current-range min` and `current-range max` restrict which points contribute to baseline and noise estimates: values must be strictly greater than the minimum and strictly less than the maximum. A blank bound leaves that side unrestricted; if both are blank, all points contribute. Points excluded from these estimates are still checked for events. This option does not filter detected events; the core API calls it `exclude_current`.",
    },
    "filter_method": {
        "en": "Choose how events are classified as valid or noise. `blockade_gmm` fits a Gaussian mixture to each trace's `blockade_ratio` values and keeps one selected component; `isolation_forest` and `lof` flag feature-space outliers; `knn_background` flags target events whose nearest neighbors resemble events from selected background traces. The inclusive `blockage_lim` interval first marks out-of-range rows as noise; method-specific filtering then evaluates in-range target events.",
    },
    "dr_method": {
        "en": "Choose a 2D embedding of the selected numeric event features. PCA is a fast linear summary; t-SNE and UMAP can reveal nonlinear neighborhoods but may take longer and emphasize local structure. The embedding is for visualization, not a replacement for the original feature values.",
    },
    "model_name": {
        "en": "Select the deep-learning architecture. `1D-CNN` is a compact convolutional baseline; `MAGJAM` is the depth-8 model; `MAGJAM_d4` is its shallower depth-4 variant. These models train on event waveforms rather than the classic feature-column selection.",
    },
    "interp_method": {
        "en": "Normalize event waveforms to the model's fixed input length (500 samples in this UI). `interp` resamples the full waveform; `padding` appends zeros to shorter events and center-truncates longer events. Interpolation preserves the full event shape, while padding preserves the original sample spacing for short events.",
    },
    "device": {
        "en": "Choose where deep-learning calculations run. `cpu` works without a GPU; `cuda` uses an available NVIDIA GPU and requires a CUDA-enabled PyTorch installation. If CUDA is unavailable, training will fail rather than silently switch devices.",
    },
    "output_dir": {
        "en": "Folder where Export writes analysis tables, a JSON parameter snapshot, and a Python reproduction script; a trained classical model bundle is included when available. The default is `~/ui_exports`. A missing folder is created automatically; make sure its parent location is writable.",
    },
}


# Help text for every visible control and result label. Repeated labels share a
# definition; detector/model-specific labels use `_CONTEXT_LABEL_HELP` below.
_LABEL_HELP = {
    "Sample annotations": "Enter a group or class name for each source file before loading. This becomes the `label` used by supervised training and label-based plots; blank annotations default to the sample ID. Reload samples after editing.",
    "Load summary": "Import status and counts for the source files and traces loaded. Check this for read errors and confirm the expected trace count before continuing.",
    "Sample information": "One row per loaded trace, showing its source file, sample/trace ID, ABF channel and sweep when applicable, point count, duration, and sampling rate.",
    "filtfilt_N": "Order of the zero-phase Butterworth filter. Higher orders create a sharper frequency cutoff but can increase edge artifacts; start low and inspect that event shapes remain intact.",
    "filtfilt_Wn": "Butterworth cutoff normalized to the Nyquist frequency (0–1). Lower values remove more high-frequency variation; higher values retain more rapid signal changes. Tune while checking short-event shape and noise.",
    "window": "Number of samples in the moving-average or median smoothing window. A larger window smooths more noise but can broaden or erase short events; choose a window shorter than the events you need to preserve.",
    "drift_window": "Number of samples used to estimate the slow moving baseline that is subtracted before smoothing. Increase it to target slower drift; too short a window may treat real events as drift.",
    "smooth_window": "Number of samples used to smooth the drift-corrected trace. Larger values reduce more high-frequency noise but can blur brief events.",
    "Preprocessing status": "Summary of the denoising method and settings applied to the loaded traces. It records the operation; use the current-trace plot to inspect the resulting waveform.",
    "sample_id": "Select a loaded sample/trace for plots, event previews, or trace-specific tables. ABF channels/sweeps may each have separate trace IDs; choices are populated after the corresponding recordings are loaded.",
    "current": "Signal to display or analyze: `denoise` uses the preprocessed trace when available; `raw` uses the original current recording. Compare them to check whether preprocessing changed event shapes.",
    "start_ms": "Start time in milliseconds. In trace plots it sets the left edge of the displayed interval; in Pre-Events it marks the beginning of the segment used for the quick detection check.",
    "end_ms": "End time in milliseconds. In trace plots it sets the right edge of the displayed interval; in Pre-Events it marks the end of the segment used for the quick detection check. It must be greater than `start_ms`.",
    "pl.current": "Current-versus-time plot for the selected trace and time interval. Use it to compare the raw and denoised signals and confirm that filtering preserves expected events.",
    "Download plot files (PNG + editable-text PDF)": "Download a ZIP containing the displayed plot as a high-resolution PNG and a PDF whose text remains editable in vector-capable software.",
    "analysis.preview_signal(...).head()": "Preview table of the first rows from the selected trace's initial 2 ms. It is a quick check of time/current values (using the denoised signal when available), not the full recording.",
    "baseline window": "Number of samples used in each local window when `baseline_method` is `rolling_quantile`. Larger windows follow slow drift more gradually; smaller windows adapt faster but may absorb event structure. This control is ignored by `global_quantile` and `global_median`.",
    "baseline q (%)": "Quantile used to estimate the baseline, entered as a percentage and converted internally to 0–1. Lower quantiles suit upward events; higher quantiles suit downward blockades. Direction selection suggests 5% for up and 95% for down, but you can change it. This setting is ignored by `global_median`.",
    "merge_event": "After detection, combine consecutive events when the gap between them is no longer than `merge_gap_ms`. This can join fragmented detections but may combine genuinely separate nearby events.",
    "merge_gap_ms": "Largest gap, in milliseconds, allowed when `merge_event` joins neighboring detections. Increase it to merge more fragments; decrease it to keep nearby events separate.",
    "sigma_k": "Noise-multiplier cutoff for threshold detection (and the residual check after PELT segmentation). A candidate must deviate from baseline by roughly this many estimated noise scales; increasing it is more conservative and usually yields fewer events.",
    "min_duration_s": "Minimum event duration in seconds. Detections shorter than this are discarded; use 0 to keep events down to the signal's sample resolution.",
    "noise_method": "How residual noise is estimated for the selected detector (where this option is shown). `mad` uses a robust, scale-adjusted median absolute deviation and is less affected by large events/outliers; `std` uses standard deviation and is more affected by them. The estimate scales or standardizes residuals in threshold, z-score, CUSUM, and PELT checks; HMM does not use this setting.",
    "z": "Numeric event feature plotted on the third axis of a 3D scatter plot. It is not the z-score detector threshold; that detector field has a context-specific explanation.",
    "drift": "Small positive offset added to each standardized residual before the one-sided CUSUM accumulation (dimensionless noise-scale units). A larger value counteracts small deviations, requiring stronger or more sustained downward evidence and usually reducing detections; it is not a baseline window.",
    "threshold": "Positive cutoff for the magnitude of the accumulated, standardized CUSUM deviation. An event is reported when sustained evidence crosses this cutoff; a larger value requires more evidence and usually produces fewer events.",
    "model": "PELT segment cost model: `l1` uses absolute deviations, `l2` uses squared deviations, and `rbf` uses a nonlinear kernel cost. Choose the cost that best separates signal regimes; `l2` is a common starting point.",
    "penalty": "PELT change-point penalty. A larger penalty discourages extra segments and usually returns fewer, broader changes; a smaller penalty finds more change points but may over-segment noise.",
    "n_components": "Number of components or hidden states used by the active mixture/HMM model. More components can represent additional blockade populations or signal regimes, but may overfit limited data; see the method-specific tooltip where shown.",
    "covariance_type": "Covariance form for the Gaussian emission distributions in the HMM: `diag`, `full`, `tied`, or `spherical`. It controls how state variability is represented; because the detector fits a one-dimensional residual, some choices may behave similarly.",
    "n_iter": "Maximum optimization iterations for the HMM fit or t-SNE embedding. Raising the cap can give the optimizer more time to converge but increases runtime; it may stop earlier if its convergence criterion is met.",
    "Use a current range for baseline/noise statistics": "When enabled, `current-range min` and `current-range max` restrict points used to estimate baseline and noise: included values must be strictly greater than the minimum and strictly less than the maximum. A blank bound leaves that side unrestricted; with both blank, all points are used. Excluded points are still checked for events—this is not an event filter. The core API calls this option `exclude_current`.",
    "current-range min (blank = None)": "Optional strict lower current bound for baseline/noise estimation, in the same units as the loaded trace (often pA). Only samples greater than this value are included; a sample exactly at the bound is excluded. Leave blank for no lower bound.",
    "current-range max (blank = None)": "Optional strict upper current bound for baseline/noise estimation, in the same units as the loaded trace (often pA). Only samples less than this value are included; a sample exactly at the bound is excluded. Leave blank for no upper bound.",
    "Simple detection result": "Counts of candidate events found in the selected trace segment by the Pre-Events check. Use this result to tune settings before running full detection across all traces.",
    "start_event": "One-based index of the first detected event to include in the event-current plot/table. The displayed interval starts at this event's boundary.",
    "end_event": "One-based index of the last detected event to include in the event-current plot/table. It must be at least `start_event`; the displayed interval ends at this event's boundary.",
    "pl.event_current_simple": "Pre-Events waveform plot for the selected trace, with the selected candidate event boundaries marked. Use it to check whether the preview settings capture the intended events.",
    "analysis.simple_events": "Event table produced by the Pre-Events check for the selected trace. Inspect event timing, duration, current change, and signal-to-noise values before full detection.",
    "Full detection result": "Counts of events found across all loaded traces using the current Events settings. Compare counts with the Pre-Events check and inspect representative traces before feature extraction.",
    "pl.event_current": "Waveform plot of the globally detected events for the selected trace, with event boundaries marked. Use it to verify detections before extracting features.",
    "analysis.events[sample_id]": "Event table for the selected trace after full detection. Review detected intervals and event measurements before proceeding to Features & Filter.",
    "max_event_per_sample": "Maximum number of detected events to process for each trace during feature extraction. Events are considered in trace order, so if a trace exceeds the cap, later events are omitted. Lower limits reduce processing time and table size; the value must be greater than zero.",
    "Enable custom_shape_features": "Add two extra waveform-shape measurements during feature extraction: peak-to-peak amplitude (`ptp`) and mean absolute current (`abs_mean`). These become additional numeric features.",
    "feature_df": "Extracted event-feature table before the current filtering step. Rows represent detected events; inspect it to confirm feature values and labels before filtering or modeling.",
    "prior_mean (blank = None)": "Optional expected, unitless `blockade_ratio` used to choose which Gaussian-mixture component to keep for each trace: among candidate component means, the closest to this value is retained as valid and the others are marked as noise. If blank, the filter chooses the most compact candidate component (lowest fitted variance).",
    "contamination": "Approximate fraction of events in each trace that Isolation Forest or LOF should flag as outliers (for example, `0.05` means about 5%). A larger value generally removes more events; choose it based on the acceptable false-removal rate and inspect the `is_noise` results.",
    "background_sample_ids": "Select one or more traces known to represent background/noise for KNN matching. Their events supply reference points and are not processed as target traces. The optional `background_blockage_lim` restricts which reference events are sampled; the general `blockage_lim` marks out-of-range background rows as noise in the output but does not itself remove them from the reference pool.",
    "k": "Number of other events searched for each target event in the combined target-and-background feature space. Only the `k` nearest neighbors are counted; larger `k` uses a broader neighborhood and more distance comparisons. If fewer than `k` other events are available, the search uses the available pool.",
    "n_noise_match": "Minimum number of background-reference events among the `k` nearest neighbors needed to flag a target event as noise. It must be between 1 and `k`; values greater than `k` are capped at `k`. A higher value requires stronger background similarity before removal.",
    "background_ratio": "Number of background reference events sampled per target trace, relative to that trace's event count (`1.0` requests an equal-sized pool). Values above 1 request more references; values below 1 request fewer. `0` or an empty eligible background pool disables KNN rejection and keeps target events that passed `blockage_lim`; an undersized pool is sampled with replacement.",
    "feature_cols (comma-separated; blank = default)": "Numeric event columns used to measure similarity between target and background events in KNN. Enter comma-separated column names; leave blank to use `duration_s`, `blockade_ratio`, `segment_skew`, and `segment_kurt`. Every selected column must exist in both groups; changing the columns changes the distance calculation and which events look background-like.",
    "background_blockage_lim (lo,hi; blank = disabled)": "Optional inclusive lower and upper bounds on the unitless `blockade_ratio` of background-reference events, entered as `lo,hi`. This restriction is applied to the reference pool before sampling; leave blank to use all selected background events.",
    "blockage_lim low": "Lower bound of the accepted, unitless `blockade_ratio` interval used before the selected filter. Values below this bound are marked as noise; values equal to the bound are retained for the next filtering step.",
    "blockage_lim high": "Upper bound of the accepted, unitless `blockade_ratio` interval used before the selected filter. Values above this bound are marked as noise; values equal to the bound are retained for the next filtering step.",
    "filtered_df": "Event-feature table after the blockade limits and selected noise/outlier filter have been applied. Check its `is_noise`/quality fields and retained rows before reduction or training.",
    "Data source": "Choose which event table the operation uses: `feature` includes all extracted events, while `filtered` uses the table after noise/outlier filtering. The selected source affects the plotted or embedded points.",
    "Visualization method": "Choose how to display the selected event data: 2D/3D scatter shows feature relationships; box significance compares a numeric feature across groups.",
    "x": "Numeric event feature plotted on the horizontal axis of a 2D/3D scatter plot. Choose a feature whose units and range make the pattern interpretable.",
    "y": "Numeric event feature plotted on the vertical axis of a 2D/3D scatter plot. In some feature/prediction plots, non-`blockade_ratio` y values are displayed on a log2 scale.",
    "x min (blank: blockade_ratio = 0; else auto)": "Lower x-axis bound. Leave blank to use 0 when the x feature is `blockade_ratio`, or automatic scaling for other features; enter a value to override the default range.",
    "x max (blank: blockade_ratio = 1; else auto)": "Upper x-axis bound. Leave blank to use 1 when the x feature is `blockade_ratio`, or automatic scaling for other features; enter a value to override the default range.",
    "y min (blank: blockade_ratio = 0; else auto)": "Lower y-axis bound. Leave blank to use 0 when the y/value feature is `blockade_ratio`, or automatic scaling for other features; enter a value to override the default range.",
    "y max (blank: blockade_ratio = 1; else auto)": "Upper y-axis bound. Leave blank to use 1 when the y/value feature is `blockade_ratio`, or automatic scaling for other features; enter a value to override the default range.",
    "Color column (value)": "Column whose values determine point colors in the feature scatter plot. Categorical values get separate colors; numeric values can be shown as a color scale.",
    "group_col": "Column that defines the categories compared in the box-significance plot. Choose a label/group column with at least two groups.",
    "value_col": "Numeric feature summarized on the box plot's y-axis and used for group significance tests. Choose a measurement such as blockade ratio or event duration.",
    "Feature/filter visualization": "Resulting plot for the chosen feature table, axes, color/group columns, and visualization method. If a plot is blank, check that data has been loaded and the selected columns contain values.",
    "random_state": "Seed for randomized reduction methods, including t-SNE and UMAP. Keeping it fixed makes repeated runs more comparable; it does not change the input data.",
    "perplexity": "t-SNE estimate of how many nearby points each observation should relate to. Higher values emphasize broader structure and can change the embedding substantially; PoreMind automatically caps it at the number of rows minus one when needed.",
    "n_neighbors": "Number of nearby observations UMAP uses to build its local graph. Smaller values emphasize local clusters; larger values preserve broader structure.",
    "min_dist": "Minimum spacing UMAP allows between embedded points. Smaller values make clusters more compact; larger values spread points more evenly. It changes visualization geometry, not the original features.",
    "Feature columns (same defaults as Train Model)": "Numeric event columns used to calculate the 2D embedding. The UI initially selects the same available core features as Train Model (`duration_s`, `blockade_ratio`, `segment_std`, `segment_skew`, `segment_kurt`); an empty selection falls back to those defaults. Edit the selection to change which measurements shape the embedding.",
    "Reduction result": "Event table augmented with the selected method's two embedding coordinates (PC1/PC2, TSNE1/TSNE2, or UMAP1/UMAP2). The coordinates are visualization values, not physical measurements.",
    "pl.plot_2d": "Two-dimensional plot of the reduction coordinates selected below, colored by the chosen column. The embedded points can rotate/reflect between runs; interpret relative neighborhoods rather than treating axis values as physical units.",
    "Color column": "Column used to color points in the dimensionality-reduction plot, commonly the known class/group label. It affects only display, not the embedding calculation.",
    "Reduction x": "Numeric column mapped to the horizontal axis of the 2D embedding plot, usually the first coordinate (for example PC1 or UMAP1).",
    "Reduction y": "Numeric column mapped to the vertical axis of the 2D embedding plot, usually the second coordinate (for example PC2 or UMAP2).",
    "Training type": "Choose the training workflow. `classic` compares feature-based machine-learning classifiers; `dl` trains a waveform-based deep-learning model.",
    "feature_cols (automatic defaults selected)": "Input measurements for classic machine-learning models. The UI initially selects the available core features (`duration_s`, `blockade_ratio`, `segment_std`, `segment_skew`, `segment_kurt`); an empty selection falls back to those defaults. Choosing a subset changes the inputs used for training and prediction. This selector does not configure waveform-based DL models.",
    "scoring": "Metric used to select the best classic classifier from stratified cross-validation: `accuracy` is the fraction of correctly classified events; `f1` balances precision and recall; `recall` measures how many true events are recovered. For two classes, F1/recall score the last class in sorted label order; for more than two classes they are macro-averaged across classes. Fold scores are combined by fold size.",
    "cv": "Number of stratified cross-validation folds used to estimate performance for classic or DL training. Each fold holds out a portion of labeled events; more folds take longer. For every class to appear in each fold, provide at least `cv` labeled events per class. DL also reserves a stratified validation subset inside each training fold, so very small classes can still fail even when they meet that minimum.",
    "epoch": "Maximum number of full passes through the DL training data. Training stops earlier if validation loss does not improve for 5 consecutive epochs in this UI; increase the cap only if learning is still progressing.",
    "batch_size": "Number of event waveforms used for one DL gradient update. Larger batches can speed training but use more GPU/CPU memory and may affect optimization behavior.",
    "learning_rate": "Step size used by the Adam optimizer to update DL model weights after each batch. Too high can make training unstable; too low can make learning slow. Start with the default and compare validation performance when tuning.",
    "Training result": "Summary of the completed model-training run, including the selected model and available class/performance metadata. Use the plots and prediction table below to inspect validation behavior.",
    'best_pkg["all_samples_feature_pred"]': "Table of labeled events with their feature values and model-predicted classes. Use it to inspect per-event predictions and identify classes or traces with frequent errors.",
    "model_name": "Select a trained model package whose stored cross-validation results will be used for the confusion matrix, metric comparison, and fold-loss plots.",
    "cm split": "Select which cross-validation predictions to summarize in the confusion matrix and metric plot: `train` uses in-fold training predictions and `test` uses held-out fold predictions. Compare them for classic models to look for overfitting; DL packages store held-out test confusion matrices only, so choose `test` for DL.",
    "metric": "Performance measure displayed in the model-comparison plot: accuracy, F1, or recall. For classic models, binary F1/recall score the last class in sorted label order and multiclass F1/recall are macro-averaged across classes; fold scores are combined by fold size. DL F1/recall are macro-averaged even for binary data, and its fold scores are averaged equally. This changes the plotted measure, not the already-trained models.",
    "fold loss type": "Loss curve to display for the selected model across CV folds: `train` shows training loss and `val` shows validation loss. Diverging curves can indicate overfitting.",
    "pl.model_cm": "Confusion matrix for the selected trained model and split. Each row is normalized to 100% of one true class; off-diagonal percentages show which other classes it is mistaken for.",
    "pl.model_metric_bar": "Bar plot comparing trained models on the selected performance metric and data split. Use it to compare scores, not as a substitute for per-class inspection.",
    "pl.plot_fold_loss": "Training or validation loss across cross-validation folds for the selected model. A falling curve indicates learning; a widening train/validation gap can indicate overfitting.",
    "Download model plots (PNG + editable-text PDF)": "Download a ZIP of the displayed confusion matrix, metric comparison, and fold-loss figures as high-resolution PNGs and editable-text PDFs.",
    "Load a saved classical model": "Open this section to register a previously exported classical `.pmmodel` bundle for prediction in the current session. Deep-learning checkpoints are not supported here.",
    "PoreMind classical model bundle": "Choose a `.pmmodel` archive previously exported by PoreMind. Loading joblib-based model files can execute code, so use only bundles from sources you trust.",
    "Loaded model": "Load status and manifest information for the selected classical model bundle. Check for errors before predicting unknown samples.",
    "Upload unknown samples": "Upload one or more ABF/CSV recordings that were not used as labeled training examples. PoreMind applies the stored analysis pipeline and selected model to generate event-level predictions.",
    "Model name (optional)": "Choose which trained model package predicts the uploaded samples. `Automatic (best available)` selects the session's preferred available model; choose a specific name to use that package instead.",
    "Prediction feature_df": "Event-level feature and prediction table for the uploaded unknown recordings. Review predicted classes and feature values to assess each event's classification.",
    "Prediction x": "Numeric feature mapped to the horizontal axis of the unknown-sample scatter plot.",
    "Prediction y": "Numeric feature mapped to the vertical axis of the unknown-sample scatter plot. Non-`blockade_ratio` y values are displayed on a log2 scale in the feature scatter.",
    "Prediction z": "Numeric feature mapped to the third axis when `plot_3d` is selected. It is ignored by 2D, event-current, and stacked-bar plots.",
    "label_col": "Prediction/class column used to color the scatter plot or label events in the event-current view. Choose a predicted-label column such as `pred_label`.",
    "Prediction group column": "Column that defines groups for the stacked-bar prediction plot, such as `trace_id` or `sample_id`. It determines how predicted classes are summarized.",
    "Prediction visualization": "Plot for unknown-sample predictions using the selected method: feature scatter, 3D scatter, event-current labels, or class-count stacked bars.",
    "Output directory": "Folder where Export writes analysis tables, a JSON parameter snapshot, and a Python reproduction script; a trained classical model bundle is included when available. The default is `~/ui_exports`. A missing folder is created automatically; make sure the parent location is writable.",
    "Export result": "Paths and status for the files created by the Export step. Open the listed output folder to review the tables, parameter snapshot, reproduction script, and any included classical model bundle.",
}


_CONTEXT_LABEL_HELP = {
    "zscore_threshold": "Z-score cutoff for the `zscore_threshold` detector. A point is a candidate when its residual crosses this many estimated noise scales in the selected direction (below −z for down events or above +z for up events). Higher values are more conservative and usually detect fewer events.",
    "gmm_components": "Number of Gaussian mixture components fitted to each trace's blockade-ratio distribution by `blockade_gmm`. Components represent blockade populations; more components can model additional populations but may over-split limited data. Two is a practical starting point.",
    "hmm_components": "Number of hidden Gaussian states used by the HMM detector to segment residual current. The state with the lowest fitted mean is treated as the event state; the current HMM implementation does not switch to the highest-mean state for `detect_direction=up`. More states can represent more regimes but increase model complexity.",
    "prediction_sample_id": "Select which uploaded unknown trace to display in the prediction event-current view. Choices appear after unknown recordings are loaded/classified; ABF channel/sweep traces may have separate IDs. This selection is used for the event-current plot, not the 2D/3D or stacked-bar views.",
}


# Keep Gradio's own component chrome in English even when the browser locale is
# Chinese.  Gradio's file uploader and table controls are translated by the
# browser locale independently of the literal labels defined in this module.
_GRADIO_EN_TRANSLATIONS = {
    "common": {
        "clear": "Clear",
        "download": "Download",
        "edit": "Edit",
        "empty": "Empty",
        "error": "Error",
        "loading": "Loading...",
        "or": "or",
        "remove": "Remove",
        "settings": "Settings",
        "submit": "Submit",
    },
    "file": {"uploading": "Uploading..."},
    "upload_text": {
        "click_to_upload": "Click to upload",
        "drop_audio": "Drop audio files here",
        "drop_csv": "Drop CSV files here",
        "drop_file": "Drop files here",
        "drop_image": "Drop image files here",
        "drop_video": "Drop video files here",
        "paste_clipboard": "Paste from clipboard",
    },
    "dataframe": {
        "add_column_left": "Add column to the left",
        "add_column_right": "Add column to the right",
        "add_row_above": "Add row above",
        "add_row_below": "Add row below",
        "clear_sort": "Clear sort",
        "delete_column": "Delete column",
        "delete_row": "Delete row",
        "drop_to_upload": "Drop CSV or TSV files here to import data",
        "new_column": "Add column",
        "new_row": "Add row",
        "sort_ascending": "Sort ascending",
        "sort_descending": "Sort descending",
    },
}


_GRADIO_ZH_TO_EN = {
    "点击上传": "Click to upload",
    "或": "or",
    "将文件拖放到此处": "Drop files here",
    "将文件拖放至此处或": "Drop files here or",
    "将文件拖放到此处或": "Drop files here or",
    "将 CSV 文件拖放到此处": "Drop CSV files here",
    "将音频拖放到此处": "Drop audio files here",
    "将图像拖放到此处": "Drop image files here",
    "将视频拖放到此处": "Drop video files here",
    "正在上传...": "Uploading...",
    "加载中": "Loading...",
    "清除": "Clear",
    "下载": "Download",
    "编辑": "Edit",
    "空": "Empty",
    "错误": "Error",
    "移除": "Remove",
    "分享": "Share",
    "提交": "Submit",
    "撤销": "Undo",
    "设置": "Settings",
    "添加列": "Add column",
    "添加行": "Add row",
    "在上方添加行": "Add row above",
    "在下方添加行": "Add row below",
    "删除行": "Delete row",
    "删除列": "Delete column",
    "在左侧添加列": "Add column to the left",
    "在右侧添加列": "Add column to the right",
    "清除排序": "Clear sort",
    "升序排序": "Sort ascending",
    "降序排序": "Sort descending",
    "正在等待文件上传完成，请稍后重试。": "Waiting for file uploads to finish. Please try again shortly.",
    "连接已丢失。正在重新加入队列...": "Connection lost. Rejoining the queue...",
    "停止": "Stop",
    "继续": "Resume",
    "取消": "Cancel",
}


def _param_classes(key: str) -> list[str]:
    return ["pm-param-help", f"pm-help-{key}"]


def _parameter_guide(label: str, key: str) -> str:
    """Render a CSS-only help badge so the tooltip does not depend on JS."""
    text = escape(_PARAM_HELP[key]["en"])
    name = escape(label)
    return (
        '<div class="pm-param-guide">'
        f'<span class="pm-param-name">{name}</span>'
        f'<span class="pm-help-badge" tabindex="0" title="{text}" data-pm-help="{text}">?</span>'
        f'<span class="pm-help-popover" role="tooltip">{text}</span>'
        '</div>'
    )


def _tooltip_head(background_url: str = "") -> str:
    """Build global UI styles, help popovers, and the scientific background."""
    data_json = json.dumps(_PARAM_HELP, ensure_ascii=False)
    return f"""
<style>
html, body {{ min-width: 0; background: #f4f6f9 !important; }}
body {{ position: relative; isolation: isolate; }}
body::before {{ content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none; background-image: url("{background_url}"); background-repeat: no-repeat; background-position: center top; background-size: min(1680px, 118vw) auto; background-attachment: fixed; opacity: .16; filter: saturate(1.05); }}
.pm-param-help label {{ cursor: help; }}
.pm-param-guide {{ position: relative; display: inline-flex; align-items: center; gap: 6px; margin: 2px 0 4px; font-size: 13px; font-weight: 600; }}
.pm-help-badge {{ display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; border-radius: 50%; background: #475569 !important; color: #ffffff !important; font-size: 11px; font-weight: 700; cursor: help; pointer-events: auto !important; position: relative; z-index: 5; }}
.pm-help-popover {{ display: none; position: absolute; z-index: 2147483647 !important; left: 22px; top: 22px; width: 330px; padding: 9px 11px; border: 1px solid #94a3b8 !important; border-radius: 7px; background: #ffffff !important; color: #111827 !important; box-shadow: 0 5px 18px rgba(0,0,0,.25); font-family: Arial, sans-serif !important; font-size: 12px !important; font-weight: 400 !important; line-height: 1.45 !important; white-space: normal !important; opacity: 1 !important; }}
.pm-help-popover * {{ color: #111827 !important; }}
.pm-param-guide:hover .pm-help-popover, .pm-help-badge:focus + .pm-help-popover {{ display: block; }}
.pm-help-icon {{
  display: inline-flex; align-items: center; justify-content: center;
  width: 16px; height: 16px; margin-left: 5px; border-radius: 50%;
  background: #64748b; color: white; font-size: 11px; font-weight: 700;
  cursor: help; vertical-align: middle;
}}
.pm-help-tip {{
  position: fixed; z-index: 2147483647 !important; max-width: 360px; padding: 9px 11px;
  border: 1px solid #94a3b8 !important; border-radius: 7px; background: #ffffff !important; color: #111827 !important;
  box-shadow: 0 5px 18px rgba(0,0,0,.25); font-family: Arial, sans-serif !important; font-size: 13px !important; line-height: 1.45 !important;
  max-height: min(60vh, 420px); overflow-y: auto; white-space: normal !important; opacity: 1 !important; pointer-events: none; display: none;
}}
.gradio-container {{ box-sizing: border-box !important; position: relative; z-index: 1; width: 1420px !important; max-width: calc(100vw - 32px) !important; min-width: 0 !important; margin: 0 auto !important; padding: 24px 32px 40px !important; background: linear-gradient(90deg, rgba(226, 237, 246, 0) 0%, rgba(226, 237, 246, .36) 16%, rgba(236, 233, 243, .42) 50%, rgba(243, 235, 240, .34) 84%, rgba(243, 235, 240, 0) 100%) !important; color: #172033 !important; font-family: Inter, Segoe UI, Arial, sans-serif !important; }}
.pm-brand-header {{ display: flex; flex-direction: row; align-items: center; justify-content: flex-start; gap: 12px; margin: 0 0 18px; text-align: left; }}
.pm-brand-logo {{ display: block; width: min(240px, 46vw); max-height: 82px; object-fit: contain; object-position: left center; margin: 0; }}
.pm-brand-caption {{ margin: 0 !important; color: #8b95a5 !important; font-size: 12px !important; font-weight: 500 !important; letter-spacing: .025em !important; line-height: 1.3 !important; text-align: left !important; }}
.pm-footer {{ display: flex; align-items: center; justify-content: center; gap: 9px; margin: 14px auto 0; padding: 12px 10px 4px; color: #7b8493 !important; font-size: 20px !important; line-height: 1.4 !important; }}
.pm-footer-link {{ color: #536176 !important; font-weight: 650 !important; text-decoration: none !important; }}
.pm-footer-link:hover {{ color: #2563eb !important; text-decoration: underline !important; }}
.gradio-container > h1, .gradio-container h1 {{ margin: 4px 0 20px !important; color: #172033 !important; font-size: 30px !important; font-weight: 750 !important; letter-spacing: -0.02em !important; }}
.gradio-container h2, .gradio-container h3 {{ color: #25324a !important; font-weight: 700 !important; letter-spacing: -0.01em !important; }}
.gradio-container .tabs {{ border: 1px solid #dbe3ef !important; border-radius: 16px !important; background: #ffffff !important; box-shadow: 0 8px 28px rgba(30, 55, 90, 0.08) !important; overflow: hidden !important; }}
.gradio-container .tab-nav {{ display: grid !important; grid-template-columns: repeat(9, minmax(0, 1fr)) !important; gap: 3px !important; padding: 8px 8px 0 !important; border-bottom: 1px solid #e6ebf2 !important; background: linear-gradient(90deg, #f0fdfa, #eff6ff, #faf5ff, #fff7ed) !important; overflow: visible !important; }}
.gradio-container .tab-nav button {{ width: 100% !important; min-width: 0 !important; border: 0 !important; border-radius: 10px 10px 0 0 !important; color: #64748b !important; font-size: 11px !important; font-weight: 700 !important; line-height: 1.2 !important; white-space: nowrap !important; padding: 10px 4px !important; transition: all .18s ease !important; }}
.gradio-container .tab-nav button:hover {{ color: #2563eb !important; background: #eef5ff !important; }}
.gradio-container .tab-nav button.selected {{ color: #1d4ed8 !important; background: #eaf2ff !important; box-shadow: inset 0 -3px 0 #2563eb !important; }}
.gradio-container .tab-nav button:nth-child(1).selected {{ color: #0f766e !important; background: #ecfdf5 !important; box-shadow: inset 0 -3px 0 #14b8a6 !important; }}
.gradio-container .tab-nav button:nth-child(2).selected {{ color: #1d4ed8 !important; background: #eff6ff !important; box-shadow: inset 0 -3px 0 #3b82f6 !important; }}
.gradio-container .tab-nav button:nth-child(3).selected {{ color: #6d28d9 !important; background: #f5f3ff !important; box-shadow: inset 0 -3px 0 #8b5cf6 !important; }}
.gradio-container .tab-nav button:nth-child(4).selected {{ color: #b45309 !important; background: #fffbeb !important; box-shadow: inset 0 -3px 0 #f59e0b !important; }}
.gradio-container .tab-nav button:nth-child(5).selected {{ color: #0e7490 !important; background: #ecfeff !important; box-shadow: inset 0 -3px 0 #06b6d4 !important; }}
.gradio-container .tab-nav button:nth-child(6).selected {{ color: #be123c !important; background: #fff1f2 !important; box-shadow: inset 0 -3px 0 #f43f5e !important; }}
.gradio-container .tab-nav button:nth-child(7).selected {{ color: #047857 !important; background: #ecfdf5 !important; box-shadow: inset 0 -3px 0 #10b981 !important; }}
.gradio-container .tab-nav button:nth-child(8).selected {{ color: #c2410c !important; background: #fff7ed !important; box-shadow: inset 0 -3px 0 #f97316 !important; }}
.gradio-container .tab-nav button:nth-child(9).selected {{ color: #7c2d12 !important; background: #fff7ed !important; box-shadow: inset 0 -3px 0 #ea580c !important; }}
.gradio-container .tabitem {{ padding: 22px !important; background: #ffffff !important; }}
.gradio-container .tabs, .gradio-container .tabitem {{ box-sizing: border-box !important; width: 100% !important; min-width: 0 !important; }}
.gradio-container .block, .gradio-container .gr-box, .gradio-container .gr-group, .gradio-container .form {{ border-color: #e1e8f1 !important; }}
.gradio-container .gr-group, .gradio-container .gr-box {{ border-radius: 12px !important; background: #fbfcfe !important; }}
.gradio-container .gr-accordion {{ margin: 12px 0 !important; border: 1px solid #dbe3ef !important; border-radius: 12px !important; background: #fbfcfe !important; box-shadow: 0 4px 14px rgba(30, 55, 90, 0.04) !important; }}
.gradio-container .gr-accordion > summary {{ padding: 13px 15px !important; color: #334155 !important; font-weight: 700 !important; }}
.gradio-container label {{ color: #475569 !important; font-size: 12px !important; font-weight: 650 !important; letter-spacing: .01em !important; }}
.gradio-container input, .gradio-container textarea, .gradio-container select {{ border: 1px solid #d5deea !important; border-radius: 9px !important; background: #ffffff !important; color: #172033 !important; box-shadow: 0 1px 2px rgba(30, 55, 90, 0.04) !important; transition: border-color .16s ease, box-shadow .16s ease !important; }}
.gradio-container input:focus, .gradio-container textarea:focus, .gradio-container select:focus {{ border-color: #60a5fa !important; box-shadow: 0 0 0 3px rgba(96, 165, 250, .18) !important; }}
.gradio-container input[type="checkbox"] {{ width: 19px !important; height: 19px !important; min-width: 19px !important; accent-color: #2563eb !important; cursor: pointer !important; }}
.pm-state-checkbox {{ box-sizing: border-box !important; padding: 10px 12px !important; border: 1px solid #d5deea !important; border-radius: 10px !important; background: #ffffff !important; transition: background .16s ease, border-color .16s ease, box-shadow .16s ease !important; }}
.pm-state-checkbox:has(input[type="checkbox"]:checked), .pm-state-checkbox:has([aria-checked="true"]) {{ background: #e8f1ff !important; border-color: #2563eb !important; box-shadow: inset 4px 0 0 #2563eb, 0 2px 8px rgba(37, 99, 235, .12) !important; }}
.pm-state-checkbox:has(input[type="checkbox"]:checked) label, .pm-state-checkbox:has([aria-checked="true"]) label {{ color: #1d4ed8 !important; }}
.pm-state-checkbox [role="checkbox"][aria-checked="true"] {{ background: #2563eb !important; border-color: #1d4ed8 !important; color: #ffffff !important; box-shadow: 0 0 0 3px rgba(37, 99, 235, .18) !important; }}
.pm-state-checkbox input[type="checkbox"]:focus-visible {{ outline: 3px solid rgba(37, 99, 235, .28) !important; outline-offset: 2px !important; }}
.gradio-container button {{ border-radius: 9px !important; font-weight: 700 !important; transition: transform .16s ease, box-shadow .16s ease, background .16s ease !important; }}
.gradio-container button:hover {{ transform: translateY(-1px) !important; box-shadow: 0 5px 14px rgba(30, 55, 90, .12) !important; }}
.gradio-container button.primary {{ background: linear-gradient(135deg, #2563eb, #4f46e5) !important; border-color: #2563eb !important; }}
.gradio-container button.stop {{ background: #fff1f2 !important; color: #be123c !important; border-color: #fecdd3 !important; }}
.pm-param-guide {{ color: #334155 !important; }}
.pm-param-name {{ color: #334155 !important; }}
.pm-help-icon {{ background: #475569 !important; color: #ffffff !important; border: 1px solid #334155 !important; pointer-events: auto !important; position: relative; z-index: 5; }}
.pm-help-icon:hover, .pm-help-icon:focus {{ background: #2563eb !important; outline: 2px solid rgba(96, 165, 250, .35) !important; outline-offset: 1px !important; }}
.gradio-container .progress-bar {{ background: linear-gradient(90deg, #2563eb, #7c3aed) !important; }}
.gradio-container .error, .gradio-container .warning {{ border-radius: 10px !important; }}
.pm-skip-inline-help label {{ cursor: pointer !important; }}
@media (max-width: 900px) {{
  .gradio-container .tab-nav {{ display: flex !important; overflow-x: auto !important; }}
  .gradio-container .tab-nav button {{ flex: 0 0 auto !important; min-width: 132px !important; }}
}}
</style>
<script>
(() => {{
  const help = {data_json};
  let tip = null;
  const isEnglish = () => true;
  const ensureTip = () => {{
    if (tip) return tip;
    tip = document.createElement("div");
    tip.className = "pm-help-tip";
    document.body.appendChild(tip);
    return tip;
  }};
  const hideTip = () => {{ if (tip) tip.style.display = "none"; }};
  const showTip = (icon, key) => {{
    const item = help[key];
    if (!item) return;
    const box = ensureTip();
    box.textContent = isEnglish() ? item.en : item.zh;
    const rect = icon.getBoundingClientRect();
    box.style.left = Math.min(rect.left, window.innerWidth - 380) + "px";
    box.style.top = Math.min(rect.bottom + 7, window.innerHeight - 100) + "px";
    box.style.display = "block";
  }};
  const bind = () => {{
    document.querySelectorAll(".pm-param-help").forEach((node) => {{
      if (node.dataset.pmHelpBound === "1") return;
      const keyClass = Array.from(node.classList).find((x) => x.startsWith("pm-help-"));
      const key = keyClass ? keyClass.slice("pm-help-".length) : "";
      if (!help[key]) return;
      const label = node.querySelector("label");
      if (!label || label.querySelector(".pm-help-icon")) return;
      if (label.textContent.includes("[?]")) {{
        label.title = isEnglish() ? help[key].en : help[key].zh;
        node.dataset.pmHelpBound = "1";
        return;
      }}
      const icon = document.createElement("span");
      icon.className = "pm-help-icon";
      icon.textContent = "?";
      icon.setAttribute("aria-label", "parameter help");
      icon.addEventListener("mouseenter", () => showTip(icon, key));
      icon.addEventListener("mouseleave", hideTip);
      label.appendChild(icon);
      node.dataset.pmHelpBound = "1";
    }});
  }};
  [0, 300, 1000, 2500].forEach((delay) => window.setTimeout(bind, delay));
  window.addEventListener("resize", hideTip);
}})();
</script>
"""


def _tooltip_js() -> str:
    """Bind durable help popovers and force Gradio chrome to English."""
    data_json = json.dumps(_PARAM_HELP, ensure_ascii=False)
    label_data_json = json.dumps(_LABEL_HELP, ensure_ascii=False)
    context_data_json = json.dumps(_CONTEXT_LABEL_HELP, ensure_ascii=False)
    ui_translation_json = json.dumps(_GRADIO_ZH_TO_EN, ensure_ascii=False)
    return f"""
() => {{
  const help = {data_json};
  const labelHelp = {label_data_json};
  const contextHelp = {context_data_json};
  const zhToEn = {ui_translation_json};
  let tip = null;
  const normalize = (text) => text.split(/\\s+/).join(" ").replace("[?]", "").trim();
  const ensureTip = () => {{
    if (tip) return tip;
    tip = document.createElement("div");
    tip.className = "pm-help-tip";
    document.body.appendChild(tip);
    return tip;
  }};
  const hideTip = () => {{ if (tip) tip.style.display = "none"; }};
  const showTip = (icon, text) => {{
    if (!text) return;
    const box = ensureTip();
    box.textContent = text;
    const rect = icon.getBoundingClientRect();
    box.style.left = Math.max(8, Math.min(rect.left, window.innerWidth - 380)) + "px";
    box.style.top = Math.min(rect.bottom + 7, window.innerHeight - 100) + "px";
    box.style.display = "block";
  }};
  const translateTextNode = (node) => {{
    const raw = node.nodeValue || "";
    let translated = raw;
    Object.entries(zhToEn)
      .sort(([left], [right]) => right.length - left.length)
      .forEach(([source, target]) => {{
        if (translated.includes(source)) translated = translated.split(source).join(target);
      }});
    if (translated !== raw) node.nodeValue = translated;
  }};
  const translateGradioChrome = () => {{
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);
    textNodes.forEach((node) => {{
      const parent = node.parentElement;
      if (!parent || parent.closest(".pm-help-tip, .pm-param-guide")) return;
      translateTextNode(node);
    }});
    document.querySelectorAll("[title], [aria-label], [placeholder]").forEach((node) => {{
      ["title", "aria-label", "placeholder"].forEach((attr) => {{
        const value = node.getAttribute(attr);
        if (value && zhToEn[value]) node.setAttribute(attr, zhToEn[value]);
      }});
    }});
  }};
  const itemForLabel = (label) => {{
    let ancestor = label;
    while (ancestor && ancestor !== document.body) {{
      const contextClass = Array.from(ancestor.classList || []).find((name) => name.startsWith("pm-tooltip-context-"));
      if (contextClass) {{
        const contextKey = contextClass.slice("pm-tooltip-context-".length);
        if (contextHelp[contextKey]) return {{ en: contextHelp[contextKey] }};
      }}
      ancestor = ancestor.parentElement;
    }}
    const name = normalize(label.textContent || "");
    if (!name) return null;
    if (labelHelp[name]) return {{ en: labelHelp[name] }};
    const key = name.toLowerCase().replace(/[^a-z0-9]+(.)/g, (_, ch) => ch.toUpperCase());
    if (help[key]) return help[key];
    return {{ en: `Help for "${{name}}" has not been configured yet.` }};
  }};
  const bindLabel = (label) => {{
    if (!label || label.dataset.pmHelpBound === "1") return;
    if (label.closest("button")) return;
    if (label.closest(".pm-skip-inline-help")) return;
    const item = itemForLabel(label);
    if (!item) return;
    label.setAttribute("title", item.en);
    const existing = label.querySelector(".pm-help-icon");
    if (existing) {{
      existing.dataset.pmHelp = item.en;
      existing.setAttribute("title", item.en);
      existing.setAttribute("tabindex", "0");
      label.dataset.pmHelpBound = "1";
      return;
    }}
    const icon = document.createElement("span");
    icon.className = "pm-help-icon";
    icon.textContent = "?";
    icon.setAttribute("tabindex", "0");
    icon.setAttribute("aria-label", "parameter help");
    icon.setAttribute("title", item.en);
    icon.dataset.pmHelp = item.en;
    label.appendChild(icon);
    label.dataset.pmHelpBound = "1";
  }};
  const bind = () => {{
    document.querySelectorAll("label").forEach(bindLabel);
  }};
  if (!document.body.dataset.pmHelpDelegateBound) {{
    const helpTarget = (target) => target && target.nodeType === 1
      ? target.closest(".pm-help-icon") : null;
    document.addEventListener("pointerover", (event) => {{
      const icon = helpTarget(event.target);
      if (icon) showTip(icon, icon.dataset.pmHelp || icon.getAttribute("title"));
    }});
    document.addEventListener("focusin", (event) => {{
      const icon = helpTarget(event.target);
      if (icon) showTip(icon, icon.dataset.pmHelp || icon.getAttribute("title"));
    }});
    document.addEventListener("pointerout", (event) => {{
      if (helpTarget(event.target)) hideTip();
    }});
    document.addEventListener("focusout", (event) => {{
      if (helpTarget(event.target)) hideTip();
    }});
    document.body.dataset.pmHelpDelegateBound = "1";
  }}
  translateGradioChrome();
  bind();
  new MutationObserver(() => {{ translateGradioChrome(); bind(); }}).observe(document.body, {{ childList: true, subtree: true }});
  window.addEventListener("resize", hideTip);
}}
"""


def _tooltip_html_js() -> str:
    """Use gr.HTML's post-render hook to set native browser tooltips on labels."""
    data_json = json.dumps(_PARAM_HELP, ensure_ascii=False)
    return f"""
const help = {data_json};
const applyHelp = () => {{
  document.querySelectorAll('.pm-param-help').forEach((node) => {{
    const keyClass = Array.from(node.classList).find((x) => x.startsWith('pm-help-'));
    const key = keyClass ? keyClass.slice('pm-help-'.length) : '';
    const label = node.querySelector('label');
    if (!help[key] || !label) return;
    label.setAttribute('title', help[key].en);
    label.setAttribute('data-poremind-help', help[key].en);
  }});
}};
applyHelp();
setTimeout(applyHelp, 250);
setTimeout(applyHelp, 1000);
new MutationObserver(applyHelp).observe(document.body, {{ childList: true, subtree: true }});
"""


@contextmanager
def _route_tqdm_to_gradio(progress):
    """Route tqdm loops used by PoreMind into Gradio's progress component."""
    if progress is None or not hasattr(progress, "tqdm"):
        yield
        return

    patched = []

    def tqdm_wrapper(iterable=None, *args, **kwargs):
        desc = kwargs.get("desc")
        if desc is None and args:
            desc = args[0]
        return progress.tqdm(
            iterable,
            desc=desc,
            total=kwargs.get("total"),
            unit=kwargs.get("unit", "steps"),
        )

    for module_name in ("tqdm.auto", "tqdm"):
        try:
            module = __import__(module_name, fromlist=["tqdm"])
        except Exception:
            continue
        original = getattr(module, "tqdm", None)
        if original is None:
            continue
        patched.append((module, original))
        module.tqdm = tqdm_wrapper

    try:
        yield
    finally:
        for module, original in patched:
            module.tqdm = original


def _to_file_map(files: list | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in files or []:
        p = Path(getattr(item, "name", str(item)))
        mapping[p.stem] = str(p)
    return mapping


def _sample_annotation_rows(files: list | None) -> list[list[str]]:
    """Create one editable annotation row for every file selected in the UI."""
    return [[sample_id, sample_id] for sample_id in _to_file_map(files)]


def _annotation_text(value) -> str:
    """Normalize an editable annotation cell without turning missing values into 'nan'."""
    if value is None:
        return ""
    try:
        if math.isnan(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"<na>", "none"} else text


def _parse_sample_annotations(table, sample_paths: dict[str, str]) -> dict[str, str]:
    """Convert Gradio's editable annotation table into sample_to_group."""
    records: list[object] = []
    if table is not None and hasattr(table, "to_dict"):
        records = list(table.to_dict(orient="records"))
    elif isinstance(table, dict):
        headers = table.get("headers", [])
        data = table.get("data", [])
        records = [dict(zip(headers, row)) for row in data]
    elif isinstance(table, (list, tuple)):
        records = list(table)

    annotations: dict[str, str] = {}
    for row in records:
        if isinstance(row, dict):
            sample_id = _annotation_text(row.get("sample_id", ""))
            group = _annotation_text(row.get("group annotation", ""))
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            sample_id = _annotation_text(row[0])
            group = _annotation_text(row[1])
        else:
            continue
        if sample_id in sample_paths:
            annotations[sample_id] = group or sample_id

    # Every uploaded source must have a deterministic label even if its table
    # row is removed or left empty.  Users can edit the annotation before load.
    return {sample_id: annotations.get(sample_id, sample_id) for sample_id in sample_paths}


def _default_feature_columns(df) -> list[str]:
    """Return the same core feature defaults used by PoreMind model training."""
    if df is None:
        return []
    available = set(select_feature_columns(df))
    return [column for column in _MODEL_DEFAULT_FEATURE_COLS if column in available]


def _none_if_blank(v):
    if v is None:
        return None
    t = str(v).strip()
    if t == "":
        return None
    return float(t)

def _parse_pair(text: str | None) -> tuple[float, float] | None:
    """Parse 'lo,hi' string into tuple. Returns None if blank."""
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    parts = [p.strip() for p in t.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"expected 'lo,hi' pair, got: {text!r}")
    return float(parts[0]), float(parts[1])


def _plot_limit(lower, upper, axis_name: str) -> tuple[float, float] | None:
    """Parse optional numeric plot bounds, requiring either both or neither."""
    try:
        low = _none_if_blank(lower)
        high = _none_if_blank(upper)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{axis_name} limits must be numbers or left blank for automatic scaling.") from exc
    if low is None and high is None:
        return None
    if low is None or high is None:
        raise ValueError(f"Set both {axis_name} min and max, or leave both blank.")
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"{axis_name} limits must be finite numbers.")
    if high <= low:
        raise ValueError(f"{axis_name} max must be greater than {axis_name} min.")
    return low, high


def _plot_axis_limit(column: str, lower, upper, axis_name: str) -> tuple[float, float] | None:
    """Use an explicit range when set; otherwise constrain blockade_ratio to [0, 1]."""
    manual_limit = _plot_limit(lower, upper, axis_name)
    if manual_limit is not None:
        return manual_limit
    if column == "blockade_ratio":
        return 0.0, 1.0
    return None


def _choose_windows_folder(initial_path: str | Path | None = None) -> str | None:
    """Open the native Windows folder picker from an STA thread and return its path."""
    if os.name != "nt":
        raise RuntimeError("The native folder picker is available on Windows; you can enter the path directly.")

    from ctypes import wintypes

    class _BrowseInfoW(ctypes.Structure):
        _fields_ = [
            ("hwndOwner", wintypes.HWND),
            ("pidlRoot", ctypes.c_void_p),
            ("pszDisplayName", wintypes.LPWSTR),
            ("lpszTitle", wintypes.LPCWSTR),
            ("ulFlags", wintypes.UINT),
            ("lpfn", ctypes.c_void_p),
            ("lParam", wintypes.LPARAM),
            ("iImage", ctypes.c_int),
        ]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.restype = wintypes.HWND
    owner = user32.GetForegroundWindow()
    selection: dict[str, str | None] = {"path": None}
    failure: list[BaseException] = []
    initial = Path(initial_path or Path.home()).expanduser()
    if not initial.is_dir():
        initial = initial.parent if initial.parent.is_dir() else Path.home()
    initial_buffer = ctypes.create_unicode_buffer(str(initial.resolve()))

    def show_dialog() -> None:
        ole32 = ctypes.WinDLL("ole32", use_last_error=True)
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        initialized = False
        pidl = None
        try:
            ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
            ole32.CoInitializeEx.restype = ctypes.c_long
            hr = ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED
            if hr < 0:
                raise OSError(f"Could not initialize the folder dialog (HRESULT 0x{hr & 0xFFFFFFFF:08X}).")
            initialized = True

            shell32.SHBrowseForFolderW.argtypes = [ctypes.POINTER(_BrowseInfoW)]
            shell32.SHBrowseForFolderW.restype = ctypes.c_void_p
            shell32.SHGetPathFromIDListW.argtypes = [ctypes.c_void_p, wintypes.LPWSTR]
            shell32.SHGetPathFromIDListW.restype = wintypes.BOOL
            ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
            ole32.CoTaskMemFree.restype = None

            send_message = user32.SendMessageW
            send_message.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            send_message.restype = ctypes.c_ssize_t
            BrowseCallback = ctypes.WINFUNCTYPE(
                ctypes.c_int,
                wintypes.HWND,
                wintypes.UINT,
                wintypes.LPARAM,
                wintypes.LPARAM,
            )

            def set_initial_folder(hwnd, message, _lparam, _data):
                if message == 1:  # BFFM_INITIALIZED
                    send_message(
                        hwnd,
                        0x0467,  # BFFM_SETSELECTIONW
                        1,
                        ctypes.cast(initial_buffer, ctypes.c_void_p).value,
                    )
                return 0

            browse_callback = BrowseCallback(set_initial_folder)
            display_name_buffer = ctypes.create_unicode_buffer(32768)
            browse = _BrowseInfoW(
                hwndOwner=owner,
                pidlRoot=None,
                pszDisplayName=ctypes.cast(display_name_buffer, wintypes.LPWSTR),
                lpszTitle="Choose the export folder",
                ulFlags=0x0001 | 0x0010 | 0x0040,  # filesystem folders + editable path + modern dialog
                lpfn=ctypes.cast(browse_callback, ctypes.c_void_p).value,
                lParam=0,
                iImage=0,
            )
            pidl = shell32.SHBrowseForFolderW(ctypes.byref(browse))
            if not pidl:
                return
            path_buffer = ctypes.create_unicode_buffer(32768)
            if not shell32.SHGetPathFromIDListW(pidl, path_buffer):
                raise OSError("The selected folder path could not be read.")
            selection["path"] = path_buffer.value
        except BaseException as exc:
            failure.append(exc)
        finally:
            if pidl:
                ole32.CoTaskMemFree(pidl)
            if initialized:
                ole32.CoUninitialize()

    dialog_thread = threading.Thread(target=show_dialog, name="poremind-folder-picker", daemon=True)
    dialog_thread.start()
    dialog_thread.join()
    if failure:
        raise RuntimeError(f"Could not open the folder picker: {failure[0]}") from failure[0]
    return selection["path"]


def _save_plot_bundle(named_figures: list[tuple[str, object]]) -> str | None:
    """Save matplotlib figures as high-resolution PNG and editable vector PDF."""
    figures = [(name, fig) for name, fig in named_figures if fig is not None and hasattr(fig, "savefig")]
    if not figures:
        return None

    import matplotlib as mpl

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    out_dir = Path.cwd() / "ui_exports" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"poremind_plots_{uuid4().hex[:10]}.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for name, fig in figures:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_.") or "plot"
            for fmt in ("png", "pdf"):
                buffer = BytesIO()
                metadata = {"Title": safe_name, "Creator": "PoreMind UI"} if fmt == "pdf" else None
                save_kwargs = {"format": fmt, "bbox_inches": "tight"}
                if fmt == "png":
                    save_kwargs["dpi"] = 300
                else:
                    save_kwargs["metadata"] = metadata
                fig.savefig(buffer, **save_kwargs)
                archive.writestr(f"{safe_name}.{fmt}", buffer.getvalue())
    return str(archive_path)


def create_app():
    try:
        import gradio as gr
    except Exception as exc:  # pragma: no cover
        raise ImportError("Gradio is required for UI. Install with `pip install gradio`.") from exc

    ctl = AnalysisController()
    assets_path = Path(__file__).with_name("assets")
    background_path = assets_path / "poremind_science_background.png"
    if not background_path.is_file():  # pragma: no cover - installation safeguard
        raise FileNotFoundError(f"Missing UI background asset: {background_path}")
    logo_path = assets_path / "poremind_logo.png"
    if not logo_path.is_file():  # pragma: no cover - installation safeguard
        raise FileNotFoundError(f"Missing PoreMind logo image: {logo_path}")
    gr.set_static_paths([background_path, logo_path])
    ui_i18n = gr.I18n(
        **{
            "en": _GRADIO_EN_TRANSLATIONS,
            "zh": _GRADIO_EN_TRANSLATIONS,
            "zh-CN": _GRADIO_EN_TRANSLATIONS,
            "zh-Hans": _GRADIO_EN_TRANSLATIONS,
        }
    )

    with gr.Blocks(title="PoreMind", fill_width=True) as demo:
        logo_url = demo.serve_static_file(logo_path)["url"]
        logo_url = quote(logo_url.replace("\\", "/"), safe="/:=")
        gr.HTML(
            '<div class="pm-brand-header">'
            f'<img class="pm-brand-logo" src="{escape(logo_url)}" alt="PoreMind logo">'
            '<div class="pm-brand-caption">PoreMind Analysis Platform</div>'
            '</div>'
        )

        with gr.Tab("1 · Import"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML(_parameter_guide("Upload ABF/CSV files", "file_input"))
                    file_input = gr.File(label="Upload ABF/CSV files", file_count="multiple", show_label=False)
                    gr.HTML(_parameter_guide("Reader", "reader"))
                    reader = gr.Dropdown(
                        choices=["abf", "csv"],
                        value="abf",
                        label="Reader [?]",
                        show_label=False,
                        elem_classes=_param_classes("reader"),
                    )
                    gr.HTML(_parameter_guide("Sample annotations", "sample_annotations"))
                    sample_annotation_table = gr.Dataframe(
                        headers=["sample_id", "group annotation"],
                        datatype=["str", "str"],
                        type="pandas",
                        row_count=(0, "dynamic"),
                        column_count=(2, "fixed"),
                        interactive=True,
                        label="Sample annotations",
                        value=[],
                    )
                    with gr.Row():
                        load_btn = gr.Button("Load samples", scale=4)
                        load_stop_btn = gr.Button("Stop", variant="stop", scale=1)
                    load_summary = gr.JSON(label="Load summary")
                with gr.Column(scale=4):
                    sample_df = gr.Dataframe(label="Sample information")

        with gr.Tab("2 · Preprocess"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### Analysis parameters")
                    gr.HTML(_parameter_guide("Denoising method", "denoise_method"))
                    denoise_method = gr.Dropdown(
                        choices=["butterworth_filtfilt", "moving_average", "median", "drift_corrected_moving_average", "none"],
                        value="butterworth_filtfilt",
                        label="Denoising method [?]",
                        show_label=False,
                        elem_classes=_param_classes("denoise_method"),
                    )
                    with gr.Group(visible=True) as grp_bw:
                        bw_n = gr.Slider(1, 8, value=2, step=1, label="filtfilt_N")
                        bw_wn = gr.Slider(0.01, 0.99, value=0.1, step=0.01, label="filtfilt_Wn")
                    with gr.Group(visible=False) as grp_ma:
                        ma_window = gr.Slider(1, 101, value=5, step=2, label="window")
                    with gr.Group(visible=False) as grp_median:
                        median_window = gr.Slider(1, 101, value=5, step=2, label="window")
                    with gr.Group(visible=False) as grp_drift:
                        drift_window = gr.Slider(51, 5001, value=1001, step=50, label="drift_window")
                        smooth_window = gr.Slider(1, 101, value=5, step=2, label="smooth_window")
                    with gr.Row():
                        denoise_run_btn = gr.Button("Run denoising", scale=4)
                        denoise_stop_btn = gr.Button("Stop", variant="stop", scale=1)
                    denoise_result = gr.JSON(label="Preprocessing status")
                with gr.Column(scale=4):
                    gr.Markdown("### Plot parameters")
                    denoise_sample = gr.Dropdown(choices=[], value=None, label="sample_id")
                    denoise_current = gr.Dropdown(choices=["denoise", "raw"], value="denoise", label="current")
                    denoise_start = gr.Number(label="start_ms", value=0.0)
                    denoise_end = gr.Number(label="end_ms", value=100.0)
                    denoise_plot_btn = gr.Button("Plot pl.current")
                    denoise_plot = gr.Plot(label="pl.current")
                    denoise_plot_download = gr.File(label="Download plot files (PNG + editable-text PDF)", file_types=[".zip"], interactive=False)
                    preview_btn = gr.Button("Preview preview_signal table")
                    preview_table = gr.Dataframe(label="analysis.preview_signal(...).head()")

        with gr.Tab("3 · Pre-Events"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### Tune detector parameters on one short trace")
                    gr.Markdown("Use this step to test settings on a selected trace. It does not replace full event detection in the next step.")
                    gr.HTML(_parameter_guide("detect_method", "detect_method"))
                    detect_method = gr.Dropdown(choices=["threshold", "zscore_threshold", "cusum", "pelt", "hmm"], value="threshold", label="detect_method", show_label=False, elem_classes=_param_classes("detect_method"))
                    gr.HTML(_parameter_guide("detect_direction", "detect_direction"))
                    detect_direction = gr.Dropdown(choices=["down", "up"], value="up", label="detect_direction", show_label=False, elem_classes=_param_classes("detect_direction"))
                    gr.HTML(_parameter_guide("baseline_method", "baseline_method"))
                    baseline_method = gr.Dropdown(choices=["rolling_quantile", "global_quantile", "global_median"], value="global_quantile", label="baseline_method", show_label=False, elem_classes=_param_classes("baseline_method"))
                    with gr.Group(visible=False) as preview_baseline_window_group:
                        b_window = gr.Slider(11, 30001, value=10000, step=10, label="baseline window")
                    b_q = gr.Slider(1, 99, value=5, step=1, label="baseline q (%)")
                    preview_merge = gr.Checkbox(label="merge_event", value=True, elem_classes=["pm-state-checkbox"])
                    preview_merge_gap = gr.Number(label="merge_gap_ms", value=2)

                    with gr.Group(visible=True) as grp_det_threshold:
                        d_sigma_k = gr.Number(label="sigma_k", value=4.0)
                        d_min_duration = gr.Number(label="min_duration_s", value=0.0)
                        d_noise_method = gr.Dropdown(choices=["mad", "std"], value="mad", label="noise_method")
                    with gr.Group(visible=False) as grp_det_z:
                        z_thr = gr.Number(label="z", value=4.0, elem_classes=["pm-tooltip-context-zscore_threshold"])
                        z_min_duration = gr.Number(label="min_duration_s", value=0.0)
                        z_noise_method = gr.Dropdown(choices=["mad", "std"], value="mad", label="noise_method")
                    with gr.Group(visible=False) as grp_det_cusum:
                        c_drift = gr.Number(label="drift", value=0.02)
                        c_threshold = gr.Number(label="threshold", value=8.0)
                        c_min_duration = gr.Number(label="min_duration_s", value=0.0)
                        c_noise_method = gr.Dropdown(choices=["mad", "std"], value="mad", label="noise_method")
                    with gr.Group(visible=False) as grp_det_pelt:
                        p_model = gr.Dropdown(choices=["l1", "l2", "rbf"], value="l2", label="model")
                        p_penalty = gr.Number(label="penalty", value=8.0)
                        p_sigma_k = gr.Number(label="sigma_k", value=3.0)
                        p_min_duration = gr.Number(label="min_duration_s", value=0.0)
                        p_noise_method = gr.Dropdown(choices=["mad", "std"], value="mad", label="noise_method")
                    with gr.Group(visible=False) as grp_det_hmm:
                        h_n_components = gr.Slider(2, 6, value=2, step=1, label="n_components", elem_classes=["pm-tooltip-context-hmm_components"])
                        h_cov = gr.Dropdown(choices=["diag", "full", "tied", "spherical"], value="diag", label="covariance_type")
                        h_n_iter = gr.Slider(20, 500, value=200, step=10, label="n_iter")
                        h_min_duration = gr.Number(label="min_duration_s", value=0.0)

                    gr.Markdown("#### Preview scope")
                    preview_sample = gr.Dropdown(choices=[], value=None, label="sample_id")
                    preview_current = gr.Dropdown(choices=["denoise", "raw"], value="denoise", label="current")
                    preview_start = gr.Number(label="start_ms", value=0)
                    preview_end = gr.Number(label="end_ms", value=10000)
                    gr.HTML(_parameter_guide("Use a current range for baseline/noise statistics", "exclude_current"))
                    preview_exclude = gr.Checkbox(label="Use a current range for baseline/noise statistics", value=False, elem_classes=["pm-skip-inline-help", "pm-state-checkbox"])
                    with gr.Group(visible=False) as preview_current_range_group:
                        preview_ex_min = gr.Textbox(label="current-range min (blank = None)", value="")
                        preview_ex_max = gr.Textbox(label="current-range max (blank = None)", value="-10")
                    with gr.Row():
                        preview_run_btn = gr.Button("Run detect_events_simple", scale=4)
                        preview_stop_btn = gr.Button("Stop", variant="stop", scale=1)
                    preview_result = gr.JSON(label="Simple detection result")

                with gr.Column(scale=4):
                    gr.Markdown("### Preview plots and table")
                    preview_plot_sample = gr.Dropdown(choices=[], value=None, label="sample_id")
                    preview_plot_current = gr.Dropdown(choices=["denoise", "raw"], value="denoise", label="current")
                    preview_plot_start_event = gr.Slider(1, 100, value=1, step=1, label="start_event")
                    preview_plot_end_event = gr.Slider(1, 200, value=10, step=1, label="end_event")
                    preview_plot_btn = gr.Button("Plot pl.event_current_simple")
                    preview_plot = gr.Plot(label="pl.event_current_simple")
                    preview_plot_download = gr.File(label="Download plot files (PNG + editable-text PDF)", file_types=[".zip"], interactive=False)
                    preview_event_table = gr.Dataframe(label="analysis.simple_events")

        with gr.Tab("4 · Events"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### Run full event detection")
                    gr.Markdown("All detector settings are synchronized with Pre-Events. Changes made here are also used by the local preview.")
                    gr.HTML(_parameter_guide("detect_method", "detect_method"))
                    event_detect_method = gr.Dropdown(choices=["threshold", "zscore_threshold", "cusum", "pelt", "hmm"], value="threshold", label="detect_method", show_label=False, elem_classes=_param_classes("detect_method"))
                    gr.HTML(_parameter_guide("detect_direction", "detect_direction"))
                    event_detect_direction = gr.Dropdown(choices=["down", "up"], value="up", label="detect_direction", show_label=False, elem_classes=_param_classes("detect_direction"))
                    gr.HTML(_parameter_guide("baseline_method", "baseline_method"))
                    global_baseline_method = gr.Dropdown(choices=["rolling_quantile", "global_quantile", "global_median"], value="global_quantile", label="baseline_method", show_label=False, elem_classes=_param_classes("baseline_method"))
                    with gr.Group(visible=False) as global_baseline_window_group:
                        global_b_window = gr.Slider(11, 30001, value=10000, step=10, label="baseline window")
                    global_b_q = gr.Slider(1, 99, value=5, step=1, label="baseline q (%)")

                    with gr.Group(visible=True) as event_grp_det_threshold:
                        event_d_sigma_k = gr.Number(label="sigma_k", value=4.0)
                        event_d_min_duration = gr.Number(label="min_duration_s", value=0.0)
                        event_d_noise_method = gr.Dropdown(choices=["mad", "std"], value="mad", label="noise_method")
                    with gr.Group(visible=False) as event_grp_det_z:
                        event_z_thr = gr.Number(label="z", value=4.0, elem_classes=["pm-tooltip-context-zscore_threshold"])
                        event_z_min_duration = gr.Number(label="min_duration_s", value=0.0)
                        event_z_noise_method = gr.Dropdown(choices=["mad", "std"], value="mad", label="noise_method")
                    with gr.Group(visible=False) as event_grp_det_cusum:
                        event_c_drift = gr.Number(label="drift", value=0.02)
                        event_c_threshold = gr.Number(label="threshold", value=8.0)
                        event_c_min_duration = gr.Number(label="min_duration_s", value=0.0)
                        event_c_noise_method = gr.Dropdown(choices=["mad", "std"], value="mad", label="noise_method")
                    with gr.Group(visible=False) as event_grp_det_pelt:
                        event_p_model = gr.Dropdown(choices=["l1", "l2", "rbf"], value="l2", label="model")
                        event_p_penalty = gr.Number(label="penalty", value=8.0)
                        event_p_sigma_k = gr.Number(label="sigma_k", value=3.0)
                        event_p_min_duration = gr.Number(label="min_duration_s", value=0.0)
                        event_p_noise_method = gr.Dropdown(choices=["mad", "std"], value="mad", label="noise_method")
                    with gr.Group(visible=False) as event_grp_det_hmm:
                        event_h_n_components = gr.Slider(2, 6, value=2, step=1, label="n_components", elem_classes=["pm-tooltip-context-hmm_components"])
                        event_h_cov = gr.Dropdown(choices=["diag", "full", "tied", "spherical"], value="diag", label="covariance_type")
                        event_h_n_iter = gr.Slider(20, 500, value=200, step=10, label="n_iter")
                        event_h_min_duration = gr.Number(label="min_duration_s", value=0.0)

                    global_merge = gr.Checkbox(label="merge_event", value=True, elem_classes=["pm-state-checkbox"])
                    global_merge_gap = gr.Number(label="merge_gap_ms", value=2)
                    gr.HTML(_parameter_guide("Use a current range for baseline/noise statistics", "exclude_current"))
                    global_exclude = gr.Checkbox(label="Use a current range for baseline/noise statistics", value=False, elem_classes=["pm-skip-inline-help", "pm-state-checkbox"])
                    with gr.Group(visible=False) as global_current_range_group:
                        global_ex_min = gr.Textbox(label="current-range min (blank = None)", value="")
                        global_ex_max = gr.Textbox(label="current-range max (blank = None)", value="-10")
                    with gr.Row():
                        global_run_btn = gr.Button("Run detect_events", scale=4)
                        global_stop_btn = gr.Button("Stop", variant="stop", scale=1)
                    global_result = gr.JSON(label="Full detection result")

                with gr.Column(scale=4):
                    gr.Markdown("### Full-detection plots and table")
                    global_plot_sample = gr.Dropdown(choices=[], value=None, label="sample_id")
                    global_plot_current = gr.Dropdown(choices=["denoise", "raw"], value="denoise", label="current")
                    global_plot_start_event = gr.Slider(1, 100, value=1, step=1, label="start_event")
                    global_plot_end_event = gr.Slider(1, 200, value=10, step=1, label="end_event")
                    global_plot_btn = gr.Button("Plot pl.event_current")
                    global_plot = gr.Plot(label="pl.event_current")
                    global_plot_download = gr.File(label="Download plot files (PNG + editable-text PDF)", file_types=[".zip"], interactive=False)
                    global_event_table = gr.Dataframe(label="analysis.events[sample_id]")

        with gr.Tab("5 · Features & Filter"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### Analysis parameters")
                    max_events = gr.Number(label="max_event_per_sample", value=1000)
                    use_custom = gr.Checkbox(label="Enable custom_shape_features", value=False, elem_classes=["pm-state-checkbox"])
                    with gr.Row():
                        extract_btn = gr.Button("Run extract_features", scale=4)
                        extract_stop_btn = gr.Button("Stop", variant="stop", scale=1)
                    feat_df = gr.Dataframe(label="feature_df")

                    gr.HTML(_parameter_guide("filter method", "filter_method"))
                    filter_method = gr.Dropdown(choices=["blockade_gmm", "isolation_forest", "lof", "knn_background"], value="blockade_gmm", label="filter method", show_label=False, elem_classes=_param_classes("filter_method"))
                    with gr.Group(visible=True) as grp_filter_gmm:
                        f_n_components = gr.Slider(2, 6, value=2, step=1, label="n_components", elem_classes=["pm-tooltip-context-gmm_components"])
                        f_prior_mean = gr.Textbox(label="prior_mean (blank = None)", value="")
                    with gr.Group(visible=False) as grp_filter_if:
                        f_if_contam = gr.Slider(0.01, 0.4, value=0.05, step=0.01, label="contamination")
                    with gr.Group(visible=False) as grp_filter_lof:
                        f_lof_contam = gr.Slider(0.01, 0.4, value=0.05, step=0.01, label="contamination")
                    with gr.Group(visible=False) as grp_filter_knn:
                        f_knn_bg_ids = gr.Dropdown(choices=[], value=[], multiselect=True, label="background_sample_ids")
                        f_knn_k = gr.Slider(1, 50, value=10, step=1, label="k")
                        f_knn_n_match = gr.Slider(1, 50, value=1, step=1, label="n_noise_match")
                        f_knn_bg_ratio = gr.Slider(0.0, 5.0, value=1.0, step=0.05, label="background_ratio")
                        f_knn_features = gr.Textbox(label="feature_cols (comma-separated; blank = default)", value="")
                        f_knn_bg_lim = gr.Textbox(label="background_blockage_lim (lo,hi; blank = disabled)", value="")
                    block_lo = gr.Number(label="blockage_lim low", value=0)
                    block_hi = gr.Number(label="blockage_lim high", value=1.0)
                    with gr.Row():
                        filter_btn = gr.Button("Run filter_events", scale=4)
                        filter_stop_btn = gr.Button("Stop", variant="stop", scale=1)
                    filtered_df = gr.Dataframe(label="filtered_df")

                with gr.Column(scale=4):
                    gr.Markdown("### Plot parameters")
                    vis_target = gr.Dropdown(choices=["feature", "filtered"], value="feature", label="Data source")
                    vis_method = gr.Dropdown(choices=["plot_2d", "plot_3d", "box_significance"], value="plot_2d", label="Visualization method")
                    with gr.Group(visible=True) as vis_axis_group:
                        vis_x = gr.Dropdown(choices=["blockade_ratio"], value="blockade_ratio", label="x")
                        vis_y = gr.Dropdown(choices=["duration_s"], value="duration_s", label="y")
                        with gr.Group(visible=False) as vis_z_group:
                            vis_z = gr.Dropdown(choices=["segment_std"], value="segment_std", label="z")
                        with gr.Row():
                            vis_x_min = gr.Textbox(label="x min (blank: blockade_ratio = 0; else auto)", value="", placeholder="Automatic")
                            vis_x_max = gr.Textbox(label="x max (blank: blockade_ratio = 1; else auto)", value="", placeholder="Automatic")
                        with gr.Row():
                            vis_y_min = gr.Textbox(label="y min (blank: blockade_ratio = 0; else auto)", value="", placeholder="Automatic")
                            vis_y_max = gr.Textbox(label="y max (blank: blockade_ratio = 1; else auto)", value="", placeholder="Automatic")
                    with gr.Group(visible=True) as vis_color_group:
                        vis_value = gr.Dropdown(choices=["label"], value="label", label="Color column (value)")
                    with gr.Group(visible=False) as vis_box_group:
                        vis_group_col = gr.Dropdown(choices=["label"], value="label", label="group_col")
                        vis_value_col = gr.Dropdown(choices=["blockade_ratio"], value="blockade_ratio", label="value_col")
                        with gr.Row():
                            vis_box_y_min = gr.Textbox(label="y min (blank: blockade_ratio = 0; else auto)", value="", placeholder="Automatic")
                            vis_box_y_max = gr.Textbox(label="y max (blank: blockade_ratio = 1; else auto)", value="", placeholder="Automatic")
                    vis_btn = gr.Button("Plot")
                    vis_plot = gr.Plot(label="Feature/filter visualization")
                    vis_plot_download = gr.File(label="Download plot files (PNG + editable-text PDF)", file_types=[".zip"], interactive=False)

        with gr.Tab("6 · Reduction"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### Analysis parameters")
                    gr.HTML(_parameter_guide("Reduction method", "dr_method"))
                    dr_method = gr.Dropdown(choices=["pca", "tsne", "umap"], value="pca", label="Reduction method", show_label=False, elem_classes=_param_classes("dr_method"))
                    with gr.Group(visible=True) as grp_dr_pca:
                        dr_random_state = gr.Slider(0, 9999, value=42, step=1, label="random_state")
                    with gr.Group(visible=False) as grp_dr_tsne:
                        tsne_perplexity = gr.Slider(2, 80, value=30, step=1, label="perplexity")
                        tsne_iter = gr.Slider(250, 3000, value=1000, step=50, label="n_iter")
                    with gr.Group(visible=False) as grp_dr_umap:
                        umap_neighbors = gr.Slider(2, 100, value=15, step=1, label="n_neighbors")
                        umap_min_dist = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="min_dist")
                    dr_cols = gr.Dropdown(choices=[], value=[], multiselect=True, label="Feature columns (same defaults as Train Model)")
                    dr_data = gr.Dropdown(choices=["filtered", "feature"], value="filtered", label="Data source")
                    with gr.Row():
                        dr_btn = gr.Button("Run reduction", scale=4)
                        dr_stop_btn = gr.Button("Stop", variant="stop", scale=1)
                    dr_df = gr.Dataframe(label="Reduction result")
                with gr.Column(scale=4):
                    gr.Markdown("### Plot parameters")
                    dr_plot_value = gr.Dropdown(choices=["label"], value="label", label="Color column")
                    dr_plot_x = gr.Dropdown(choices=["PC1"], value="PC1", label="Reduction x")
                    dr_plot_y = gr.Dropdown(choices=["PC2"], value="PC2", label="Reduction y")
                    with gr.Row():
                        dr_x_min = gr.Textbox(label="x min (blank: blockade_ratio = 0; else auto)", value="", placeholder="Automatic")
                        dr_x_max = gr.Textbox(label="x max (blank: blockade_ratio = 1; else auto)", value="", placeholder="Automatic")
                    with gr.Row():
                        dr_y_min = gr.Textbox(label="y min (blank: blockade_ratio = 0; else auto)", value="", placeholder="Automatic")
                        dr_y_max = gr.Textbox(label="y max (blank: blockade_ratio = 1; else auto)", value="", placeholder="Automatic")
                    dr_plot_btn = gr.Button("Plot 2D embedding")
                    dr_plot = gr.Plot(label="pl.plot_2d")
                    dr_plot_download = gr.File(label="Download plot files (PNG + editable-text PDF)", file_types=[".zip"], interactive=False)

        with gr.Tab("7 · Train Model"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### Analysis parameters")
                    model_type = gr.Dropdown(choices=["classic", "dl"], value="classic", label="Training type")
                    with gr.Group(visible=True) as grp_model_classic:
                        m_cols = gr.Dropdown(choices=[], value=[], multiselect=True, label="feature_cols (automatic defaults selected)")
                        m_cv = gr.Slider(2, 10, value=5, step=1, label="cv")
                        m_scoring = gr.Dropdown(choices=["accuracy", "f1", "recall"], value="accuracy", label="scoring")
                    with gr.Group(visible=False) as grp_model_dl:
                        gr.HTML(_parameter_guide("model_name", "model_name"))
                        dl_name = gr.Dropdown(
                            choices=["1D-CNN", "MAGJAM", "MAGJAM_d4"],
                            value="1D-CNN",
                            label="model_name",
                            show_label=False,
                            elem_classes=_param_classes("model_name"),
                        )
                        gr.HTML(_parameter_guide("interp_method", "interp_method"))
                        dl_interp_method = gr.Dropdown(
                            choices=["interp", "padding"],
                            value="interp",
                            label="interp_method",
                            show_label=False,
                            elem_classes=_param_classes("interp_method"),
                        )
                        dl_cv = gr.Slider(2, 10, value=5, step=1, label="cv")
                        dl_epoch = gr.Slider(1, 100, value=10, step=1, label="epoch")
                        dl_bs = gr.Slider(8, 256, value=64, step=8, label="batch_size")
                        dl_lr = gr.Number(value=1e-3, label="learning_rate")
                        gr.HTML(_parameter_guide("device", "device"))
                        dl_device = gr.Dropdown(choices=["cpu", "cuda"], value="cpu", label="device", show_label=False, elem_classes=_param_classes("device"))
                    with gr.Row():
                        model_train_btn = gr.Button("Train", scale=4)
                        model_stop_btn = gr.Button("Stop", variant="stop", scale=1)
                    model_result = gr.JSON(label="Training result")
                    model_table = gr.Dataframe(label='best_pkg["all_samples_feature_pred"]')
                with gr.Column(scale=4):
                    gr.Markdown("### Plot parameters")
                    model_name_for_plot = gr.Dropdown(choices=[], value=None, label="model_name")
                    cm_split = gr.Dropdown(choices=["train", "test"], value="test", label="cm split")
                    metric_name = gr.Dropdown(choices=["accuracy", "f1", "recall"], value="accuracy", label="metric")
                    fold_loss_type = gr.Dropdown(choices=["train", "val"], value="train", label="fold loss type")
                    model_plot_btn = gr.Button("Model visualization")
                    model_cm_plot = gr.Plot(label="pl.model_cm")
                    model_metric_plot = gr.Plot(label="pl.model_metric_bar")
                    model_fold_plot = gr.Plot(label="pl.plot_fold_loss")
                    model_plot_download = gr.File(label="Download model plots (PNG + editable-text PDF)", file_types=[".zip"], interactive=False)

        with gr.Tab("8 · Predict"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### Analysis parameters")
                    with gr.Accordion("Load a saved classical model", open=False):
                        gr.Markdown(
                            "Load a PoreMind `.pmmodel` bundle exported from Step 9. "
                            "Classical scikit-learn models are supported; deep-learning bundles are not included yet. "
                            "Only load model bundles from trusted sources."
                        )
                        saved_model_file = gr.File(
                            label="PoreMind classical model bundle",
                            file_count="single",
                            file_types=[".pmmodel"],
                        )
                        with gr.Row():
                            load_model_btn = gr.Button("Load model", variant="secondary", scale=1)
                            loaded_model_result = gr.JSON(label="Loaded model")
                    pred_files = gr.File(label="Upload unknown samples", file_count="multiple")
                    pred_model_name = gr.Dropdown(choices=[("Automatic (best available)", "")], value="", label="Model name (optional)")
                    with gr.Row():
                        pred_btn = gr.Button("classify_new_samples", scale=4)
                        pred_stop_btn = gr.Button("Stop", variant="stop", scale=1)
                    pred_df = gr.Dataframe(label="Prediction feature_df")
                with gr.Column(scale=4):
                    gr.Markdown("### Plot parameters")
                    pred_plot_kind = gr.Dropdown(choices=["plot_2d", "plot_3d", "event_current_label", "stacked_bar"], value="plot_2d", label="Visualization method")
                    with gr.Group(visible=True) as pred_axis_group:
                        pred_plot_x = gr.Dropdown(choices=["blockade_ratio"], value="blockade_ratio", label="Prediction x")
                        pred_plot_y = gr.Dropdown(choices=["duration_s"], value="duration_s", label="Prediction y")
                        with gr.Group(visible=False) as pred_plot_z_group:
                            pred_plot_z = gr.Dropdown(choices=["segment_std"], value="segment_std", label="Prediction z")
                        with gr.Row():
                            pred_x_min = gr.Textbox(label="x min (blank: blockade_ratio = 0; else auto)", value="", placeholder="Automatic")
                            pred_x_max = gr.Textbox(label="x max (blank: blockade_ratio = 1; else auto)", value="", placeholder="Automatic")
                        with gr.Row():
                            pred_y_min = gr.Textbox(label="y min (blank: blockade_ratio = 0; else auto)", value="", placeholder="Automatic")
                            pred_y_max = gr.Textbox(label="y max (blank: blockade_ratio = 1; else auto)", value="", placeholder="Automatic")
                    pred_label_col = gr.Dropdown(choices=["pred_label"], value="pred_label", label="label_col")
                    with gr.Group(visible=False) as pred_group_col_group:
                        pred_group_col = gr.Dropdown(choices=["trace_id", "sample_id"], value="trace_id", label="Prediction group column")
                    with gr.Group(visible=False) as pred_sample_group:
                        pred_sample_id = gr.Dropdown(choices=[], value=None, label="sample_id", elem_classes=["pm-tooltip-context-prediction_sample_id"])
                    pred_plot_btn = gr.Button("Plot")
                    pred_plot = gr.Plot(label="Prediction visualization")
                    pred_plot_download = gr.File(label="Download plot files (PNG + editable-text PDF)", file_types=[".zip"], interactive=False)

        with gr.Tab("9 · Export"):
            gr.HTML(_parameter_guide("Output directory", "output_dir"))
            with gr.Row():
                export_dir = gr.Textbox(
                    label="Output directory",
                    value=str(Path.home() / "ui_exports"),
                    show_label=False,
                    interactive=True,
                    elem_classes=_param_classes("output_dir"),
                    scale=5,
                )
                choose_export_dir_btn = gr.Button("Browse…", variant="secondary", scale=1)
            with gr.Row():
                export_btn = gr.Button("Export", scale=4)
                export_stop_btn = gr.Button("Stop", variant="stop", scale=1)
            export_result = gr.JSON(label="Export result")

        gr.HTML(
            '<footer class="pm-footer">'
            '<span>Developed by LuChenLab</span>'
            '<span aria-hidden="true">·</span>'
            '<a class="pm-footer-link" href="https://github.com/LuChenLab/PoreMind" '
            'target="_blank" rel="noopener noreferrer">GitHub</a>'
            '</footer>'
        )

        def on_denoise_method(method):
            return (
                gr.update(visible=method == "butterworth_filtfilt"),
                gr.update(visible=method == "moving_average"),
                gr.update(visible=method == "median"),
                gr.update(visible=method == "drift_corrected_moving_average"),
            )

        def on_detect_method(method):
            return (
                gr.update(visible=method == "threshold"),
                gr.update(visible=method == "zscore_threshold"),
                gr.update(visible=method == "cusum"),
                gr.update(visible=method == "pelt"),
                gr.update(visible=method == "hmm"),
            )

        def on_baseline_method(method):
            """rolling_quantile alone needs a local baseline window."""
            return gr.update(visible=method == "rolling_quantile")

        def on_current_range(enabled):
            return gr.update(visible=bool(enabled))

        def sync_value(value):
            """Copy a user-edited value to the matching control in the other stage."""
            return gr.update(value=value)

        def sync_method_to_events(method):
            return gr.update(value=method), *on_detect_method(method)

        def sync_method_to_pre_events(method):
            return gr.update(value=method), *on_detect_method(method)

        def sync_baseline_to_events(method):
            return gr.update(value=method), on_baseline_method(method)

        def sync_baseline_to_pre_events(method):
            return gr.update(value=method), on_baseline_method(method)

        def quantile_for_direction(direction):
            """Return the direction-aware quantile in the UI's percent units."""
            return 5 if direction == "up" else 95

        def sync_direction_to_events(direction):
            q = quantile_for_direction(direction)
            return gr.update(value=direction), gr.update(value=q), gr.update(value=q)

        def sync_direction_to_pre_events(direction):
            q = quantile_for_direction(direction)
            return gr.update(value=direction), gr.update(value=q), gr.update(value=q)

        def sync_range_to_events(enabled):
            return gr.update(value=enabled), on_current_range(enabled)

        def sync_range_to_pre_events(enabled):
            return gr.update(value=enabled), on_current_range(enabled)

        def on_filter_method(method):
            return (
                gr.update(visible=method == "blockade_gmm"),
                gr.update(visible=method == "isolation_forest"),
                gr.update(visible=method == "lof"),
                gr.update(visible=method == "knn_background"),
            )

        def on_dr_method(method):
            return (
                gr.update(visible=method == "pca"),
                gr.update(visible=method == "tsne"),
                gr.update(visible=method == "umap"),
            )

        def on_reduction_plot_axes(method, data_name):
            source_df = _ui_dataframe(data_name)
            numeric_columns = list(source_df.select_dtypes(include="number").columns) if source_df is not None else []
            prefix = {"pca": "PC", "tsne": "TSNE", "umap": "UMAP"}.get(method, "PC")
            axis_columns = [f"{prefix}1", f"{prefix}2"]
            choices = list(dict.fromkeys([*numeric_columns, *axis_columns]))
            return (
                gr.update(choices=choices, value=axis_columns[0]),
                gr.update(choices=choices, value=axis_columns[1]),
            )

        def on_model_type(method):
            return gr.update(visible=method == "classic"), gr.update(visible=method == "dl")

        def on_feature_plot_method(method):
            show_axes = method in {"plot_2d", "plot_3d"}
            return (
                gr.update(visible=show_axes),
                gr.update(visible=method == "plot_3d"),
                gr.update(visible=show_axes),
                gr.update(visible=method == "box_significance"),
            )

        def on_predict_plot_kind(kind):
            return (
                gr.update(visible=kind in {"plot_2d", "plot_3d"}),
                gr.update(visible=kind == "plot_3d"),
                gr.update(visible=kind == "stacked_bar"),
                gr.update(visible=kind == "event_current_label"),
            )

        def _ui_dataframe(source: str):
            analysis = ctl.session.analysis
            if analysis is None:
                return None
            if source == "feature":
                return analysis.feature_df
            if source == "filtered":
                return analysis.filtered_df if analysis.filtered_df is not None else analysis.feature_df
            return None

        def _choice_value(options, current, preferred=None):
            if current in options:
                return current
            if preferred in options:
                return preferred
            return options[0] if options else None

        def _multi_choice_value(options, current, defaults):
            current = [item for item in (current or []) if item in options]
            if current:
                return current
            return [item for item in defaults if item in options]


        def refresh_plot_and_feature_columns(
            vis_source,
            dr_source,
            cur_x,
            cur_y,
            cur_z,
            cur_color,
            cur_group,
            cur_value_col,
            cur_dr_cols,
            cur_model_cols,
        ):
            vis_df = _ui_dataframe(vis_source)
            if vis_df is None:
                vis_numeric = ["blockade_ratio", "duration_s", "segment_std"]
                vis_columns = ["label", "sample_id", *vis_numeric]
            else:
                vis_numeric = list(vis_df.select_dtypes(include="number").columns)
                vis_columns = list(vis_df.columns)
            color_columns = vis_columns
            group_columns = vis_columns
            x_value = _choice_value(vis_numeric, cur_x, "blockade_ratio")
            y_value = _choice_value(vis_numeric, cur_y, "duration_s")
            z_value = _choice_value(vis_numeric, cur_z, "segment_std")
            color_value = _choice_value(color_columns, cur_color, "label")
            group_value = _choice_value(group_columns, cur_group, "label")
            value_columns = vis_numeric
            plot_value = _choice_value(value_columns, cur_value_col, "blockade_ratio")

            dr_df = _ui_dataframe(dr_source)
            if dr_df is None:
                dr_available = []
                dr_defaults = []
            else:
                dr_available = select_feature_columns(dr_df)
                dr_defaults = _default_feature_columns(dr_df)
            model_df = _ui_dataframe("feature")
            if model_df is None:
                model_available = []
                model_defaults = []
            else:
                model_available = select_feature_columns(model_df)
                model_defaults = _default_feature_columns(model_df)

            return (
                gr.update(choices=vis_numeric, value=x_value),
                gr.update(choices=vis_numeric, value=y_value),
                gr.update(choices=vis_numeric, value=z_value),
                gr.update(choices=color_columns, value=color_value),
                gr.update(choices=group_columns, value=group_value),
                gr.update(choices=value_columns, value=plot_value),
                gr.update(choices=dr_available, value=_multi_choice_value(dr_available, cur_dr_cols, dr_defaults)),
                gr.update(choices=model_available, value=_multi_choice_value(model_available, cur_model_cols, model_defaults)),
            )

        def _available_model_names():
            analysis = ctl.session.analysis
            if analysis is None:
                return []
            names = set(analysis.model_cv_results)
            names.update(analysis.DL_model_packages)
            if analysis.best_model_package:
                if analysis.best_model_package.get("best_model"):
                    names.add(str(analysis.best_model_package["best_model"]))
                names.update(str(name) for name in analysis.best_model_package.get("models", {}))
            if analysis.DL_model_package and analysis.DL_model_package.get("model_name"):
                names.add(str(analysis.DL_model_package["model_name"]))
            return sorted(names)

        def refresh_model_choices(current_plot_name=None, current_prediction_name=""):
            names = _available_model_names()
            plot_default = _choice_value(names, current_plot_name, "Random Forest")
            prediction_choices = [("Automatic (best available)", "")] + [(name, name) for name in names]
            prediction_values = [value for _, value in prediction_choices]
            prediction_default = current_prediction_name if current_prediction_name in prediction_values else ""
            return (
                gr.update(choices=names, value=plot_default),
                gr.update(choices=prediction_choices, value=prediction_default),
            )

        def refresh_prediction_columns(current_label="pred_label", current_sample=None, current_x=None, current_y=None, current_z=None, current_group=None):
            pred_df = ctl.session.outputs.get("pred_df")
            if pred_df is None or not len(pred_df.columns):
                columns = ["pred_label"]
                numeric_columns = ["blockade_ratio", "duration_s", "segment_std"]
                sample_ids = []
            else:
                columns = list(pred_df.columns)
                numeric_columns = list(pred_df.select_dtypes(include="number").columns)
                sample_column = "trace_id" if "trace_id" in pred_df.columns else "sample_id"
                sample_ids = sorted(str(value) for value in pred_df[sample_column].dropna().unique()) if sample_column in pred_df.columns else []
            label_default = next((column for column in columns if column == "pred_label" or column.startswith("pred_label_")), columns[0] if columns else None)
            group_default = "trace_id" if "trace_id" in columns else ("sample_id" if "sample_id" in columns else (columns[0] if columns else None))
            x_default = "blockade_ratio" if "blockade_ratio" in numeric_columns else (numeric_columns[0] if numeric_columns else None)
            y_default = "duration_s" if "duration_s" in numeric_columns else (numeric_columns[1] if len(numeric_columns) > 1 else x_default)
            z_default = "segment_std" if "segment_std" in numeric_columns else (numeric_columns[2] if len(numeric_columns) > 2 else y_default)
            return (
                gr.update(choices=columns, value=_choice_value(columns, current_label, label_default)),
                gr.update(choices=sample_ids, value=_choice_value(sample_ids, current_sample, sample_ids[0] if sample_ids else None)),
                gr.update(choices=numeric_columns, value=_choice_value(numeric_columns, current_x, x_default)),
                gr.update(choices=numeric_columns, value=_choice_value(numeric_columns, current_y, y_default)),
                gr.update(choices=numeric_columns, value=_choice_value(numeric_columns, current_z, z_default)),
                gr.update(choices=columns, value=_choice_value(columns, current_group, group_default)),
            )

        column_selector_inputs = [
            vis_target,
            dr_data,
            vis_x,
            vis_y,
            vis_z,
            vis_value,
            vis_group_col,
            vis_value_col,
            dr_cols,
            m_cols,
        ]
        column_selector_outputs = [
            vis_x,
            vis_y,
            vis_z,
            vis_value,
            vis_group_col,
            vis_value_col,
            dr_cols,
            m_cols,
        ]

        def _detect_params(method, sigma_k, min_dur, nmethod, z, z_min, z_nm, c_d, c_t, c_min, c_nm, p_m, p_p, p_s, p_min, p_nm, h_n, h_cov_t, h_iter, h_min):
            if method == "threshold":
                return {"sigma_k": float(sigma_k), "min_duration_s": float(min_dur), "noise_method": nmethod}
            if method == "zscore_threshold":
                return {"z": float(z), "min_duration_s": float(z_min), "noise_method": z_nm}
            if method == "cusum":
                return {"drift": float(c_d), "threshold": float(c_t), "min_duration_s": float(c_min), "noise_method": c_nm}
            if method == "pelt":
                return {"model": p_m, "penalty": float(p_p), "sigma_k": float(p_s), "min_duration_s": float(p_min), "noise_method": p_nm}
            return {"n_components": int(h_n), "covariance_type": h_cov_t, "n_iter": int(h_iter), "min_duration_s": float(h_min)}

        def do_load(files, reader_name, annotation_table, progress=gr.Progress(track_tqdm=True)):
            progress(0.05, desc="Preparing sample inputs")
            sample_paths = _to_file_map(files)
            if not sample_paths:
                raise ValueError("Upload at least one ABF or CSV file before loading samples.")
            sample_to_group = _parse_sample_annotations(annotation_table, sample_paths)
            progress(0.35, desc="Loading traces and metadata")
            out = ctl.load_samples(sample_paths=sample_paths, sample_to_group=sample_to_group, reader=reader_name)
            progress(1.0, desc="Samples loaded")
            trace_ids = list(out["summary"]["trace_ids"])
            first_trace = trace_ids[0] if trace_ids else None
            return (
                out["summary"],
                out["sample_df"],
                gr.update(choices=trace_ids, value=first_trace),
                gr.update(choices=trace_ids, value=first_trace),
                gr.update(choices=trace_ids, value=first_trace),
                gr.update(choices=trace_ids, value=first_trace),
                gr.update(choices=trace_ids, value=[]),
                gr.update(choices=trace_ids, value=first_trace),
                gr.update(value=None),  # feature_df
                gr.update(value=None),  # filtered_df
                None,  # feature/filter plot
                None,  # feature/filter plot download
                gr.update(choices=[], value=[]),  # reduction feature columns
                gr.update(value=None),  # reduction result
                None,  # reduction plot
                None,  # reduction plot download
                gr.update(choices=[], value=[]),  # model feature columns
                None,  # model result
                gr.update(value=None),  # model prediction table
                gr.update(choices=[], value=None),  # model name for plots
                None,  # confusion matrix plot
                None,  # model metric plot
                None,  # fold loss plot
                None,  # model plot download
                gr.update(choices=[("Automatic (best available)", "")], value=""),
                gr.update(value=None),  # prediction table
                None,  # prediction plot
                None,  # prediction plot download
            )

        def do_denoise(method, n, wn, w_ma, w_median, dwin, swin, progress=gr.Progress(track_tqdm=True)):
            progress(0.05, desc="Preparing denoising parameters")
            if method == "butterworth_filtfilt":
                out = ctl.run_denoise(method=method, filtfilt_N=int(n), filtfilt_Wn=float(wn))
                progress(1.0, desc="Denoising complete")
                return out
            if method == "moving_average":
                out = ctl.run_denoise(method=method, window=int(w_ma))
                progress(1.0, desc="Denoising complete")
                return out
            if method == "median":
                out = ctl.run_denoise(method=method, window=int(w_median))
                progress(1.0, desc="Denoising complete")
                return out
            if method == "drift_corrected_moving_average":
                out = ctl.run_denoise(method=method, drift_window=int(dwin), smooth_window=int(swin))
                progress(1.0, desc="Denoising complete")
                return out
            out = ctl.run_denoise(method=method)
            progress(1.0, desc="Denoising complete")
            return out

        def draw_current(sid, current, start_ms, end_ms):
            fig = ctl.plot_current(sample_id=(sid or None), current=current, start_ms=float(start_ms), end_ms=float(end_ms))
            return fig, _save_plot_bundle([("current_trace", fig)])

        def do_preview_signal(sid):
            analysis = ctl._require_analysis()
            sid = sid or next(iter(analysis.traces.keys()))
            return analysis.preview_signal(sid, start_s=0.0, end_s=0.002).head()

        def run_detect_simple(
            method,
            direction,
            bmethod,
            bwin,
            bq,
            pmerge,
            pmerge_gap,
            psid,
            pcur,
            pstart,
            pend,
            pexclude,
            p_min,
            p_max,
            *det_values,
            progress=gr.Progress(track_tqdm=True),
        ):
            progress(0.05, desc="Preparing simple event detection")
            if not psid:
                raise ValueError("Choose a sample_id for the Pre-Events check.")
            if float(pend) <= float(pstart):
                raise ValueError("end_ms must be greater than start_ms for the Pre-Events check.")
            params = _detect_params(method, *det_values)
            if bmethod == "rolling_quantile":
                baseline_params = {"window": int(bwin), "q": float(bq) / 100.0}
            elif bmethod == "global_quantile":
                baseline_params = {"q": float(bq) / 100.0}
            else:
                baseline_params = None
            stats_params = {"min": _none_if_blank(p_min), "max": _none_if_blank(p_max)} if pexclude else None
            with _route_tqdm_to_gradio(progress):
                out = ctl.run_detect(
                    stage="preview",
                    sample_id=(psid or None),
                    current=pcur,
                    start_ms=float(pstart),
                    end_ms=float(pend),
                    detect_method=method,
                    detect_params=params,
                    baseline_method=bmethod,
                    baseline_params=baseline_params,
                    detect_direction=direction,
                    merge_event=bool(pmerge),
                    merge_event_params={"merge_gap_ms": float(pmerge_gap)},
                    exclude_current=bool(pexclude),
                    exclude_current_params=stats_params,
                )
            progress(1.0, desc="Simple event detection complete")
            return out

        def run_detect_global(
            method,
            direction,
            gb_method,
            gb_window,
            gb_q,
            g_merge,
            g_gap,
            g_exc,
            g_min,
            g_max,
            *det_values,
            progress=gr.Progress(track_tqdm=True),
        ):
            progress(0.05, desc="Preparing full event detection")
            params = _detect_params(method, *det_values)
            if gb_method == "rolling_quantile":
                baseline_params = {"window": int(gb_window), "q": float(gb_q) / 100.0}
            elif gb_method == "global_quantile":
                baseline_params = {"q": float(gb_q) / 100.0}
            else:
                baseline_params = None
            with _route_tqdm_to_gradio(progress):
                out = ctl.run_detect(
                    stage="global",
                    detect_method=method,
                    detect_params=params,
                    baseline_method=gb_method,
                    baseline_params=baseline_params,
                    detect_direction=direction,
                    merge_event=bool(g_merge),
                    merge_event_params={"merge_gap_ms": float(g_gap)},
                    exclude_current=bool(g_exc),
                    exclude_current_params={"min": _none_if_blank(g_min), "max": _none_if_blank(g_max)},
                )
            progress(1.0, desc="Full event detection complete")
            return out

        def plot_simple_events(sid, current, s_evt, e_evt):
            if not sid:
                raise ValueError("Choose a sample_id before plotting simple events.")
            fig = ctl.plot_event_current_simple(sample_id=sid, current=current, start_event=int(s_evt), end_event=int(e_evt))
            return fig, ctl.simple_events_df(sid), _save_plot_bundle([("pre_events", fig)])

        def plot_global_events(sid, current, s_evt, e_evt):
            if not sid:
                raise ValueError("Choose a sample_id before plotting detected events.")
            fig = ctl.plot_event_current(sample_id=sid, current=current, start_event=int(s_evt), end_event=int(e_evt))
            return fig, ctl.events_df(sid), _save_plot_bundle([("events", fig)])

        def custom_shape_features(x):
            import numpy as np

            x = np.asarray(x, dtype=float)
            return {"ptp": float(np.max(x) - np.min(x)), "abs_mean": float(np.mean(np.abs(x)))}

        def do_extract(max_e, use_custom_flag, progress=gr.Progress(track_tqdm=True)):
            progress(0.05, desc="Preparing feature extraction")
            cfn = {"custom": custom_shape_features} if use_custom_flag else None
            with _route_tqdm_to_gradio(progress):
                out = ctl.extract_features(max_event_per_sample=int(max_e), custom_feature_fns=cfn)
            progress(1.0, desc="Feature extraction complete")
            return out

        def do_filter(method, n_comp, prior_mean, if_c, lof_c, knn_bg_ids, knn_k, knn_nm, knn_bg_ratio, knn_features, knn_bg_lim, lo, hi, progress=gr.Progress(track_tqdm=True)):
            progress(0.05, desc="Preparing filter parameters")
            if method == "blockade_gmm":
                params = {"n_components": int(n_comp), "prior_mean": _none_if_blank(prior_mean)}
            elif method == "isolation_forest":
                params = {"contamination": float(if_c)}
            elif method == "lof":
                params = {"contamination": float(lof_c)}
            else:
                if isinstance(knn_bg_ids, (list, tuple)):
                    ids = [str(sample_id).strip() for sample_id in knn_bg_ids if str(sample_id).strip()]
                else:
                    ids = [s.strip() for s in str(knn_bg_ids or "").split(",") if s.strip()]
                feats = [s.strip() for s in (knn_features or "").split(",") if s.strip()] or None
                bg_lim = _parse_pair(knn_bg_lim)
                params = {
                    "background_sample_ids": ids,
                    "k": int(knn_k),
                    "n_noise_match": int(knn_nm),
                    "background_ratio": float(knn_bg_ratio),
                }
                if feats is not None:
                    params["feature_cols"] = feats
                if bg_lim is not None:
                    params["background_blockage_lim"] = bg_lim
            progress(0.35, desc="Filtering event feature table")
            out = ctl.filter_events(method=method, parameters=params, blockage_lim=(float(lo), float(hi)))
            progress(1.0, desc="Filtering complete")
            return out

        def draw_feature_filter(target, method, x, y, z, value, gcol, vcol, x_min, x_max, y_min, y_max, box_y_min, box_y_max):
            data = "full" if target == "feature" else "filtered"
            if method == "plot_2d":
                xlim = _plot_axis_limit(x, x_min, x_max, "x-axis")
                ylim = _plot_axis_limit(y, y_min, y_max, "y-axis")
                fig = ctl.plot_2d(
                    data=data,
                    x=x,
                    y=y,
                    value=value,
                    xlim=xlim,
                    ylim=ylim,
                    y_log2=(y != "blockade_ratio"),
                )
            elif method == "plot_3d":
                xlim = _plot_axis_limit(x, x_min, x_max, "x-axis")
                ylim = _plot_axis_limit(y, y_min, y_max, "y-axis")
                zlim = _plot_axis_limit(z, None, None, "z-axis")
                fig = ctl.plot_3d(
                    data=data,
                    x=x,
                    y=y,
                    z=z,
                    value=value,
                    xlim=xlim,
                    ylim=ylim,
                    zlim=zlim,
                    y_log2=(y != "blockade_ratio"),
                )
            else:
                box_ylim = _plot_axis_limit(vcol, box_y_min, box_y_max, "y-axis")
                fig = ctl.box_significance(data=data, group_col=gcol, value_col=vcol, ylim=box_ylim)
            return fig, _save_plot_bundle([("features_filter", fig)])

        def do_dr(method, selected_cols, data_name, random_state, perplexity, n_iter, nn, min_dist, cur_x, cur_y, progress=gr.Progress(track_tqdm=True)):
            progress(0.05, desc="Preparing reduction parameters")
            cols = list(selected_cols or [])
            if not cols:
                cols = _default_feature_columns(_ui_dataframe(data_name)) or None
            kwargs = {"random_state": int(random_state)}
            if method == "tsne":
                kwargs.update({"perplexity": float(perplexity), "n_iter": int(n_iter)})
            if method == "umap":
                kwargs.update({"n_neighbors": int(nn), "min_dist": float(min_dist)})
            progress(0.35, desc=f"Running {method.upper()} reduction")
            out = ctl.do_dimensionality_reduction(method=method, feature_cols=cols, data=data_name, **kwargs)
            progress(1.0, desc="Reduction complete")
            columns = list(out.columns)
            color_value = _choice_value(columns, "label", "label")
            numeric_columns = list(out.select_dtypes(include="number").columns)
            prefix = {"pca": "PC", "tsne": "TSNE", "umap": "UMAP"}[method]
            axis_columns = [f"{prefix}1", f"{prefix}2"]
            return (
                out,
                gr.update(choices=columns, value=color_value),
                gr.update(choices=numeric_columns, value=_choice_value(numeric_columns, cur_x, axis_columns[0])),
                gr.update(choices=numeric_columns, value=_choice_value(numeric_columns, cur_y, axis_columns[1])),
            )

        def draw_dr(data_name, value, x, y, x_min, x_max, y_min, y_max):
            xlim = _plot_axis_limit(x, x_min, x_max, "x-axis")
            ylim = _plot_axis_limit(y, y_min, y_max, "y-axis")
            plot_data = "filtered" if data_name == "filtered" else "full"
            fig = ctl.plot_2d(data=plot_data, x=x, y=y, xlim=xlim, ylim=ylim, y_log2=False, value=value)
            return fig, _save_plot_bundle([("dimensionality_reduction", fig)])

        def do_train(model_kind, cols, cv, scoring, dl_model_name, dl_interp_method, dl_cv_v, dl_epoch_v, dl_bs_v, dl_lr_v, dl_device_v, progress=gr.Progress(track_tqdm=True)):
            progress(0.05, desc="Preparing model training")
            with _route_tqdm_to_gradio(progress):
                if model_kind == "classic":
                    fcols = list(cols or []) or None
                    progress(0.20, desc="Training classical models")
                    out = ctl.train_model(feature_cols=fcols, cv=int(cv), scoring=scoring), ctl.model_prediction_table()
                    progress(1.0, desc="Training complete")
                    return out
                out = ctl.train_dl_model(
                    model_name=dl_model_name,
                    interp_method=dl_interp_method,
                    cv=int(dl_cv_v),
                    epoch=int(dl_epoch_v),
                    batch_size=int(dl_bs_v),
                    learning_rate=float(dl_lr_v),
                    device=dl_device_v,
                )
            progress(1.0, desc="Training complete")
            return out, ctl.model_prediction_table()

        def draw_model(model_name, split, metric, loss_type):
            cm = ctl.plot_model_cm(model_name=model_name, split=split)
            bar = ctl.plot_model_metric_bar(metric=metric, split=split)
            try:
                loss = ctl.plot_fold_loss(model_name=model_name, type=loss_type)
            except Exception:
                loss = None
            return cm, bar, loss, _save_plot_bundle([
                ("model_confusion_matrix", cm),
                ("model_metric_comparison", bar),
                ("model_fold_loss", loss),
            ])

        def do_predict(files, model_name, progress=gr.Progress(track_tqdm=True)):
            progress(0.05, desc="Preparing unknown samples")
            progress(0.35, desc="Classifying new samples")
            out = ctl.predict_new(new_sample_paths=_to_file_map(files), model=(model_name or None))
            progress(1.0, desc="Prediction complete")
            return out

        def do_load_model(file_value):
            if not file_value:
                return {"loaded": False, "error": "Choose a .pmmodel bundle first."}
            bundle_path = file_value[0] if isinstance(file_value, list) else file_value
            try:
                return ctl.load_classical_model_bundle(bundle_path)
            except Exception as exc:
                return {"loaded": False, "error": str(exc)}

        def draw_predict(kind, sid, label_col, x, y, z, group_col, x_min, x_max, y_min, y_max):
            prediction_analysis = ctl.session.outputs.get("prediction_analysis")
            if prediction_analysis is None:
                raise ValueError("Run prediction before plotting unknown samples.")
            plotter = prediction_analysis.pl
            if kind == "plot_2d":
                xlim = _plot_axis_limit(x, x_min, x_max, "x-axis")
                ylim = _plot_axis_limit(y, y_min, y_max, "y-axis")
                ax = plotter.plot_2d(
                    data="full",
                    x=x,
                    y=y,
                    value=label_col,
                    xlim=xlim,
                    ylim=ylim,
                    y_log2=(y != "blockade_ratio"),
                )
            elif kind == "plot_3d":
                xlim = _plot_axis_limit(x, x_min, x_max, "x-axis")
                ylim = _plot_axis_limit(y, y_min, y_max, "y-axis")
                zlim = _plot_axis_limit(z, None, None, "z-axis")
                ax = plotter.plot_3d(
                    data="full",
                    x=x,
                    y=y,
                    z=z,
                    value=label_col,
                    xlim=xlim,
                    ylim=ylim,
                    zlim=zlim,
                    y_log2=(y != "blockade_ratio"),
                )
            elif kind == "event_current_label":
                ax = plotter.event_current_label(sample_id=(sid or None), lable_col=label_col)
            else:
                ax = plotter.stacked_bar(group_col=group_col, value_col=label_col, data="full")
            fig = ax.figure
            return fig, _save_plot_bundle([("prediction", fig)])

        def do_export(path_text, progress=gr.Progress(track_tqdm=True)):
            progress(0.05, desc="Preparing export folder")
            out_dir = Path(path_text or (Path.home() / "ui_exports")).expanduser()
            progress(0.35, desc="Writing analysis tables")
            tables = ctl.export_tables(out_dir)
            progress(0.65, desc="Writing reproducibility files")
            params_json = ctl.export_params_json(out_dir / "params_snapshot.json")
            script = ctl.export_analysis_script(out_dir / "reproduce_analysis.py")
            model_bundle = None
            analysis = ctl.session.analysis
            if analysis is not None and analysis.best_model_package is not None:
                model_bundle = ctl.save_classical_model_bundle(out_dir / "classical_model.pmmodel")
            progress(1.0, desc="Export complete")
            return {
                "tables": tables,
                "params_json": params_json,
                "script": script,
                "classical_model_bundle": model_bundle,
            }

        def choose_export_directory(current_path):
            chosen = _choose_windows_folder(current_path or (Path.home() / "ui_exports"))
            return chosen or current_path

        detect_param_inputs = [
            d_sigma_k,
            d_min_duration,
            d_noise_method,
            z_thr,
            z_min_duration,
            z_noise_method,
            c_drift,
            c_threshold,
            c_min_duration,
            c_noise_method,
            p_model,
            p_penalty,
            p_sigma_k,
            p_min_duration,
            p_noise_method,
            h_n_components,
            h_cov,
            h_n_iter,
            h_min_duration,
        ]
        event_detect_param_inputs = [
            event_d_sigma_k,
            event_d_min_duration,
            event_d_noise_method,
            event_z_thr,
            event_z_min_duration,
            event_z_noise_method,
            event_c_drift,
            event_c_threshold,
            event_c_min_duration,
            event_c_noise_method,
            event_p_model,
            event_p_penalty,
            event_p_sigma_k,
            event_p_min_duration,
            event_p_noise_method,
            event_h_n_components,
            event_h_cov,
            event_h_n_iter,
            event_h_min_duration,
        ]

        denoise_method.change(on_denoise_method, inputs=[denoise_method], outputs=[grp_bw, grp_ma, grp_median, grp_drift])
        detect_method.change(on_detect_method, inputs=[detect_method], outputs=[grp_det_threshold, grp_det_z, grp_det_cusum, grp_det_pelt, grp_det_hmm])
        event_detect_method.change(on_detect_method, inputs=[event_detect_method], outputs=[event_grp_det_threshold, event_grp_det_z, event_grp_det_cusum, event_grp_det_pelt, event_grp_det_hmm])
        detect_method.input(
            sync_method_to_events,
            inputs=[detect_method],
            outputs=[event_detect_method, event_grp_det_threshold, event_grp_det_z, event_grp_det_cusum, event_grp_det_pelt, event_grp_det_hmm],
        )
        event_detect_method.input(
            sync_method_to_pre_events,
            inputs=[event_detect_method],
            outputs=[detect_method, grp_det_threshold, grp_det_z, grp_det_cusum, grp_det_pelt, grp_det_hmm],
        )
        baseline_method.change(on_baseline_method, inputs=[baseline_method], outputs=[preview_baseline_window_group])
        global_baseline_method.change(on_baseline_method, inputs=[global_baseline_method], outputs=[global_baseline_window_group])
        baseline_method.input(
            sync_baseline_to_events,
            inputs=[baseline_method],
            outputs=[global_baseline_method, global_baseline_window_group],
        )
        global_baseline_method.input(
            sync_baseline_to_pre_events,
            inputs=[global_baseline_method],
            outputs=[baseline_method, preview_baseline_window_group],
        )
        detect_direction.input(
            sync_direction_to_events,
            inputs=[detect_direction],
            outputs=[event_detect_direction, b_q, global_b_q],
        )
        event_detect_direction.input(
            sync_direction_to_pre_events,
            inputs=[event_detect_direction],
            outputs=[detect_direction, global_b_q, b_q],
        )
        preview_exclude.change(on_current_range, inputs=[preview_exclude], outputs=[preview_current_range_group])
        global_exclude.change(on_current_range, inputs=[global_exclude], outputs=[global_current_range_group])
        preview_exclude.input(
            sync_range_to_events,
            inputs=[preview_exclude],
            outputs=[global_exclude, global_current_range_group],
        )
        global_exclude.input(
            sync_range_to_pre_events,
            inputs=[global_exclude],
            outputs=[preview_exclude, preview_current_range_group],
        )

        for pre_component, event_component in [
            (b_window, global_b_window),
            (b_q, global_b_q),
            (preview_merge, global_merge),
            (preview_merge_gap, global_merge_gap),
            (preview_ex_min, global_ex_min),
            (preview_ex_max, global_ex_max),
            *zip(detect_param_inputs, event_detect_param_inputs),
        ]:
            pre_component.input(sync_value, inputs=[pre_component], outputs=[event_component])
            event_component.input(sync_value, inputs=[event_component], outputs=[pre_component])
        filter_method.change(on_filter_method, inputs=[filter_method], outputs=[grp_filter_gmm, grp_filter_if, grp_filter_lof])
        dr_method.change(on_dr_method, inputs=[dr_method], outputs=[grp_dr_pca, grp_dr_tsne, grp_dr_umap])
        dr_method.change(on_reduction_plot_axes, inputs=[dr_method, dr_data], outputs=[dr_plot_x, dr_plot_y])
        model_type.change(on_model_type, inputs=[model_type], outputs=[grp_model_classic, grp_model_dl])
        vis_method.change(
            on_feature_plot_method,
            inputs=[vis_method],
            outputs=[vis_axis_group, vis_z_group, vis_color_group, vis_box_group],
        )
        pred_plot_kind.change(
            on_predict_plot_kind,
            inputs=[pred_plot_kind],
            outputs=[pred_axis_group, pred_plot_z_group, pred_group_col_group, pred_sample_group],
        )
        vis_target.change(refresh_plot_and_feature_columns, inputs=column_selector_inputs, outputs=column_selector_outputs)
        dr_data.change(refresh_plot_and_feature_columns, inputs=column_selector_inputs, outputs=column_selector_outputs)
        dr_data.change(on_reduction_plot_axes, inputs=[dr_method, dr_data], outputs=[dr_plot_x, dr_plot_y])

        file_input.change(
            _sample_annotation_rows,
            inputs=[file_input],
            outputs=[sample_annotation_table],
            queue=False,
        )

        load_event = load_btn.click(
            do_load,
            inputs=[file_input, reader, sample_annotation_table],
            outputs=[
                load_summary,
                sample_df,
                denoise_sample,
                preview_sample,
                preview_plot_sample,
                global_plot_sample,
                f_knn_bg_ids,
                pred_sample_id,
                feat_df,
                filtered_df,
                vis_plot,
                vis_plot_download,
                dr_cols,
                dr_df,
                dr_plot,
                dr_plot_download,
                m_cols,
                model_result,
                model_table,
                model_name_for_plot,
                model_cm_plot,
                model_metric_plot,
                model_fold_plot,
                model_plot_download,
                pred_model_name,
                pred_df,
                pred_plot,
                pred_plot_download,
            ],
            show_progress="full",
            show_progress_on=[load_summary, sample_df],
        )
        load_stop_btn.click(fn=None, cancels=[load_event], queue=False)

        denoise_event = denoise_run_btn.click(
            do_denoise,
            inputs=[denoise_method, bw_n, bw_wn, ma_window, median_window, drift_window, smooth_window],
            outputs=[denoise_result],
            show_progress="full",
            show_progress_on=[denoise_result],
        )
        denoise_stop_btn.click(fn=None, cancels=[denoise_event], queue=False)
        denoise_plot_btn.click(
            draw_current,
            inputs=[denoise_sample, denoise_current, denoise_start, denoise_end],
            outputs=[denoise_plot, denoise_plot_download],
        )
        preview_btn.click(do_preview_signal, inputs=[denoise_sample], outputs=[preview_table])

        preview_event = preview_run_btn.click(
            run_detect_simple,
            inputs=[detect_method, detect_direction, baseline_method, b_window, b_q, preview_merge, preview_merge_gap, preview_sample, preview_current, preview_start, preview_end, preview_exclude, preview_ex_min, preview_ex_max, *detect_param_inputs],
            outputs=[preview_result],
            show_progress="full",
            show_progress_on=[preview_result],
        )
        preview_stop_btn.click(fn=None, cancels=[preview_event], queue=False)
        global_event = global_run_btn.click(
            run_detect_global,
            inputs=[event_detect_method, event_detect_direction, global_baseline_method, global_b_window, global_b_q, global_merge, global_merge_gap, global_exclude, global_ex_min, global_ex_max, *event_detect_param_inputs],
            outputs=[global_result],
            show_progress="full",
            show_progress_on=[global_result],
        )
        global_stop_btn.click(fn=None, cancels=[global_event], queue=False)

        preview_plot_btn.click(
            plot_simple_events,
            inputs=[preview_plot_sample, preview_plot_current, preview_plot_start_event, preview_plot_end_event],
            outputs=[preview_plot, preview_event_table, preview_plot_download],
        )
        global_plot_btn.click(
            plot_global_events,
            inputs=[global_plot_sample, global_plot_current, global_plot_start_event, global_plot_end_event],
            outputs=[global_plot, global_event_table, global_plot_download],
        )

        extract_event = extract_btn.click(
            do_extract,
            inputs=[max_events, use_custom],
            outputs=[feat_df],
            show_progress="full",
            show_progress_on=[feat_df],
        )
        extract_event.then(
            refresh_plot_and_feature_columns,
            inputs=column_selector_inputs,
            outputs=column_selector_outputs,
            queue=False,
        )
        extract_stop_btn.click(fn=None, cancels=[extract_event], queue=False)
        filter_event = filter_btn.click(
            do_filter,
            inputs=[filter_method, f_n_components, f_prior_mean, f_if_contam, f_lof_contam, f_knn_bg_ids, f_knn_k, f_knn_n_match, f_knn_bg_ratio, f_knn_features, f_knn_bg_lim, block_lo, block_hi],
            outputs=[filtered_df],
            show_progress="full",
            show_progress_on=[filtered_df],
        )
        filter_event.then(
            refresh_plot_and_feature_columns,
            inputs=column_selector_inputs,
            outputs=column_selector_outputs,
            queue=False,
        )
        filter_stop_btn.click(fn=None, cancels=[filter_event], queue=False)
        vis_btn.click(
            draw_feature_filter,
            inputs=[vis_target, vis_method, vis_x, vis_y, vis_z, vis_value, vis_group_col, vis_value_col, vis_x_min, vis_x_max, vis_y_min, vis_y_max, vis_box_y_min, vis_box_y_max],
            outputs=[vis_plot, vis_plot_download],
        )

        dr_event = dr_btn.click(
            do_dr,
            inputs=[dr_method, dr_cols, dr_data, dr_random_state, tsne_perplexity, tsne_iter, umap_neighbors, umap_min_dist, dr_plot_x, dr_plot_y],
            outputs=[dr_df, dr_plot_value, dr_plot_x, dr_plot_y],
            show_progress="full",
            show_progress_on=[dr_df],
        )
        dr_stop_btn.click(fn=None, cancels=[dr_event], queue=False)
        dr_plot_btn.click(
            draw_dr,
            inputs=[dr_data, dr_plot_value, dr_plot_x, dr_plot_y, dr_x_min, dr_x_max, dr_y_min, dr_y_max],
            outputs=[dr_plot, dr_plot_download],
        )

        model_event = model_train_btn.click(
            do_train,
            inputs=[model_type, m_cols, m_cv, m_scoring, dl_name, dl_interp_method, dl_cv, dl_epoch, dl_bs, dl_lr, dl_device],
            outputs=[model_result, model_table],
            show_progress="full",
            show_progress_on=[model_result, model_table],
        )
        model_stop_btn.click(fn=None, cancels=[model_event], queue=False)
        model_plot_btn.click(
            draw_model,
            inputs=[model_name_for_plot, cm_split, metric_name, fold_loss_type],
            outputs=[model_cm_plot, model_metric_plot, model_fold_plot, model_plot_download],
        )
        model_event.then(
            refresh_model_choices,
            inputs=[model_name_for_plot, pred_model_name],
            outputs=[model_name_for_plot, pred_model_name],
            queue=False,
        )

        load_model_event = load_model_btn.click(
            do_load_model,
            inputs=[saved_model_file],
            outputs=[loaded_model_result],
        )
        load_model_event.then(
            refresh_model_choices,
            inputs=[model_name_for_plot, pred_model_name],
            outputs=[model_name_for_plot, pred_model_name],
            queue=False,
        )

        pred_event = pred_btn.click(
            do_predict,
            inputs=[pred_files, pred_model_name],
            outputs=[pred_df],
            show_progress="full",
            show_progress_on=[pred_df],
        )
        pred_event.then(
            refresh_model_choices,
            inputs=[model_name_for_plot, pred_model_name],
            outputs=[model_name_for_plot, pred_model_name],
            queue=False,
        )
        pred_event.then(
            refresh_prediction_columns,
            inputs=[pred_label_col, pred_sample_id, pred_plot_x, pred_plot_y, pred_plot_z, pred_group_col],
            outputs=[pred_label_col, pred_sample_id, pred_plot_x, pred_plot_y, pred_plot_z, pred_group_col],
            queue=False,
        )
        pred_stop_btn.click(fn=None, cancels=[pred_event], queue=False)
        pred_plot_btn.click(
            draw_predict,
            inputs=[pred_plot_kind, pred_sample_id, pred_label_col, pred_plot_x, pred_plot_y, pred_plot_z, pred_group_col, pred_x_min, pred_x_max, pred_y_min, pred_y_max],
            outputs=[pred_plot, pred_plot_download],
        )

        choose_export_dir_btn.click(
            choose_export_directory,
            inputs=[export_dir],
            outputs=[export_dir],
            queue=False,
        )

        export_event = export_btn.click(
            do_export,
            inputs=[export_dir],
            outputs=[export_result],
            show_progress="full",
            show_progress_on=[export_result],
        )
        export_stop_btn.click(fn=None, cancels=[export_event], queue=False)
        gr.HTML(value="", visible="hidden", js_on_load=_tooltip_html_js())
        demo.load(fn=None, js=_tooltip_js(), queue=False)

    demo.queue(default_concurrency_limit=1)
    background_url = demo.serve_static_file(background_path)["url"]
    background_url = quote(background_url.replace("\\", "/"), safe="/:=")
    demo.poremind_head = _tooltip_head(background_url)
    demo.poremind_js = _tooltip_js()
    demo.poremind_i18n = ui_i18n
    return demo


def main() -> None:
    app = create_app()
    app.launch(
        head=getattr(app, "poremind_head", None),
        i18n=getattr(app, "poremind_i18n", None),
        js=getattr(app, "poremind_js", None),
        footer_links=[],
    )


if __name__ == "__main__":
    main()

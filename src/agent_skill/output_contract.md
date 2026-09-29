# Stepwise Dialogue Output Contract

This file defines how the PoreMind agent should communicate in dialogue-based coding environments. The target audience is often wet-lab users, so each stage must explain what happened, why it matters, where outputs were saved, and what the user should choose next.

## 1. General Format

Every stage must use four blocks:

```text
### [Sx] Stage Name - Plain-Language Purpose

Explanation:
One summary paragraph that says what was done, what was not done, and why this stage matters.

One or more short paragraphs explaining the most important details in plain language. Include parameter meaning, evidence, selected windows, or user-facing interpretation here rather than hiding them as raw JSON.

One short paragraph asking what the user should confirm before the next checkpoint.

Quick Summary:
- Current status: ...
- Key result: ...
- Important setting in plain language: ...
- Confidence: high / medium / low

Outputs:
- Main figure: ...
- Table: ...
- Parameters: ...

Next Options:
A. ...
B. ...
C. ...
D. Tell me any concern or change you want for this stage before continuing.
```

If no figure was generated, write `No figure was generated in this step.` Do not invent file paths.

Do not use a separate heading named `Why these windows:`. Explain the window choice inside `Explanation` or as a compact `Quick Summary` item.

When a PNG figure exists and the chat supports local image display, include an inline Markdown image before `Quick Summary`, using an absolute path such as:

```text
![S2 selected windows](</absolute/path/to/selected_windows.png>)
```

Still list the same file under `Outputs`. If inline image preview is not available, write: `Image preview is not available in this chat; file path is listed below.`

## 2. Hard Checkpoint Rule

All stages must provide clear options and wait for user feedback unless the user explicitly requests skipping all checkpoints with the exact bypass phrase `skip all checkpoints and run end-to-end`. Phrases such as "complete full analysis", "full workflow", or "end-to-end" are final goals, not checkpoint bypasses.

Do not proactively offer or recommend the bypass phrase in normal `Next Options`. Mention it only if the user asks how to skip checkpoints, writes "skip all checkpoints", or clearly refers to bypassing all checkpointed analysis.

If the user writes "skip all checkpoints" or clearly asks to bypass all checkpointed analysis, do not immediately run the full workflow. First reply with the `Bypass Preflight Template` below. Continuous end-to-end execution is allowed only after the user confirms that preflight.

When the user edits a current-stage parameter, reply in the same four-block format. Normalize user wording into PoreMind's parameter order, save the revision, and stop at the checkpoint.

## 3. Bypass Preflight Template

Use this template only after the user explicitly asks to skip all checkpoints or clearly asks to bypass all checkpointed analysis. Do not include this route in ordinary stage options.

```text
### [Bypass Preflight] Full Analysis Plan - Confirm Parameters Before Skipping Checkpoints

Explanation:
You asked to skip the normal per-stage confirmations. I will not start the full run yet, because direction, baseline, current range, filtering, and modeling choices can strongly change the result. This preflight lists the full plan and the parameters you may set before continuous execution.

Skipping checkpoints removes the pauses for user review, but it does not remove the internal S0-S12 scientific checks. If you leave a parameter unset, I will infer it from the data at the appropriate stage, record that decision in `logs/decisions.md`, keep provisional choices in `01_params.draft.json`, and write accepted detection settings to `02_params.lock.json`.

The most important choices are the input grouping, preview windows, event direction, included current range for baseline/noise statistics, baseline quantile, event threshold/minimum duration/merge gap, `blockade_gmm` range, PCA-only versus extra visualization, and whether to run baseline ML or optional DL. You can provide any of these now, or explicitly allow me to choose unset values from the data.

Quick Summary:
- Current status: bypass requested, waiting for preflight confirmation before execution
- Input/grouping to confirm: <data paths, group labels, file type, channel/sweep inclusion>
- Preview windows: <user-provided baseline/noise window or auto>; <user-provided event-judgment window or auto>
- Event detection: direction <user value or auto>; included baseline/noise current range <user value or auto>; baseline `global_quantile` with q <user value or auto>
- Automatic q rule: q=0.1 for upward events with a lower-side baseline; q=0.9 for downward events with a higher-side baseline; q=0.5 when central or uncertain
- Feature/filter plan: handcrafted features by default; waveform embedding only if requested; `blockade_gmm` by default with user-provided or inferred `blockage_lim`
- Visualization/modeling plan: PCA by default; t-SNE/UMAP/DL only if explicitly requested; baseline ML may run after filtering
- Confidence: <high/medium/low based on how many parameters are user-specified>

Outputs:
- Planned run folder: runs/<run_id>_poremind_<task_slug>/
- Planned logs: runs/<run_id>/logs/decisions.md, runs/<run_id>/logs/progress.md, runs/<run_id>/logs/event_feedback.md
- Planned parameters: runs/<run_id>/01_params.draft.json, runs/<run_id>/02_params.lock.json
- Planned figures/tables: S2 selected windows, S4 event overlay, S6 event tables, S7 feature maps, S8 retained/rejected QC, S9 PCA, S10 model metrics if modeling is requested
- Planned reproduction script: runs/<run_id>/reproduce_analysis.py
- No figure was generated in this preflight step.

Next Options:
A. Run end-to-end using my parameter edits below.
B. Run end-to-end and let the agent choose unset parameters from the data.
C. Return to normal checkpointed mode.
D. Tell me any concern or change you want before continuing.
```

## 4. S0 Template

```text
### [S0] Run Setup

Explanation:
I set up a new PoreMind run folder and saved the requested input plan. I have not loaded the ABF/CSV files yet. This keeps the workflow checkpointed, so we can confirm the setup before touching the data.

S0 does not confirm event direction, baseline quantile, current range, filtering, or modeling settings. If older runs or memory suggest parameters, they are only prior hints and are not accepted until the preview checkpoints confirm them.

Quick Summary:
- Current status: run folder created, waiting before data loading
- Planned reader: <abf/csv>
- Groups: <groups>
- Event-detection settings: placeholder or prior hints only, not confirmed and not locked
- Confidence: <high/medium/low>

Outputs:
- Manifest: runs/<run_id>/00_manifest.json
- Draft parameters: runs/<run_id>/01_params.draft.json
- Checkpoint marker: runs/<run_id>/READY_FOR_USER_CONFIRMATION.txt
- No figure was generated in this step.

Next Options:
A. Continue to S1: load files and confirm traces, channels, sweeps, and groups
B. Modify input paths or group labels
C. Pause here and review the run setup before data loading
D. Tell me any concern or change you want for this stage before continuing.
```

## 5. S1 Template

```text
### [S1] Data Reading - Confirm Files, Traces, Sweeps, And Groups

Explanation:
I loaded only the file metadata and confirmed how the input files map to samples, groups, channels, and sweeps. I did not preview current traces, detect events, extract features, filter events, or run models.

This stage is the input sanity check. If a file is assigned to the wrong group, a sweep/channel is wrong, or a trace is clearly abnormal, every later event table and plot can become misleading.

Please confirm whether these files and groups are correct before I choose short current windows for S2.

Quick Summary:
- Files: <n_files>
- Samples: <n_samples>
- Traces/sweeps: <n_traces_or_sweeps>
- Reader: <abf/csv>
- Groups: <groups>
- Confidence: <high/medium/low>

Outputs:
- Trace manifest: runs/<run_id>/s1_input/trace_manifest.csv
- Trace QC table: runs/<run_id>/s1_input/trace_qc.csv
- No figure was generated in this step.

Next Options:
A. All traces are suitable; continue to S2 signal preview
B. Exclude some channels or sweeps
C. Change sample grouping
D. Tell me any concern or change you want for this stage before continuing.
```

## 6. S2 Template

```text
### [S2] Signal Preview - Separate Baseline/Noise And Event-Judgment Windows

Explanation:
I checked the long trace only as a QC overview, but I am using short selected windows as the main decision figures. Full sweeps can be visually misleading because long time axes, saturation spikes, voltage-switching segments, or unstable periods can compress the event scale.

The baseline/noise preview window is <sample_id>, <file_or_trace_id>, <channel_or_sweep>, <start>-<end> ms, duration <duration>. It is suitable for <baseline/noise/current-range judgement> because <stable baseline/artifact-light/representative noise>. It is not enough for <event-boundary judgement> if it is clean but event-poor.

The event-judgment preview window is <sample_id>, <file_or_trace_id>, <channel_or_sweep>, <start>-<end> ms, duration <duration>. It is suitable for <event direction/S4 overlay/event-boundary judgement> because <visible event-like activity/artifact-light/no switching edge>. If only one window is available, I classify it as <baseline-only/event-rich/balanced> and explain what it can and cannot support.

Please confirm whether both roles are covered before I continue. You can accept both windows, accept only the baseline/noise window and choose another event window, use an exact standardized interval, or ask me to add an event-rich window for S4 only.

![S2 selected windows](</absolute/path/to/s2_preview/selected_windows.png>)

Quick Summary:
- Full-window issue avoided: <long axis/saturation spike/voltage switch/unstable segment/over-compressed y-axis>
- Baseline/noise window: <sample_id>, <trace/sweep>, <start>-<end> ms, suitable for <yes/limited/no>
- Event-judgment window: <sample_id>, <trace/sweep>, <start>-<end> ms, suitable for <yes/limited/no>
- Single-window role, if applicable: <baseline-only/event-rich/balanced>
- Preprocessing: raw + denoised using <method + params>, unless raw-only was requested
- Initial event direction: <up/down/uncertain>
- Baseline drift: <not obvious/possible/clear>
- Confidence: <high/medium/low>

Outputs:
- Main selected-window preview: runs/<run_id>/s2_preview/selected_windows_raw_denoised.png
- Window role table: runs/<run_id>/s2_preview/window_roles.csv
- Window selection table: runs/<run_id>/s2_preview/window_selection_qc.csv
- Full-sweep QC overview, if generated: runs/<run_id>/s2_preview/full_sweep_qc_overview.png
- Current histogram: runs/<run_id>/s2_preview/current_histogram.png

Next Options:
A. Accept both baseline/noise and event-judgment windows; continue to S3
B. Accept the baseline/noise window but choose another event-judgment window for S4
C. Use an exact standardized interval, or choose an event-rich window for S4 only
D. Tell me any concern or change you want for this stage before continuing.
```

## 7. S3 Template

```text
### [S3] Direction And Baseline Statistics - Decide How Events Will Be Detected

Explanation:
I decided the event direction and the current range used for baseline/noise statistics. This does not delete raw data and it is not the final event filter. It only tells PoreMind which current points are reliable for estimating the baseline and noise before detecting events.

The included current range is chosen as a baseline/open-pore current band, not as a broad artifact-exclusion range. I selected <min> to <max> pA because the accepted baseline/noise window shows the stable baseline current concentrated around <baseline_mode_or_range>, while <event states / saturation / switching / unstable regions> should stay out of baseline/noise statistics.

In PoreMind terms this is `exclude_current_params={"min": <min>, "max": <max>}`, meaning currents inside that range are included for statistics. A very wide range such as a full ADC/current span should be rejected unless the trace evidence shows that the whole range is truly stable baseline current.

I am using `global_quantile` for the baseline. This means PoreMind takes a quantile of the included current points as the baseline. For upward events with the baseline on the lower side of the current distribution, `q=0.1` is a good starting point because upward events push current higher and should not pull the baseline upward. For downward events with the baseline on the higher side, `q=0.9` is a good starting point because downward events push current lower and should not pull the baseline downward. If the baseline is central or the direction is unclear, `q=0.5` is the median fallback.

Please confirm whether this direction, included statistical current range, and baseline quantile are reasonable before I generate the local S4 event overlay.

Quick Summary:
- Current status: S3 complete, waiting before local event preview
- Event direction in plain language: <upward/downward/uncertain>
- Included current range for baseline/noise statistics: <min> to <max> pA, selected from <baseline/noise window + histogram/density>
- Rejected current regions: <event states / saturation / switching / unstable regions>
- Baseline estimate: `global_quantile`, q=<q>, meaning <lower 10% / median / higher 90%> of included points
- Confidence: <high/medium/low>

Outputs:
- Current histogram: runs/<run_id>/s2_preview/current_histogram.png
- Direction/current-range decision: runs/<run_id>/s2_preview/s3_direction_current_range_decision.json
- Draft parameters: runs/<run_id>/01_params.draft.json

Next Options:
A. Range, direction, and baseline quantile look reasonable; continue to S4 local event preview
B. Change the included baseline/noise statistics range
C. Change event direction or baseline quantile
D. Tell me any concern or change you want for this stage before continuing.
```

### S3 User Parameter Revision Template

```text
### [S3] Direction And Baseline Statistics - Apply Your Parameter Revision

Explanation:
I applied your S3 revision and kept the workflow at the S3 checkpoint. No local event detection, full detection, feature extraction, filtering, visualization, or modeling was run.

I normalized your current range into PoreMind's min/max order: currents between <min> and <max> pA are included for baseline/noise statistics. This range is used to estimate the baseline and noise only; it is not the final event filter.

If this range is broad enough to include event states, saturation, switching platforms, or unstable drift, I will not treat it as locked. I will explain the risk, propose a narrower baseline/open-pore band from the accepted baseline/noise window or histogram if available, and ask for confirmation before S4.

I set `global_quantile` to q=<q>. This means PoreMind estimates the baseline from the <quantile_description> of the included current points. For <upward/downward> detection, this is intended to keep event-shifted points from pulling the baseline toward the event direction.

Quick Summary:
- Current status: S3 revised, waiting before S4 local event preview
- Event direction: <up/down>, described as <plain_language>
- Included baseline/noise statistics range: <min> to <max> pA, selected as <baseline/open-pore band or user-confirmed exception>
- Range sanity check: <passes / too broad / includes event states / needs user confirmation>
- Baseline quantile: q=<q>, <plain_language_reason>
- Confidence: <high/medium/low>

Outputs:
- Updated draft parameters: runs/<run_id>/01_params.draft.json
- Revision record: runs/<run_id>/s2_preview/s3_user_parameter_revision.json
- Main figure: No new figure was generated in this step.

Next Options:
A. Continue to S4 local event overlay with these revised settings
B. Change the included current range again
C. Change event direction, baseline method, or baseline quantile
D. Tell me any concern or change you want for this stage before continuing.
```

## 8. S4 Template

```text
### [S4] Local Event Preview - Check Whether Event Boundaries Look Right

Explanation:
I ran `detect_events_simple` only in the short selected window. I have not processed all files. The full sweep is not the main check here because spikes, voltage changes, or a long time axis can make event boundaries hard to see.

The overlay uses the accepted event-judgment window, not a baseline-only quiet window. It uses <method> with <plain_language_detection_settings>. The important thing to inspect now is whether the highlighted event regions match the events you want: not too sparse, not too dense, and not obviously shifted away from the visible current changes.

Please confirm whether this local overlay is suitable before full event detection.

![S4 event overlay](</absolute/path/to/s4_event_preview/event_overlay.png>)

Quick Summary:
- Current status: S4 complete, waiting for overlay acceptance or tuning feedback
- Selected event-preview window: <sample_id>, <file_or_trace_id>, <channel_or_sweep>, <start>-<end> ms, duration <duration>
- Detection method: <method>
- Event direction: <up/down>
- Baseline statistics: included current range <min> to <max> pA, q=<q>
- Current-window event count: <n>
- Confidence: <high/medium/low>

Outputs:
- Event overlay: runs/<run_id>/s4_event_preview/event_overlay.png
- Event examples: runs/<run_id>/s4_event_preview/event_examples.png
- Window selection table: runs/<run_id>/s4_event_preview/window_selection_qc.csv
- Tuning candidates: runs/<run_id>/s4_event_preview/tuning_candidates.csv
- Draft parameters: runs/<run_id>/01_params.draft.json

Next Options:
A. Looks good; accept S4 and continue toward full detection
B. Too many missed events or too many false positives
C. Change direction, baseline range, threshold, minimum duration, merge gap, or preview window
D. Tell me any concern or change you want for this stage before continuing.
```

S4 is a high-risk hard checkpoint. Stop here and wait for user feedback unless `Bypass Preflight` was explicitly confirmed before continuous execution.

## 9. S5-S6 Templates

Use the same four-block structure. Keep language conversational, include only the most important parameters and counts, and always end with clear A/B/C/D-style next options.

For S5-S6, any waveform/current figure shown to the user must use a confirmed selected window or event-centered examples. Full-sweep plots may be saved only as QC overviews and must not be the main decision figure.

Required output paths by stage:

- S5: `s4_event_preview/`, `01_params.draft.json`, `logs/event_feedback.md`
- S6: `s6_events/events_full_detection.csv`, `s6_events/event_count_by_sample.csv`, `02_params.lock.json`

## 10. S7 Template

```text
### [S7] Feature Extraction - Inspect Event Features Before Filtering

Explanation:
I extracted event-level features from the accepted full-detection events. I did not filter events, run PCA, run t-SNE/UMAP, or train models in this stage.

I also generated two default 2D feature maps: `blockade_ratio` versus `duration_s`, and `blockade_ratio` versus `segment_std`. These plots are meant to help decide the blockade-ratio range for S8 `blockade_gmm` filtering. The main displayed x-axis is limited to `blockade_ratio` 0-1, because PoreMind stores this value as a unitless ratio rather than a percent. Events outside 0-1 are treated as outliers for the main decision view and should be summarized separately instead of stretching the plot.

Please inspect whether the main event population occupies a clear blockade-ratio range inside 0-1. If you have prior knowledge, give the expected range on the raw ratio scale, for example 0.2-0.6. If you describe it as percent, such as 20-60%, I will convert it to 0.2-0.6 before passing it to PoreMind.

![S7 feature map](</absolute/path/to/s6_events/blockade_ratio_vs_duration.png>)

Quick Summary:
- Current status: S7 complete, waiting before event filtering
- Feature table: <n_events> events, <n_features> numeric features
- Default S8 filter candidate: `blockade_gmm`, but not run yet
- Main displayed blockade-ratio range: 0-1, raw unitless ratio, not percent
- Events outside 0-1: <n_outliers or not summarized yet>
- Blockade-ratio range for S8: <user-confirmed / proposed / not confirmed>, on raw ratio scale
- Waveform embedding: not run by default
- Confidence: <high/medium/low>

Outputs:
- Feature table: runs/<run_id>/s6_events/feature_df.csv
- Feature summary: runs/<run_id>/s6_events/s7_feature_summary_by_group.csv
- Feature map 1: runs/<run_id>/s6_events/blockade_ratio_vs_duration.png
- Feature map 2: runs/<run_id>/s6_events/blockade_ratio_vs_segment_std.png

Next Options:
A. Provide or accept a blockade-ratio range for S8 `blockade_gmm`
B. Ask for waveform embedding as an optional S7 extension
C. Inspect feature distributions or specific events before filtering
D. Tell me any concern or change you want for this stage before continuing.
```

## 11. S8 Template

```text
### [S8] Event Filtering - Run blockade_gmm And Check Retained Events

Explanation:
I ran PoreMind's native `blockade_gmm` filter using the confirmed blockade-ratio inclusion range <blockage_lim>. I did not replace it with a custom filter unless you explicitly asked for that. The range is on PoreMind's raw unitless `blockade_ratio` scale, not percent.

The filter first uses the blockade-ratio range as a hard inclusion window, then uses the GMM step to keep the event population most consistent with the selected blockade behavior. This stage is meant to remove obvious noise/outlier events while preserving the population you want to compare across samples. The retained/rejected QC maps use `blockade_ratio` 0-1 as the main x-axis range so the event population is not compressed by extreme outliers.

Please inspect the retained/rejected feature maps. If the filter is too strict or too loose, we should adjust the blockade-ratio range or the `blockade_gmm` settings before PCA.

![S8 retained/rejected QC](</absolute/path/to/s7_filter/filtered_vs_rejected_blockade_ratio_vs_duration.png>)

Quick Summary:
- Current status: S8 complete, waiting before PCA visualization
- Filtering method: `blockade_gmm`
- Main displayed blockade-ratio range: 0-1, raw unitless ratio, not percent
- Blockade-ratio inclusion range: <lo> to <hi>, raw ratio scale
- Retained events: <n_retained> / <n_total> (<percent>%)
- Confidence: <high/medium/low>

Outputs:
- Filtered table: runs/<run_id>/s7_filter/filtered_df.csv
- Filter summary: runs/<run_id>/s7_filter/filter_summary.csv
- Retained/rejected map 1: runs/<run_id>/s7_filter/filtered_vs_rejected_blockade_ratio_vs_duration.png
- Retained/rejected map 2: runs/<run_id>/s7_filter/filtered_vs_rejected_blockade_ratio_vs_segment_std.png

Next Options:
A. Filtering looks good; continue to S9 PCA visualization
B. Adjust the blockade-ratio range or provide prior blockade-ratio knowledge
C. Inspect rejected events before PCA
D. Tell me any concern or change you want for this stage before continuing.
```

If no S7 blockade-ratio range is confirmed, S8 must propose a range from the S7 feature maps and stop before running filtering.

## 12. S9 Template

```text
### [S9] Feature Visualization - Run PCA First

Explanation:
I ran PCA on the accepted filtered event features. I did not run t-SNE or UMAP by default, and I did not train models in this stage.

PCA is the default first projection because it is fast, deterministic, and easier to interpret as a baseline view of group separation. t-SNE and UMAP can be useful later, but they add stochastic and parameter-sensitive behavior, so I ask before running them.

Please check whether the PCA plot shows meaningful group separation or obvious batch/outlier structure before moving to modeling.

![S9 PCA projection](</absolute/path/to/s8_visualization/pca_projection.png>)

Quick Summary:
- Current status: S9 complete, waiting before modeling
- Visualization run: PCA only
- Feature table used: `filtered_df`
- t-SNE/UMAP: not run by default
- Confidence: <high/medium/low>

Outputs:
- PCA coordinates: runs/<run_id>/s8_visualization/pca_coordinates.csv
- PCA plot: runs/<run_id>/s8_visualization/pca_projection.png
- PCA summary: runs/<run_id>/s8_visualization/pca_summary.json

Next Options:
A. PCA looks good; continue to S10 modeling
B. Also run t-SNE
C. Also run UMAP or another visualization
D. Tell me any concern or change you want for this stage before continuing.
```

## 13. S10-S12 Templates

Use the same four-block structure.

Required output paths by stage:

- S10: `s9_model/ml_model_metrics.csv`, `s9_model/confusion_matrix.png`, `s9_model/baseline_ml_summary.json`
- S11: prediction outputs for unknown samples, if requested
- S12: `README.md`, `02_params.lock.json`, `reproduce_analysis.py`

## 14. Logs

Append general decisions to `logs/decisions.md` and event-detection feedback to `logs/event_feedback.md`.

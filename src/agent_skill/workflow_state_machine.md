# Workflow State Machine

This file defines the PoreMind agent execution order. Recipes may provide warm-start hints, but they must not bypass this state machine.

Hard rule: every S0-S12 stage is a checkpoint. The agent may run only one stage per turn, report outputs and next options, then stop and wait for explicit user confirmation. The only exception is when the user explicitly asks to skip all checkpoints with the exact bypass phrase `skip all checkpoints and run end-to-end`. Do not proactively offer or recommend the bypass phrase in normal next options.

Bypass preflight branch: if the user writes "skip all checkpoints" or clearly asks to bypass all checkpointed analysis, first produce a `Bypass Preflight` response and list the full plan plus user-settable parameters. Do not run continuously until the user confirms that preflight. After confirmation, continuous execution still follows S0-S12 internally, logs each decision, writes provisional settings to `01_params.draft.json`, writes accepted detection settings to `02_params.lock.json`, and summarizes user-provided versus agent-inferred parameters at the end.

## S0 Initialize Task

Create the run folder, `00_manifest.json`, `01_params.draft.json`, and `logs/decisions.md`. Do not confirm or imply event direction, baseline quantile, current range, filtering, or modeling settings in S0. Memory-derived or older-run settings may be recorded only as prior hints, not accepted parameters. Stop and wait for confirmation to enter S1.

## S1 Read Data and Confirm Groups

Use `create_analysis_object(...).load()`, build a trace manifest, identify channels/sweeps or CSV columns, and flag clearly abnormal traces. Stop and wait for confirmation that samples, traces, and groups are correct.

## S2 Preview Raw and Denoised Traces

Use `analysis.preview_signal`, `analysis.denoise`, and `analysis.pl.current` or equivalent saved plots. Generate a current histogram. Full sweeps may be saved only as QC overviews; they are not the main decision figure. When denoising is available, show raw + denoised current unless the user explicitly asks for raw-only. Select and label two roles: a `baseline/noise preview window` for baseline/noise/current-range judgement and an `event-judgment preview window` for event direction and S4 overlay judgement. Save `s2_preview/selected_windows_raw_denoised.png`, `s2_preview/window_roles.csv`, and `s2_preview/window_selection_qc.csv`. If only one window is shown, classify it as `baseline-only`, `event-rich`, or `balanced`. Stop and wait for confirmation that the required roles are covered.

## S3 Decide Event Direction and Current Range

Choose `detect_direction`, choose or request `exclude_current_params`, default to `baseline_method="global_quantile"`, and choose `baseline_params.q` from event direction and baseline position: `q=0.1` for upward events when the baseline is on the lower side, `q=0.9` for downward events when the baseline is on the higher side, and `q=0.5` when the baseline is central or uncertain. S3 may use a confirmed baseline/noise window for baseline/current-range decisions, but it must not imply that S4 event-boundary tuning is ready unless an event-judgment window is also accepted. Explain that `exclude_current_params` defines the included current range for baseline/noise statistics, not the final event filter. Choose this range as a narrow baseline/open-pore current band from the accepted baseline/noise window and local histogram/density. Reject broad artifact-exclusion ranges, ranges inferred only from full sweeps, and ranges that include event-state peaks, saturation, switching platforms, or unstable drift. Update `01_params.draft.json` and stop for confirmation.

## S4 Local Event-Detection Preview

Run `analysis.detect_events_simple` only on an accepted event-judgment window. Do not tune parameters from a baseline-only quiet window. If the S2 event-judgment window is missing, event-poor, or marked limited, first select or request an event-rich artifact-light window and stop for confirmation. Save event overlays/examples and present fixed feedback options. The reply must state whether the selected window is representative enough for the sample and ask whether the user wants to keep, move, extend, shorten, or replace the window. Stop and wait for confirmation.

## S5 Parameter Iteration

Adjust only parameters related to user feedback. If feedback is about the preview window, change only the selected window first and regenerate the local preview before changing detection parameters. Save a new preview and draft parameters. Stop and wait for confirmation.

## S6 Full Event Detection

After confirmation, run `analysis.detect_events`, save event counts and QC outputs, and write `02_params.lock.json`. Stop and wait for confirmation.

## S7 Feature Extraction

Run `analysis.extract_features`, save `s6_events/feature_df.csv`, and generate default 2D feature maps: `blockade_ratio` vs `duration_s` and `blockade_ratio` vs `segment_std`. The main user-facing maps must display `blockade_ratio` from 0 to 1 on the x-axis. PoreMind stores `blockade_ratio` as a raw unitless ratio, not a percentage; values outside 0-1 should be counted or summarized as outliers rather than allowed to stretch the main plot. Ask the user to confirm the blockade-ratio inclusion range for S8 `blockade_gmm` and whether there is prior blockade-ratio knowledge. Offer waveform embedding only as an optional S7 extension; do not run it by default. Stop and wait for confirmation.

## S8 Event Filtering and QC

Use PoreMind native `analysis.filter_events(method="blockade_gmm")` by default with the user-confirmed or S7-proposed `blockage_lim`. If no blockade-ratio range has been confirmed, propose a range from S7 feature maps and stop before filtering. Save `s7_filter/filtered_df.csv`, report retention/QC, and generate retained/rejected maps for `blockade_ratio` vs `duration_s` and `blockade_ratio` vs `segment_std` with the displayed `blockade_ratio` axis limited to 0-1. If the user gives a percent range, convert it to raw ratio before calling PoreMind. Waveform examples must be event-centered or selected-window plots, not full sweeps as the main figure. Stop and wait for confirmation.

## S9 Visualization

Run PCA by default and save PCA coordinates/plots under `s8_visualization/`. Do not run t-SNE or UMAP by default; ask the user whether they want either after PCA. If showing raw current or waveform context, use confirmed selected windows or event-centered examples. Stop and wait for confirmation before modeling or reporting.

## S10 Modeling and Representation

Run ML/DL only after prior checkpoints are accepted. PoreMindWaveformEncoder-compatible embeddings belong to optional S7 feature extraction, not default S10 modeling. Any waveform interpretation figure must be event-centered or selected-window based. For the current DL API, `build_DL_model` defaults to `interp_length=500` and `interp_method="interp"`; `interp_method="padding"` is the alternative. `model_name="MAGJAM"` selects d8 and `model_name="MAGJAM_d4"` selects d4; both use waveform inputs only. Stop and wait for confirmation of results and interpretation limits.

## S11 Unknown-Sample Prediction

Run `analysis.classify_new_samples` using the same preprocessing, event-detection, feature, and model parameters. Stop and wait for confirmation.

## S12 Report and Reproduction

Save `README.md`, `reproduce_analysis.py`, confirmed parameters, limitations, and stage outputs. The workflow is complete; wait for the user to decide whether to export, rerun, change parameters, or stop.

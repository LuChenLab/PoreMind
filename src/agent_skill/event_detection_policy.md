# Event Detection Policy

Event detection is the most important checkpoint in single-molecule nanopore analysis. A poor event boundary will affect every downstream feature, filter, plot, and model.

## 1. Gate Rule

Unless the user explicitly asks to skip all checkpoints with the exact bypass phrase `skip all checkpoints and run end-to-end`, do not run full event detection before local preview is accepted. Do not proactively offer that phrase in normal next options.

If the user writes "skip all checkpoints" or clearly asks to bypass all checkpointed analysis, first show the `Bypass Preflight` plan and parameter checklist. Bypass removes the user pause after each stage only after confirmation; it does not remove event-detection safeguards. During continuous execution, still choose direction, baseline/current range, quantile, threshold, minimum duration, and merge settings from the S1-S4 evidence, save the reasoning, and report low-confidence choices in the final summary.

Do not directly run:

```python
analysis.detect_events(...)
```

First run:

```python
analysis.detect_events_simple(...)
```

Save an event overlay and wait for user confirmation.

A request for "complete full event detection, feature extraction, event filtering, visualization, and ML classification" is a final goal, not permission to skip checkpoints.

The local preview must use a short selected event-judgment window. A full sweep may be saved as a QC overview, but it must not be the main event-boundary decision figure. Do not tune event parameters from a baseline-only quiet window.

## 2. Required Parameters Before Full Detection

Full detection requires explicit values for:

```text
detect_direction
exclude_current / exclude_current_params
baseline_method / baseline_params
detect_method / detect_params
merge_event / merge_event_params
```

## 3. Tuning Order

Use this order:

```text
1. Event direction: up or down
2. Effective current range: exclude_current_params
3. Baseline strategy: default global_quantile
4. Detection method: default threshold
5. Threshold parameters: sigma_k, z, penalty, etc.
6. min_duration_s
7. merge_event settings
8. Local preview
9. User feedback
10. Full detection after confirmation
```

For step 2, `exclude_current_params` must be a narrow baseline/open-pore current band, not a broad artifact-exclusion range. Derive it from the accepted `baseline/noise preview window` plus the local current histogram/density. Reject ranges that are hundreds or thousands of pA wide without clear evidence, include event-state peaks, include saturation or voltage-switching platforms, or come only from a full-sweep histogram. If the baseline band is unclear, stop and ask the user to confirm the open-pore current range before tuning thresholds.

## 4. Detection Methods

- `threshold`: default; good for clear, sparse, high-amplitude events.
- `zscore_threshold`: useful when a z-score threshold is easier to explain.
- `cusum`: useful for small but sustained shifts.
- `pelt`: useful when precise change points matter.
- `hmm`: useful for multi-level events or hidden state transitions.

Warm-start examples:

```python
threshold = {"sigma_k": 5.0, "min_duration_s": 0.0, "noise_method": "mad"}
zscore_threshold = {"z": 4.0, "min_duration_s": 0.0, "noise_method": "mad"}
cusum = {"drift": 0.02, "threshold": 8.0, "min_duration_s": 0.0, "noise_method": "mad"}
pelt = {"model": "l2", "penalty": 8.0, "sigma_k": 3.0, "min_duration_s": 0.0, "noise_method": "mad"}
hmm = {"n_components": 2, "covariance_type": "diag", "n_iter": 200, "min_duration_s": 0.0}
```

## 5. Candidate Parameter Preview

Candidate values may be tested in a local window, but do not run a full-data brute-force grid.

Record candidate event count, event rate, median duration, median blockade ratio, median SNR, short-event fraction, and visual fit. Save to `s4_event_preview/tuning_candidates.csv`.

Before comparing candidate parameters, confirm that the selected event-judgment window is visually suitable. The reply must state the selected sample/trace/sweep/start/end/duration, explain why the full sweep is not being used as the main figure, and say whether the short window is clear enough to judge events. If the user rejects the window, first choose or ask for another window; do not retune parameters from an unclear preview. If only a baseline/noise window has been accepted, S4 must first select or request an event-rich artifact-light window.

Allowed window-change requests:

- Exact range: sample/trace/sweep/start/end.
- Move earlier or later by a user-provided duration.
- Extend or shorten the current window.
- Switch to another trace, channel, sweep, or sample.

## 6. Feedback Mapping

- A looks good: freeze parameters, write `02_params.lock.json`, and proceed only after the checkpoint is accepted.
- B missed events: check direction, current range, threshold, minimum duration, and weak-event methods.
- C false positives: check noise regions, raise threshold, increase minimum duration, or improve preprocessing.
- D wrong direction: switch `down`/`up` and regenerate preview.
- E fragmented boundaries: enable/increase merge gap, or consider `pelt`/`hmm`.
- F boundaries too wide: reduce merge gap, raise threshold, or re-check baseline.
- G unstable baseline: compare `global_quantile` and `rolling_quantile`.
- H wrong current range: redraw histogram/annotation and confirm min/max.
- I another window: keep parameters and preview a different representative window.
- J exact window range: regenerate the preview for the requested sample/trace/sweep/start/end before tuning.
- K move/extend/shorten window: update the selected window and regenerate the overlay before tuning.

Never treat automatic tuning as user confirmation.

When the user gives a specific parameter revision, keep the four-block output format. In `Explanation`, confirm the requested change in plain language, explain what it means for baseline/noise statistics or event detection, and state that no later stage was run. In `Quick Summary`, report the updated values in plain language first and raw parameter names only second. In `Outputs`, list the updated `01_params.draft.json` and a revision record. In `Next Options`, include options to continue, revise again, inspect an overlay, or raise any concern.

For `exclude_current_params`, explain that the min/max range defines which current points are included for baseline/noise statistics. For `global_quantile`, explain that PoreMind takes a quantile of the included points as the baseline: use `q=0.1` for upward events when baseline is lower-side, `q=0.9` for downward events when baseline is higher-side, and `q=0.5` when central or uncertain.

If the user's revision gives an extremely broad current range, do not silently accept it. Reply in the four-block format, explain that this would mix baseline and event/artifact states in baseline/noise statistics, propose a narrower baseline/open-pore band from the current histogram if available, and ask for confirmation.

## 7. Full-Detection QC

After full detection, report event counts per sample/trace, total event count, event rate, median duration, median blockade ratio, abnormal samples, and traces with no events. If obvious problems appear, stop and offer options to continue, inspect, retune, or exclude abnormal traces.

## 8. Long-Running Steps

Before heavy steps, explain that computation may take time and append progress to `logs/progress.md`. On failure, write `logs/errors.md` and preserve completed outputs where possible.

## 9. Uncertainty

If event direction, current range, drift, or event boundaries cannot be judged reliably from the selected window, say so plainly. Ask the user to confirm the open-pore current range or choose another representative window.

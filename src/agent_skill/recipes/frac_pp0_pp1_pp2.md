# Recipe: FraC PP0 / PP1 / PP2 Phosphopeptide

This recipe is warm-start guidance for FraC phosphopeptide-style PP0/PP1/PP2 analysis. It must not bypass the checkpoint workflow.

## Typical Pattern

- Events are often downward blockades, but direction must be confirmed from trace preview.
- PP1/PP2 may overlap, so avoid over-interpreting event-level models.
- Peak-based blockade features can be useful when segment means blur the distribution.

## Warm-Start Parameters

```python
detect_direction = "down"
detect_method = "threshold"
detect_params = {"sigma_k": 5.0, "min_duration_s": 0.0002, "noise_method": "mad"}
baseline_method = "global_quantile"
baseline_params = {"q": 0.9}  # down/higher-side baseline warm start; use q=0.1 for upward/lower-side, q=0.5 if uncertain
merge_event = True
merge_event_params = {"merge_gap_ms": 0.05}
exclude_current = True
exclude_current_params = None  # confirm as a narrow baseline/open-pore band from the baseline/noise window
```

Do not use a broad range such as `-1000 to 1000 pA` as the baseline/noise statistics band. For this style of negative-current ABF data, a stable open-pore band may be much narrower, for example around the dominant baseline mode such as `-120 to -60 pA`, but it must be confirmed from the actual preview window and histogram.

## Suggested Features

Use built-in features plus optional KDE/peak features when justified.

```python
["duration_s", "blockade_ratio", "segment_std", "segment_skew", "segment_kurt"]
```

## Suggested Outputs

- `s4_event_preview/event_overlay.png`
- `s6_events/feature_df.csv`
- `s7_filter/filtered_df.csv`
- `s8_visualization/pca_projection.png`
- `s9_model/ml_model_metrics.csv`

## Common Failure Modes

- Wrong event direction.
- Saturation or blocked-pore regions included in baseline/noise statistics.
- A broad `exclude_current_params` range mixes baseline and event states, inflates noise, and causes missed PP0 events.
- Over-filtering PP0 and losing useful comparison events.
- Treating event-level CV as sample-level validation.

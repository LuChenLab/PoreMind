# Recipe: RNA Conformation / Length / Reaction Time-Course

This recipe is warm-start guidance for RNA conformation, length, or time-course experiments.

## Typical Pattern

- Event distributions may change gradually across time or reaction condition.
- Baseline drift is common and must be checked during trace preview.
- Time-course interpretation should focus on distribution shifts, not only classifier accuracy.

## Warm-Start Parameters

```python
detect_direction = "down"
detect_method = "threshold"
detect_params = {"sigma_k": 4.0, "min_duration_s": 0.0001, "noise_method": "mad"}
baseline_method = "global_quantile"
baseline_params = {"q": 0.9}  # down/higher-side baseline warm start; use q=0.1 for upward/lower-side, q=0.5 if uncertain
merge_event = True
merge_event_params = {"merge_gap_ms": 0.1}
```

## Suggested Outputs

- `s8_visualization/event_summary_by_time.csv`
- `s8_visualization/reaction_trajectory_duration.png`
- `s8_visualization/reaction_trajectory_blockade.png`

## Common Failure Modes

- Treating drift as a biological shift.
- Mixing time points before QC.
- Over-interpreting event-level predictions as sample-level conclusions.

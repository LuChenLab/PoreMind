# Recipe: Solid-State Nanopore Protein

This recipe is warm-start guidance for solid-state nanopore protein events.

## Typical Pattern

- Events may be sparse, heterogeneous, and affected by pore state.
- Open-pore current stability is a major QC point.
- Filtering should be conservative and interpretable.

## Warm-Start Parameters

```python
detect_direction = "down"
detect_method = "threshold"
detect_params = {"sigma_k": 5.0, "min_duration_s": 0.0002, "noise_method": "mad"}
baseline_method = "global_quantile"
baseline_params = {"q": 0.9}  # down/higher-side baseline warm start; use q=0.1 for upward/lower-side, q=0.5 if uncertain
merge_event = False
exclude_current = True
exclude_current_params = None
```

## Suggested Outputs

- Trace QC table.
- Local event overlay.
- Event count by sample/trace.
- Filtered feature distributions.

## Common Failure Modes

- Including blocked-pore plateaus in baseline/noise statistics.
- Calling noise spikes as short protein events.
- Comparing groups when pore state differs strongly.

# Recipe: MspA polydA / polydC

This recipe is warm-start guidance for MspA polydA/polydC analysis.

## Typical Pattern

- Event direction and open-pore current must be confirmed from trace preview.
- polydA/polydC separation may appear in blockade ratio, dwell time, or waveform shape.
- Avoid training models before event boundaries and filtering are accepted.

## Warm-Start Parameters

```python
detect_direction = "down"
detect_method = "threshold"
detect_params = {"sigma_k": 5.0, "min_duration_s": 0.0002, "noise_method": "mad"}
baseline_method = "global_quantile"
baseline_params = {"q": 0.9}  # down/higher-side baseline warm start; use q=0.1 for upward/lower-side, q=0.5 if uncertain
merge_event = True
merge_event_params = {"merge_gap_ms": 0.1}
exclude_current = True
exclude_current_params = None
```

## Suggested Outputs

- Local event overlay.
- Feature distributions by group.
- PCA projection after filtering by default; ask before t-SNE/UMAP.
- Event-level baseline ML only after filtering is accepted.

## Common Failure Modes

- Current histogram includes blocked-pore regions.
- Event windows are too wide because of baseline drift.
- Model performance is interpreted beyond event-level separation.

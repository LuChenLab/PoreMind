# Recipe: MspA Amino Acid

This recipe is warm-start guidance for MspA single-amino-acid event analysis. Parameters are not fixed truth; confirm them through checkpoints.

## Typical Pattern

- Events may have multi-level current patterns.
- Waveform shape may matter more than simple summary statistics.
- Compare handcrafted features first; add `residual-CNN`, `MAGJAM`, or `MAGJAM_d4` only when event boundaries are reliable and the user asks for optional waveform-based DL classification. `PoreMindWaveformEncoder` remains a future-compatible interface and is not available in the current package.

## Warm-Start Parameters

```python
detect_direction = "down"
detect_method = "threshold"
detect_params = {"sigma_k": 3.0, "min_duration_s": 0.0, "noise_method": "mad"}
baseline_method = "global_quantile"
baseline_params = {"q": 0.9}  # down/higher-side baseline warm start; use q=0.1 for upward/lower-side, q=0.5 if uncertain
merge_event = True
merge_event_params = {"merge_gap_ms": 2}
exclude_current = True
exclude_current_params = None
```

## Suggested Features and Models

Start with `duration_s`, `blockade_ratio`, `segment_std`, `segment_skew`, and `segment_kurt`. Add waveform features, embeddings, or residual-CNN only after event boundaries are accepted and the user chooses the optional S7 extension.

## Common Failure Modes

- Multi-level events split into too many fragments.
- Weak events missed by a high threshold.
- Handcrafted features overlap strongly.
- Deep models trained on unreliable event windows.

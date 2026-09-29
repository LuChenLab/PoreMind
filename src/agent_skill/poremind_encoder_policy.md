# PoreMindWaveformEncoder Policy

`PoreMindWaveformEncoder` is an optional S7 event-waveform representation tool. It does not detect events and does not fix event boundaries.

Do not run waveform embeddings by default. Use it only when the user explicitly asks for embeddings, or when S7 feature maps show that handcrafted features are insufficient and the user accepts this optional extension. Use it only after:

1. Event direction, current range, and event-detection parameters are confirmed.
2. `analysis.detect_events(...)` has been run.
3. `analysis.extract_features(...)` has been run.
4. `analysis.filtered_df` exists, or the user has explicitly chosen to use unfiltered events.

## Current Package Status

`PoreMindWaveformEncoder` is not implemented/exported by the current PoreMind package. Keep the policy below as a future-compatible extension, but do not call this import or fabricate embedding columns in the current environment.

The currently available waveform-based DL models are MAGJAM and MAGJAM_d4 through `MultiSampleAnalysis.build_DL_model`; they are classifiers, not interchangeable encoder objects.

## Import (future-compatible only)

```python
from poremind import PoreMindWaveformEncoder  # unavailable in the current package
```

If this class is not exposed in the current PoreMind environment, do not invent embeddings. Say that the interface is unavailable and continue with handcrafted features and/or PoreMind DL.

## Adapter Guidance

Use a small adapter layer because the final encoder API may vary:

```python
def add_poremind_embeddings(analysis, encoder_config=None, data="filtered", prefix="PME"):
    from poremind import PoreMindWaveformEncoder
    encoder = PoreMindWaveformEncoder(**(encoder_config or {}))
    df = analysis.filtered_df if data == "filtered" else analysis.feature_df
    if df is None:
        raise ValueError("feature_df/filtered_df is required before encoding")
    return encoder, df
```

## Recommended Inputs

Use confirmed event windows, denoised waveforms, `start_idx`, `end_idx`, `trace_id`, optional expansion, and optional normalization such as `blockade`, `mad`, `minmax`, or `none`.

## Recommended Outputs

Embedding columns should be named:

```text
PME_001, PME_002, ..., PME_N
```

Save:

```text
s6_events/embeddings.csv
s6_events/encoder_config.json
s6_events/embedding_projection.png
s6_events/feature_vs_embedding.png
```

## Required Comparison

When embeddings are later used in modeling, compare:

1. Handcrafted features only.
2. Encoder embeddings only.
3. Handcrafted + embeddings.

Do not report embeddings alone as the final modeling result unless the user explicitly asks for that.

## Good Use Cases

Use embeddings when amino acid, peptide, or nucleic-acid events have complex waveform shapes; handcrafted features overlap strongly; events are multi-level; or the user wants unsupervised waveform visualization. Do not run embeddings by default.

Avoid embeddings when event boundaries are unreliable, class counts are very small, sample groups are not confirmed, or the user only wants quick event counting.

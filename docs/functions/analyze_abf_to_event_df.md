# `analyze_abf_to_event_df`
**Module:** `pipeline.py`

Compact low-level pipeline from one ABF/CSV input to an event dataframe. For the richer multi-sample workflow, use `create_analysis_object`.

## Parameters

- `path` (`str | Path`): Input path.
- `config` (`AnalysisConfig | None`): Configuration. Defaults to ABF input, `butterworth_filtfilt`, `rolling_quantile(window=501, q=0.5)`, and threshold detection with `sigma_k=5.0` and `min_duration_s=2e-4`.
- `**reader_kwargs` (`Any`): Reader arguments. ABF with no reader arguments expands every channel/sweep; passing `channel`/`sweep` reads one pair. CSV accepts `current_col`, `time_col`, and `sampling_rate_hz`.

## Returns

`pandas.DataFrame`. For ABF, `channel` and `sweep` columns are included.


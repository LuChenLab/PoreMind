# `read_csv`
**Module:** `io.py`

Reads CSV waveform data into a Trace object.

## Parameters
- `path` (`str | Path`): CSV file path.
- `current_col` (`str`): Current column name.
- `time_col` (`str | None`): Optional time column name.
- `sampling_rate_hz` (`float | None`): Sampling rate when time column is absent.

## Returns
- `Trace`

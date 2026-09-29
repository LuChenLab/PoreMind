# `MultiSampleAnalysis.preview_signal`
**Module:** `workflow.py`

Returns downsampled time-current table for a selected time range.

## Parameters
- `sample_id` (`str`): Trace id to preview.
- `start_s` (`float`): Start time in seconds.
- `end_s` (`float | None`): End time in seconds; None means to trace end.
- `max_points` (`int`): Maximum points after downsampling.

## Returns
- `pandas.DataFrame with columns time/current.`

# `PlotAccessor.event_current_label`
**Module:** `pl.py`

Visualize detected events and annotate each selected event with a label from `feature_df`.

## Parameters
- `sample_id` (`str | None`): Trace id; None selects first trace.
- `current` (`str`): `raw` or `denoise`.
- `start_event` (`int`): 1-based start event index (inclusive).
- `end_event` (`int`): 1-based end event index (inclusive intent).
- `lable_col` (`str`): Column name in `feature_df` used as annotation text.
- `label_size` (`float`): Font size for annotations.
- `lable_color` (`dict[str, str] | None`): Optional color map for label text (key is label string, value is color).
- `label_offset` (`float`): Offset above each event-segment mean for placing text labels (default `3.0`).
- `start_ms` (`float`): Fallback start time when selected range has no events.
- `end_ms` (`float`): Fallback end time when selected range has no events.
- `ylim` (`tuple[float, float] | None`): Optional y-axis range.
- `width` (`float`): Figure width.
- `height` (`float`): Figure height.

## Notes
- This method is based on `event_current` and adds text labels near each selected event's segment mean plus `label_offset`.
- Labels are matched by `trace_id`/`sample_id` + `event_id` from `feature_df`.

## Returns
- `matplotlib Axes`

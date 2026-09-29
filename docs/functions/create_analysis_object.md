# `create_analysis_object`
**Module:** `workflow.py`

Factory that creates a MultiSampleAnalysis object for multi-sample nanopore workflows.

## Parameters
- `sample_paths` (`dict[str, str | Path]`): Mapping from sample id to input file path.
- `sample_to_group` (`dict[str, str] | None`): Optional sample-to-label mapping for supervised modeling.
- `reader` (`str`): Input reader type: "abf" or "csv".
- `reader_kwargs` (`dict[str, Any] | None`): Reader-specific keyword arguments.

## Returns
- `MultiSampleAnalysis instance.`

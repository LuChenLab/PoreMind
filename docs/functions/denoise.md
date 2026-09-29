# `MultiSampleAnalysis.denoise`
**Module:** `workflow.py`

Apply one preprocessing method to every loaded trace and cache the result in `analysis.denoised`.

## Parameters

- `method` (`str`): `butterworth_filtfilt` by default; also supports `none`, `moving_average`, `median`, and `drift_corrected_moving_average`.
- `**kwargs` (`Any`): Method-specific parameters. The Butterworth defaults are `filtfilt_N=2` and `filtfilt_Wn=0.1`.

## Returns

`MultiSampleAnalysis`.


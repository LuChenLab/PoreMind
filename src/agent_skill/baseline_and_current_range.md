# Baseline and Current Range Policy

`detect_direction` and `exclude_current_params` matter more than the threshold. If event direction or the effective current range is wrong, baseline/noise estimation will be wrong and threshold tuning will not be reliable.

## 1. Event Direction

User language mapping:

```text
blockade / current drop / downward peak -> detect_direction = "down"
enhancement / current increase / upward peak -> detect_direction = "up"
```

When uncertain, preview raw trace, denoised trace, current histogram, and candidate baseline/event annotations.

Direction is relative to the baseline, not simply the sign of current. In a negative-current experiment, a move from -100 pA to -80 pA is an upward event.

Direction should be judged from a short selected window plus the current histogram. Full-sweep plots can reveal gross QC problems, but they must not be the main visual basis because long axes, saturation spikes, voltage-switching segments, or unstable periods can hide event shape.

S2 should distinguish a `baseline/noise preview window` from an `event-judgment preview window`. The baseline/noise window can be quiet and stable because it is used for current-range and noise judgement. The event-judgment window must contain enough event-like activity to judge direction and later S4 boundaries. A clean but event-poor window is not enough for event-boundary tuning.

## 2. Meaning of exclude_current

`exclude_current=True` with `exclude_current_params={"min": ..., "max": ...}` defines which current values are included for baseline/noise statistics. In the current workflow implementation, PoreMind builds a statistics mask from the raw/denoised signal and keeps points with `signal > min` and `signal < max`. It does not delete raw data and is not the final event filter.

Do not use `exclude_current_params` as a broad artifact-exclusion range. It must describe the baseline/open-pore current band used for baseline/noise statistics. A range such as `-1000 to 1000 pA` is usually invalid for nanopore event detection because it pools baseline, event states, voltage-switching platforms, unstable segments, and large artifacts into the same statistics set.

Avoid relying on implicit defaults. Prefer explicit min/max values or ask the user to confirm them.

If the user writes a reversed range such as "-50 to -110", normalize it to PoreMind order: `{"min": -110, "max": -50}`. Explain that currents between -110 and -50 pA are included for baseline/noise statistics.

## 3. Choosing the Effective Current Range

Priority:

1. Use a clear user-provided range.
2. Expand a user-provided open-pore range and ask for confirmation.
3. Infer from the histogram peak and trace platforms.
4. If uncertain, stop and ask for confirmation.

The range must be chosen from the accepted `baseline/noise preview window`, not from the full sweep alone. Inspect the raw + denoised trace and the current histogram/density for that short stable window, identify the dominant baseline/open-pore current mode, and choose a narrow band around that mode that covers ordinary baseline fluctuation.

Use valleys between current-state peaks as boundaries when visible. If no clear valley exists, start from a robust local band around the baseline mode, for example the central dense baseline cluster or median +/- 3-5 MAD, then verify with the S4 overlay.

Reject a candidate `exclude_current_params` range when:

- It is hundreds or thousands of pA wide while the visible baseline fluctuation is much narrower.
- It includes both baseline and event-state peaks.
- It includes long zero-current platforms, saturation regions, voltage channels, blocked-pore plateaus, or unstable drift segments.
- It was inferred only from a full-sweep histogram without a short baseline/noise window.
- S4 overlay shows missed events, giant plateau-like events, or an obviously shifted baseline.

For upward events, event-shifted high-current points should stay outside the baseline/noise band when possible. For downward events, event-shifted low-current points should stay outside the baseline/noise band when possible.

Example: if the accepted baseline/noise window shows a stable open-pore current centered near -85 pA and most baseline fluctuation lies between -120 and -60 pA, use `exclude_current_params={"min": -120, "max": -60}`. Do not use `{"min": -1000, "max": 1000}` for the same trace merely because it excludes extreme saturation.

When reporting a selected baseline/current window, state the selected sample/trace/sweep/start/end/duration, why it is representative enough, and whether the user should accept it or provide a different range.

If the user provides a standardized interval such as 2.0-3.0 s, classify it by role: `baseline-only`, `event-rich`, or `balanced`. If it is baseline-only, ask whether to add a separate event-judgment window before S4.

## 4. Default Baseline Strategy

Default method:

```python
baseline_method = "global_quantile"
```

Do not default to rolling baseline. Switch to `rolling_quantile` only when the user reports baseline drift, the preview shows clear drift, segmented medians/quantiles shift strongly, or global baseline gives inconsistent detection across a trace.

## 5. Choosing q

`global_quantile` estimates the baseline as `np.quantile(valid_current_points, q)` after applying the `exclude_current_params` statistics mask. Choose `q` from the event direction and where the baseline/open-pore current sits in the current distribution.

- Upward events with baseline/open-pore current on the lower side of the distribution: start with `q=0.1`. Upward events push current higher, so a lower quantile reduces event-shifted high-current points pulling the baseline upward.
- Downward events with baseline/open-pore current on the higher side of the distribution: start with `q=0.9`. Downward events push current lower, so a higher quantile reduces event-shifted low-current points pulling the baseline downward.
- Sparse events, a central baseline, or unclear distribution shape: use `q=0.5` as the median fallback and revisit after S4 overlay.

Do not describe `q=0.1` or `q=0.9` as arbitrary defaults. Always connect them to event direction and baseline position.

## 6. Rolling Window

For `rolling_quantile`, `window` is in samples, not seconds. Convert from sampling rate. The window should be much longer than an event but shorter than the drift timescale.

If median dwell time is known, start with a window at least 50-200 times the median dwell time times sampling rate.

## 7. User-Facing Wording

Use plain wording:

```text
I am confirming the baseline/noise statistics range, not deleting data. The main open-pore current appears to sit around <range>, while <event states / saturation / switching / unstable region> should not be used for baseline/noise estimation.

I am including currents between <min> and <max> pA for baseline/noise statistics because that band covers the stable baseline/open-pore fluctuation in the accepted baseline/noise window. Then I use `global_quantile` with q=<q>, meaning PoreMind takes the <lower 10% / median / higher 90%> quantile inside that included range as the baseline.

For upward events, q=0.1 is useful when the baseline sits on the lower side because upward events move current higher and should not pull the baseline upward. For downward events, q=0.9 is useful when the baseline sits on the higher side because downward events move current lower and should not pull the baseline downward.

I am using <sample/trace/sweep, start-end> as the main visual window because the full sweep contains <spike/voltage switch/unstable segment/long axis>. This short window looks <clear/limited/not enough> for judging events. Is this window acceptable, or would you like me to move, extend, shorten, or replace it?
```

"""Template: stepwise PoreMind analysis run.

This is not meant to replace agent judgement. The agent should edit this file
with the user's actual paths, groups, event parameters, and accepted feedback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from poremind import create_analysis_object
from save_outputs import append_log, make_event_summary, make_run_dir, save_ax, save_df, write_json


# -----------------------------------------------------------------------------
# 0. USER/AGENT CONFIG — fill these before running.
# -----------------------------------------------------------------------------
STOP_AFTER_STAGE = "S0"  # hard-checkpoint default. Use "ALL" only after Bypass Preflight is explicitly confirmed.

SAMPLE_PATHS = {
    # "sample_1": "data/sample_1.abf",
}
SAMPLE_TO_GROUP = {
    # "sample_1": "group_A",
}
READER = "abf"
READER_KWARGS = None

PREPROCESS = {"method": "butterworth_filtfilt", "params": {"filtfilt_N": 2, "filtfilt_Wn": 0.1}}

# Draft event parameters. The agent should not lock these until preview is accepted.
EVENT_DETECTION = {
    "detect_method": "threshold",
    "detect_params": {"sigma_k": 5.0, "min_duration_s": 0.0, "noise_method": "mad"},
    "baseline_method": "global_quantile",  # skill default
    # Choose q by direction and baseline position:
    # q=0.1 for upward events with lower-side baseline, q=0.9 for downward
    # events with higher-side baseline, q=0.5 when central or uncertain.
    "baseline_params": {"q": 0.9},
    "detect_direction": "down",
    "merge_event": False,
    "merge_event_params": {"merge_gap_ms": 0.0},
    "exclude_current": True,
    # Fill from the accepted baseline/noise window as a narrow
    # baseline/open-pore band, not a broad artifact-exclusion range.
    "exclude_current_params": None,
}

REPRESENTATIVE = {"sample_id": None, "start_ms": 0.0, "end_ms": 1000.0}
WINDOW_ROLES = {
    # Fill during S2. A single window may be classified as baseline-only,
    # event-rich, or balanced, but S4 tuning requires an accepted
    # event-judgment window.
    "baseline_noise_window": {"sample_id": None, "start_ms": 0.0, "end_ms": 1000.0, "role": "baseline/noise"},
    "event_judgment_window": {"sample_id": None, "start_ms": 0.0, "end_ms": 1000.0, "role": "event-judgment"},
}

FEATURE_EXTRACTION = {"custom_feature_fns": None, "max_event_per_sample": None}
WAVEFORM_EMBEDDING = {"enabled": False}
FILTERING = {
    "method": "blockade_gmm",
    "parameters": None,
    # Confirm this in S7 from blockade_ratio feature maps before S8 filtering.
    "blockage_lim": (0.1, 1.0),
}

BLOCKADE_RATIO_PLOT_LIM = (0.0, 1.0)
MODELING = {
    "feature_cols": ["duration_s", "blockade_ratio", "segment_std", "segment_skew", "segment_kurt"],
    "label_col": "label",
    "cv": 10,
    "scoring": "accuracy",
}


# -----------------------------------------------------------------------------
# 1. Run folder and manifest.
# -----------------------------------------------------------------------------
run_dir = make_run_dir()
write_json(run_dir / "00_manifest.json", {"sample_paths": SAMPLE_PATHS, "sample_to_group": SAMPLE_TO_GROUP, "reader": READER})
write_json(
    run_dir / "01_params.draft.json",
    {
        "input": {"sample_paths": SAMPLE_PATHS, "sample_to_group": SAMPLE_TO_GROUP, "reader": READER, "reader_kwargs": READER_KWARGS},
        "preprocess": PREPROCESS,
        "event_detection": EVENT_DETECTION,
        "feature_extraction": {"max_event_per_sample": FEATURE_EXTRACTION["max_event_per_sample"]},
        "filtering": FILTERING,
        "modeling": MODELING,
    },
)
append_log(run_dir / "logs/decisions.md", "S0 init", f"Created run folder: {run_dir}")
append_log(run_dir / "logs/progress.md", "S0 init", f"Created run folder: {run_dir}")
(run_dir / "READY_FOR_USER_CONFIRMATION.txt").write_text(
    "S0 initialization is complete.\n"
    "Stop here and wait for user confirmation before loading data or running S1.\n",
    encoding="utf-8",
)
if STOP_AFTER_STAGE == "S0":
    print(f"S0 complete. Run folder: {run_dir}")
    print("Waiting for user confirmation before S1.")
    sys.exit(0)


# -----------------------------------------------------------------------------
# 2. Load and denoise.
# -----------------------------------------------------------------------------
analysis = create_analysis_object(
    sample_paths=SAMPLE_PATHS,
    sample_to_group=SAMPLE_TO_GROUP,
    reader=READER,
    reader_kwargs=READER_KWARGS,
).load()
analysis.denoise(PREPROCESS["method"], **PREPROCESS["params"])

# Save trace manifest if available.
trace_rows = []
for trace_id, trace in analysis.traces.items():
    trace_rows.append(
        {
            "trace_id": trace_id,
            "sample_id": getattr(trace, "sample_id", None),
            "channel": getattr(trace, "channel", None),
            "sweep": getattr(trace, "sweep", None),
            "sampling_rate_hz": getattr(trace, "sampling_rate_hz", None),
            "n_points": len(getattr(trace, "current", [])),
        }
    )
trace_manifest = pd.DataFrame(trace_rows)
save_df(trace_manifest, run_dir / "s1_input/trace_manifest.csv")
append_log(run_dir / "logs/progress.md", "S1 hard stop", "Data loading and trace manifest are complete. Waiting for user confirmation before S2 preview.")
if STOP_AFTER_STAGE == "S1":
    print(f"S1 complete. Trace manifest: {run_dir / 's1_input/trace_manifest.csv'}")
    print("Waiting for user confirmation before S2.")
    sys.exit(0)

# Preview representative current.
rep_sid = REPRESENTATIVE["sample_id"] or (next(iter(analysis.traces.keys())) if analysis.traces else None)
if rep_sid:
    ax = analysis.pl.current(sample_id=rep_sid, current="denoise", start_ms=REPRESENTATIVE["start_ms"], end_ms=REPRESENTATIVE["end_ms"])
    save_ax(ax, run_dir / "s2_preview/selected_windows_raw_denoised.png")
    save_df(pd.DataFrame(WINDOW_ROLES.values()), run_dir / "s2_preview/window_roles.csv")
append_log(run_dir / "logs/progress.md", "S2 hard stop", "Trace preview is complete. Waiting for user confirmation before S3 direction/current-range decision.")
if STOP_AFTER_STAGE == "S2":
    print(f"S2 complete. Preview: {run_dir / 's2_preview/selected_windows_raw_denoised.png'}")
    print("Waiting for user confirmation before S3.")
    sys.exit(0)


# -----------------------------------------------------------------------------
# 3. Local detection preview. STOP here for user confirmation.
# -----------------------------------------------------------------------------
append_log(
    run_dir / "logs/progress.md",
    "S3 hard stop",
    "Draft direction/current-range/baseline parameters are available in 01_params.draft.json. Waiting for user confirmation before S4 local event preview.",
)
if STOP_AFTER_STAGE == "S3":
    print(f"S3 complete. Params draft: {run_dir / '01_params.draft.json'}")
    print("Waiting for user confirmation before S4.")
    sys.exit(0)

preview_events = analysis.detect_events_simple(
    sample_id=rep_sid,
    current="denoise",
    start_ms=REPRESENTATIVE["start_ms"],
    end_ms=REPRESENTATIVE["end_ms"],
    **EVENT_DETECTION,
)
append_log(
    run_dir / "logs/event_feedback.md",
    "S4 preview generated",
    f"sample_id={rep_sid}\nparams={json.dumps(EVENT_DETECTION, ensure_ascii=False)}\npreview_event_counts={ {k: len(v) for k, v in preview_events.items()} }",
)
append_log(
    run_dir / "logs/progress.md",
    "S4 hard stop",
    "Local event-detection preview has been generated. Waiting for explicit user confirmation before full detection.",
)

if rep_sid:
    ax = analysis.pl.event_current_simple(sample_id=rep_sid, current="denoise", start_event=1, end_event=20)
    save_ax(ax, run_dir / "s4_event_preview/event_overlay.png")

print("\n=== STOP FOR USER FEEDBACK ===")
print(f"Run folder: {run_dir}")
print("Preview figure: s4_event_preview/event_overlay.png")
print("Options: A correct / B missed / C false positives / D wrong direction / E fragmented / F too wide / G drift / H current range / I another window")
(run_dir / "s4_event_preview/READY_FOR_USER_CONFIRMATION.txt").write_text(
    "S4 local event-detection preview is complete.\n"
    "Do not run full detection until the user confirms option A. Only bypass checkpoints after an explicit Bypass Preflight confirmation.\n",
    encoding="utf-8",
)
if STOP_AFTER_STAGE == "S4":
    sys.exit(0)

# After user confirms, an agent should continue with the block below.


# -----------------------------------------------------------------------------
# 4. Full detection, features, filter, model. Run only after user confirmation.
# -----------------------------------------------------------------------------
def continue_after_user_acceptance():
    write_json(run_dir / "02_params.lock.json", {
        "input": {"sample_paths": SAMPLE_PATHS, "sample_to_group": SAMPLE_TO_GROUP, "reader": READER, "reader_kwargs": READER_KWARGS},
        "preprocess": PREPROCESS,
        "event_detection": EVENT_DETECTION,
        "feature_extraction": {"max_event_per_sample": FEATURE_EXTRACTION["max_event_per_sample"]},
        "filtering": FILTERING,
        "modeling": MODELING,
    })

    analysis.detect_events(**EVENT_DETECTION)
    feature_df = analysis.extract_features(**FEATURE_EXTRACTION)
    save_df(feature_df, run_dir / "s6_events/feature_df.csv")
    save_df(make_event_summary(feature_df), run_dir / "s6_events/event_count_by_sample.csv")
    if "blockade_ratio" in feature_df.columns:
        outlier_count = int(((feature_df["blockade_ratio"] < BLOCKADE_RATIO_PLOT_LIM[0]) | (feature_df["blockade_ratio"] > BLOCKADE_RATIO_PLOT_LIM[1])).sum())
        write_json(
            run_dir / "s6_events/blockade_ratio_plot_range.json",
            {
                "display_range": list(BLOCKADE_RATIO_PLOT_LIM),
                "scale": "raw unitless ratio, not percent",
                "events_outside_display_range": outlier_count,
            },
        )

    try:
        ax = analysis.pl.plot_2d(x="blockade_ratio", y="duration_s", data="full", value="label", y_log2=True)
        ax.set_xlim(*BLOCKADE_RATIO_PLOT_LIM)
        ax.set_xlabel("blockade_ratio (raw ratio, not percent)")
        save_ax(ax, run_dir / "s6_events/blockade_ratio_vs_duration.png")
        ax = analysis.pl.plot_2d(x="blockade_ratio", y="segment_std", data="full", value="label", y_log2=False)
        ax.set_xlim(*BLOCKADE_RATIO_PLOT_LIM)
        ax.set_xlabel("blockade_ratio (raw ratio, not percent)")
        save_ax(ax, run_dir / "s6_events/blockade_ratio_vs_segment_std.png")
    except Exception as exc:
        append_log(run_dir / "logs/errors.md", "S7 feature maps failed", str(exc))

    analysis.filter_events(**FILTERING)
    if analysis.filtered_df is not None:
        save_df(analysis.filtered_df, run_dir / "s7_filter/filtered_df.csv")
        if "blockade_ratio" in analysis.filtered_df.columns:
            outlier_count = int(((analysis.filtered_df["blockade_ratio"] < BLOCKADE_RATIO_PLOT_LIM[0]) | (analysis.filtered_df["blockade_ratio"] > BLOCKADE_RATIO_PLOT_LIM[1])).sum())
            write_json(
                run_dir / "s7_filter/blockade_ratio_qc_plot_range.json",
                {
                    "display_range": list(BLOCKADE_RATIO_PLOT_LIM),
                    "scale": "raw unitless ratio, not percent",
                    "filtered_events_outside_display_range": outlier_count,
                },
            )

    try:
        ax = analysis.pl.plot_2d(x="blockade_ratio", y="duration_s", data="filtered", value="label", y_log2=True)
        ax.set_xlim(*BLOCKADE_RATIO_PLOT_LIM)
        ax.set_xlabel("blockade_ratio (raw ratio, not percent)")
        save_ax(ax, run_dir / "s7_filter/filtered_vs_rejected_blockade_ratio_vs_duration.png")
        ax = analysis.pl.plot_2d(x="blockade_ratio", y="segment_std", data="filtered", value="label", y_log2=False)
        ax.set_xlim(*BLOCKADE_RATIO_PLOT_LIM)
        ax.set_xlabel("blockade_ratio (raw ratio, not percent)")
        save_ax(ax, run_dir / "s7_filter/filtered_vs_rejected_blockade_ratio_vs_segment_std.png")
    except Exception as exc:
        append_log(run_dir / "logs/errors.md", "S8 filter maps failed", str(exc))

    try:
        pca_df = analysis.do_pca(data="filtered")
        save_df(pca_df, run_dir / "s8_visualization/pca_coordinates.csv")
        ax = analysis.pl.plot_2d(x="PC1", y="PC2", data="filtered", value="label", y_log2=False)
        save_ax(ax, run_dir / "s8_visualization/pca_projection.png")
    except Exception as exc:
        append_log(run_dir / "logs/errors.md", "S9 PCA failed", str(exc))

    try:
        model_pkg = analysis.build_best_model(**MODELING)
        # Save scores table.
        scores = pd.DataFrame([{"model": k, "score": v} for k, v in model_pkg.get("scores", {}).items()])
        save_df(scores, run_dir / "s9_model/ml_model_metrics.csv")
        ax = analysis.pl.model_metric_bar(metric="accuracy", split="test")
        save_ax(ax, run_dir / "s9_model/ml_model_metrics.png")
        ax = analysis.pl.model_cm(model_pkg["best_model"], split="test")
        save_ax(ax, run_dir / "s9_model/confusion_matrix.png")
    except Exception as exc:
        append_log(run_dir / "logs/errors.md", "modeling failed", str(exc))

    return analysis

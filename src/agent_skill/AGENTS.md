# AGENTS.md - PoreMind Agent Skill Entrypoint

You are working in a project that uses **PoreMind Agent Skill** for single-molecule nanopore analysis.

Before doing nanopore analysis, read and follow:

1. `SKILL.md` - highest-priority behavior rules.
2. `api_registry.md` - current PoreMind API names and parameters.
3. `event_detection_policy.md` - event-detection tuning protocol.
4. `output_contract.md` - required stepwise dialogue format.

Hard rules:

- Every workflow stage is a hard checkpoint. Run only one stage per turn, report outputs and next-step options, then wait for explicit user confirmation before starting the next stage.
- A request such as "complete full analysis", "run the full workflow", or "end-to-end" is only a final goal, not permission to skip checkpoints.
- Before explicit user confirmation of the S4 event overlay, do not run or create-and-run any script that calls full `analysis.detect_events(...)`, `extract_features`, `filter_events`, dimensionality reduction, or ML/DL modeling.
- The only exception is when the user explicitly asks to skip all checkpoints and writes the exact bypass phrase. Do not proactively offer or recommend that phrase in normal next options.
- If the user asks to skip all checkpoints, first show a `Bypass Preflight` response with the full analysis plan and parameter checklist. Run continuously only after that preflight is confirmed; unset parameters must be inferred from the data and logged stage by stage.
- Default language is English. Use Chinese only when the user clearly asks or continues in Chinese; keep only useful technical terms in English.
- Use PoreMind as the primary backend. Do not reimplement PoreMind functionality unless a missing function requires an extension.
- Consult `api_registry.md` for current signatures. DL uses `interp_length=500` and `interp_method="interp"` by default; `MAGJAM`/`MAGJAM_d4` are waveform-only d8/d4 choices, and custom `model` values must be instantiated modules with `forward`.
- Default baseline method is `global_quantile`, not source-code default `rolling_quantile`. Choose `q=0.1` for upward events with lower-side baseline, `q=0.9` for downward events with higher-side baseline, and `q=0.5` when central or uncertain.
- `exclude_current_params` must be a narrow baseline/open-pore current band from the accepted baseline/noise window. Reject broad artifact-exclusion ranges such as full current spans unless the evidence clearly proves they are stable baseline current.
- Treat full-sweep current plots as QC overviews only. For S2, S4, and later waveform/current examples, show a short selected window as the main decision figure, state the selected sample/trace/sweep/start/end/duration, explain why the full window is misleading, and ask the user to accept or change the window.
- In S2, separate a `baseline/noise preview window` from an `event-judgment preview window`. Clean but event-poor windows are not enough for S4 tuning; event-rich but artifact-limited windows need explicit confirmation.
- When PNG figures exist and the chat supports local image display, inline-preview the key figure with Markdown image syntax and still list the file under Outputs.
- In S7, produce `blockade_ratio` vs `duration_s` and `blockade_ratio` vs `segment_std`, with the main displayed `blockade_ratio` axis limited to 0-1 because it is a raw ratio, not percent. Then ask for the blockade-ratio range before S8. In S8, default to native `blockade_gmm` and keep `blockage_lim` on the raw ratio scale. In S9, run PCA only by default and ask before t-SNE/UMAP.
- Save a compact reproducible run folder with `02_params.lock.json`, `reproduce_analysis.py`, stage-specific output folders, and feedback logs.

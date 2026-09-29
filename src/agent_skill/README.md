# PoreMind Agent Skill v0.1

This is a PoreMind analysis skill for code-oriented agents such as Codex, Claude Code, Cursor, Continue, and Aider.

Goal: help users run checkpointed single-molecule nanopore analysis with PoreMind, from ABF/CSV current traces to event detection, feature extraction, event filtering, visualization, ML/DL modeling, optional waveform representations, unknown-sample prediction, and reproducible reports. The current package also exposes the PyTorch MAGJAM classifiers through `build_DL_model`.

## Quick Use

### Codex / OpenAI Codex CLI

Place this directory in the project and keep `AGENTS.md` and `SKILL.md`. Codex should read `AGENTS.md`, which points to `SKILL.md`.

### Claude Code

Keep `CLAUDE.md` and `SKILL.md`. Claude Code should read `CLAUDE.md`, which points to `SKILL.md`.

### Cursor

Reference this directory as project documentation. This repository does not currently include a Cursor `.mdc` adapter file.

### Aider / Continue / Other Agents

Use `SKILL.md` as the highest-priority project instruction, then consult `api_registry.md`, `event_detection_policy.md`, and `output_contract.md`.

## Most Important Rules

1. PoreMind is the primary backend. Do not reimplement existing PoreMind functionality.
2. Every workflow stage is a hard checkpoint. Only bypass checkpoints if the user explicitly asks to skip all checkpoints with the exact bypass phrase; do not proactively offer that phrase in normal next options.
3. If the user asks to skip all checkpoints, first show a `Bypass Preflight` plan and parameter checklist; run continuously only after that preflight is confirmed.
4. Event detection must be previewed locally with `detect_events_simple` before full detection.
5. Default baseline method is `global_quantile`; choose `q=0.1` for upward/lower-side baseline, `q=0.9` for downward/higher-side baseline, and `q=0.5` when central or uncertain.
6. `exclude_current_params` must be a narrow baseline/open-pore current band from the accepted baseline/noise window, not a broad artifact-exclusion range.
7. Every reply uses `Explanation`, `Quick Summary`, `Outputs`, and `Next Options`.
8. Save `02_params.lock.json`, stage outputs, user feedback, and `reproduce_analysis.py`.
9. `PoreMindWaveformEncoder` is an optional S7 representation extension, not a default step and not an event detector.
10. Default response language is English. Reply in Chinese when the user clearly asks or continues in Chinese; keep only useful technical terms in English.
11. Full-sweep current plots are QC overviews only. The main user-facing current figure must be a short selected window, with the selected range, reason, visual clarity, and user confirmation stated explicitly.
12. S7 must produce feature maps for `blockade_ratio` vs `duration_s` and `blockade_ratio` vs `segment_std`; the main displayed `blockade_ratio` axis should be 0-1 because it is a raw ratio, not percent. S8 defaults to `blockade_gmm`; S9 defaults to PCA only.
13. S2 must separate `baseline/noise preview window` from `event-judgment preview window`; clean but event-poor windows are not enough for S4 tuning.
14. When PNG figures exist and the chat supports local images, inline-preview the main stage figure with Markdown image syntax and still list the file under Outputs.

## Current API Synchronization Notes

- The Python workflow defaults to `baseline_method="rolling_quantile"` with `window=10000` and `q=0.5`; the `global_quantile` default above is an Agent Skill policy choice and should be passed explicitly when used.
- `build_DL_model` uses `interp_length=500` and `interp_method="interp"` by default. The additional `"padding"` option zero-pads short segments and center-truncates long segments.
- `build_DL_model(model_name="MAGJAM")` selects the d8 MAGJAM model; `model_name="MAGJAM_d4"` selects d4. Both variants use waveform inputs only and disable handcrafted feature concatenation.
- `model_name="1D-CNN"` is a valid result key for the default/custom convolutional path; it does not identify a separate public class. A custom architecture must be passed as an instantiated `model` object.
- A custom DL extractor must be an instantiated PyTorch module with `forward`, for example `model=one_DCNN()`, not a string name.
- New-sample prediction uses the unified `analysis.classify_new_samples(...)` method for both ML and DL; there is no separate `classify_new_samples_DL` method in the current workflow.

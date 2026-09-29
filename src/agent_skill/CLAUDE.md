# CLAUDE.md - PoreMind Agent Skill Entrypoint

This project uses **PoreMind Agent Skill** for checkpointed single-molecule nanopore analysis.

Read `SKILL.md` first, then consult `api_registry.md`, `event_detection_policy.md`, and `output_contract.md`.

Rules:

- Use PoreMind as the primary backend.
- Consult `api_registry.md` for current signatures. DL uses `interp_length=500` and `interp_method="interp"` by default; `MAGJAM`/`MAGJAM_d4` are waveform-only d8/d4 choices, and custom `model` values must be instantiated modules with `forward`.
- Every workflow stage is a hard checkpoint. Only bypass checkpoints if the user explicitly asks to skip all checkpoints with the exact bypass phrase; do not proactively offer that phrase.
- If the user asks to skip all checkpoints, first show `Bypass Preflight` with the full plan and parameter checklist. Run continuously only after that preflight is confirmed.
- Do not run full event detection until a local `detect_events_simple` preview is accepted.
- Default baseline method is `global_quantile`; choose q by direction and baseline position (`q=0.1` upward/lower-side, `q=0.9` downward/higher-side, `q=0.5` central/uncertain).
- `exclude_current_params` must be a narrow baseline/open-pore current band from the accepted baseline/noise window, not a broad artifact-exclusion range.
- S7 must include 2D feature maps for blockade-ratio range confirmation, with the main displayed `blockade_ratio` axis limited to 0-1 because it is a raw ratio, not percent; S8 defaults to `blockade_gmm`; S9 defaults to PCA only.
- Use English by default. Use Chinese only when the user clearly asks or continues in Chinese.
- Reply with `Explanation`, `Quick Summary`, `Outputs`, and `Next Options`.

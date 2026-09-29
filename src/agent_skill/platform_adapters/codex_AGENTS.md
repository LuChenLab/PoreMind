# AGENTS.md - PoreMind Nanopore Analysis

Use the project PoreMind skill for nanopore analysis.

Rules:

- Read `SKILL.md` first.
- Every workflow stage is a hard checkpoint.
- Run only one stage per turn. Only bypass checkpoints if the user explicitly asks to skip all checkpoints with the exact bypass phrase; do not proactively offer that phrase.
- If the user asks to skip all checkpoints, first reply with `Bypass Preflight`, list the full plan and parameter checklist, and run continuously only after that preflight is confirmed.
- Before full event detection, run `detect_events_simple` and ask for feedback.
- Default baseline method is `global_quantile`; choose q by direction and baseline position (`q=0.1` upward/lower-side, `q=0.9` downward/higher-side, `q=0.5` central/uncertain).
- `exclude_current_params` must be a narrow baseline/open-pore current band from the accepted baseline/noise window, not a broad artifact-exclusion range.
- S7 must include 2D feature maps for blockade-ratio range confirmation, with the main displayed `blockade_ratio` axis limited to 0-1 because it is a raw ratio, not percent; S8 defaults to `blockade_gmm`; S9 defaults to PCA only.
- Save `02_params.lock.json`, stage outputs, logs, and `reproduce_analysis.py`.
- Use English by default; use Chinese when the user clearly asks or continues in Chinese.

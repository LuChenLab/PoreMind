# CLAUDE.md - PoreMind Nanopore Analysis

Use the project PoreMind skill for checkpointed nanopore analysis.

Rules:

- Read `SKILL.md` first.
- Use PoreMind as the backend.
- Every S0-S12 stage is a hard checkpoint.
- If the user asks to skip all checkpoints, first show `Bypass Preflight` with the full plan and parameter checklist; run continuously only after that preflight is confirmed.
- Do not run full event detection before local preview is accepted.
- Reply in English by default; use Chinese when the user clearly asks or continues in Chinese.
- Use `Explanation`, `Quick Summary`, `Outputs`, and `Next Options`.

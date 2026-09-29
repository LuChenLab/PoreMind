# Aider Conventions for PoreMind Analysis

1. Use PoreMind APIs rather than reimplementing analysis logic.
2. Keep one-stage-per-turn checkpoint behavior.
3. Do not run full detection before local event preview is accepted.
4. If the user asks to skip all checkpoints, first show `Bypass Preflight` with the full plan and parameter checklist; run continuously only after that preflight is confirmed.
5. Choose `exclude_current_params` as a narrow baseline/open-pore current band from the accepted baseline/noise window, not a broad artifact-exclusion range.
6. Preserve reproducibility: `02_params.lock.json`, `reproduce_analysis.py`, stage-specific output folders, and logs.
7. Use English by default; use Chinese when the user clearly asks or continues in Chinese.

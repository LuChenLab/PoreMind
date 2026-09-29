# Minimal User Prompt Example

Use PoreMind to analyze these ABF files. Sample groups:

- PP0: data/PP0_01.abf, data/PP0_02.abf
- PP1: data/PP1_01.abf, data/PP1_02.abf
- PP2: data/PP2_01.abf, data/PP2_02.abf

Goal: first perform checkpointed event detection, then feature plots and PP0/PP1/PP2 classification after I approve the checkpoints. The event direction is uncertain, so please inspect representative traces first. Use `global_quantile` as the default baseline; suggest rolling baseline only if drift is visible. For each stage, give me `Explanation`, `Quick Summary`, `Outputs`, and `Next Options`.
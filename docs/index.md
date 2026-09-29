<p><img src="_generated/poremind_logo.png" align="right" width="260" alt="PoreMind"></p>

# PoreMind documentation

[简体中文](zh/index.md)

PoreMind is a Python framework for single-molecule nanopore signal analysis. It converts ABF and CSV current recordings into standardized event-level data and provides signal preprocessing, event detection, feature analysis, machine learning, waveform deep learning, prediction, and visualization through a shared workflow.

## Start here

- [Install PoreMind](installation.md) and its optional readers or model dependencies.
- Follow [Quick Start (API)](_generated/quickstart.ipynb).
- Review the [analysis workflow](workflow.md), [supported models](models.md), or [function reference](functions/README.md).
- Launch the [local Web UI](local-ui.md) for interactive analysis.

## Key features

- **A shared event representation.** Detected events retain waveform data, temporal boundaries, local baseline, sample metadata, and event features so downstream analysis uses the same event records.
- **Several signal-processing choices.** Choose denoising, baseline, detection, and event-quality methods to fit the signal. PCA, t-SNE, and UMAP support exploration of event-feature tables.
- **Classical ML and waveform DL.** Train classical models on event features or use `1D-CNN`, `residual-CNN`, and **MAGJAM** directly on event waveforms. MAGJAM has d8 and d4 variants.
- **Three ways to work.** Use the Python API, local Web UI, or the [PoreMind Agent Skill](https://github.com/LuChenLab/PoreMind/blob/main/src/agent_skill/SKILL.md).

## Analysis in brief

```text
ABF / CSV traces → preprocessing → event detection → event features and QC
                → visualization → ML / DL → prediction for new samples
```

This book documents the implementation in the current repository. The [method framework](nanopore_single_molecule_framework.md) summarizes its data model, supported methods, and current scope.
